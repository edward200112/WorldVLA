from __future__ import annotations

import argparse
import json
import random
import sys
from itertools import chain
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.action_tokenizer import ActionTokenizer
from models.attention_mask import TokenType, build_selective_attention_mask
from models.backbone_chameleon import (
    DEFAULT_DRIVE_SPECIAL_TOKENS,
    build_model_and_processor,
    forward_batch,
)


class TensorListDataset(Dataset):
    """从 `torch.load` 得到的样本列表中读取数据。"""

    def __init__(self, samples: Sequence[Mapping[str, Any]]) -> None:
        """初始化数据集。"""
        self.samples = list(samples)
        if len(self.samples) == 0:
            raise ValueError("训练数据为空。")

    def __len__(self) -> int:
        """返回样本总数。"""
        return len(self.samples)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        """返回一条样本字典。"""
        return self.samples[index]


def parse_args() -> argparse.Namespace:
    """解析训练脚本参数。"""
    parser = argparse.ArgumentParser(description="训练 unified-token 自动驾驶最小版 SFT 模型。")

    parser.add_argument("--data_path", type=str, required=True, help="训练数据路径，要求是 torch.save 保存的样本列表。")
    parser.add_argument("--output_dir", type=str, default="outputs/train_sft", help="checkpoint 输出目录。")

    parser.add_argument("--model_name", type=str, default="facebook/chameleon-7b", help="Hugging Face 模型名。")
    parser.add_argument("--hf_token", type=str, default=None, help="访问 gated 模型时使用的 Hugging Face token。")
    parser.add_argument("--load_in_4bit", dest="load_in_4bit", action="store_true", help="是否使用 4-bit 量化加载 Chameleon。")
    parser.add_argument("--no_load_in_4bit", dest="load_in_4bit", action="store_false", help="显式关闭 4-bit 量化加载。")
    parser.set_defaults(load_in_4bit=True)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device_map", type=str, default="auto", help="传给 Hugging Face 的 device_map。")
    parser.add_argument("--attn_implementation", type=str, default="eager", help="注意力实现方式。")

    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank。")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha。")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout。")
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        help="LoRA 作用的模块名列表。",
    )
    parser.add_argument("--gradient_checkpointing", action="store_true", help="是否开启 gradient checkpointing。")

    parser.add_argument("--trajectory_horizon", type=int, default=10, help="未来轨迹长度 T。")
    parser.add_argument("--action_dim", type=int, default=3, help="未来轨迹最后一维，默认是 (x, y, yaw)。")
    parser.add_argument("--action_hidden_dim", type=int, default=256, help="ActionTokenizer MLP 隐层维度。")
    parser.add_argument("--action_latent_dim", type=int, default=64, help="ActionTokenizer latent 维度。")
    parser.add_argument("--num_action_latent_tokens", type=int, default=4, help="每条轨迹压缩出的 action token 数。")
    parser.add_argument("--action_codebook_size", type=int, default=512, help="action codebook 大小。")
    parser.add_argument("--action_commitment_cost", type=float, default=0.25, help="VQ tokenizer commitment cost。")

    parser.add_argument("--bev_codebook_size", type=int, default=256, help="future BEV token 词表大小。")
    parser.add_argument("--bev_patch_size", type=int, default=16, help="从 future_bev_image 离散成 token 时的 patch 大小。")

    parser.add_argument("--num_epochs", type=int, default=1, help="训练 epoch 数。")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size。")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker 数。")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率。")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减。")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪阈值。")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累计步数。")
    parser.add_argument("--amp_dtype", type=str, default="bfloat16", choices=["none", "float16", "bfloat16"])

    parser.add_argument("--lambda_bev", type=float, default=1.0, help="future BEV token loss 的权重。")
    parser.add_argument("--lambda_recon", type=float, default=1.0, help="action tokenizer recon loss 的权重。")

    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument("--log_every", type=int, default=1, help="每隔多少个 micro step 打印一次日志。")
    parser.add_argument("--save_every", type=int, default=100, help="每隔多少个 optimizer step 保存一次 checkpoint。")

    return parser.parse_args()


def seed_everything(seed: int) -> None:
    """固定随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_samples(data_path: Path) -> list[Mapping[str, Any]]:
    """读取训练样本列表。"""
    data = torch.load(data_path, map_location="cpu")
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "samples" in data and isinstance(data["samples"], list):
        return data["samples"]
    raise ValueError("data_path 必须保存为 list[dict]，或 dict 且包含 `samples` 字段。")


def simple_collate(samples: Sequence[Mapping[str, Any]]) -> dict[str, list[Any]]:
    """把样本列表整理成按字段聚合的字典。"""
    keys: set[str] = set()
    for sample in samples:
        keys.update(sample.keys())
    return {key: [sample.get(key) for sample in samples] for key in keys}


def build_dynamic_token_vocab(prefix: str, vocab_size: int) -> list[str]:
    """为 action 或 BEV 生成一整套离散 token 字符串。"""
    if vocab_size <= 1:
        raise ValueError("vocab_size 必须大于 1。")
    width = max(4, len(str(vocab_size - 1)))
    return [f"<{prefix}_{index:0{width}d}>" for index in range(vocab_size)]


def _to_numpy_array(value: Any) -> np.ndarray:
    """把 PIL / numpy / torch 输入统一转成 numpy 数组。"""
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, Image.Image):
        return np.asarray(value)
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    raise TypeError(f"不支持的数据类型: {type(value)}")


def to_pil_image(image: Any) -> Image.Image:
    """把输入图像转成 PIL.Image.Image。"""
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    array = _to_numpy_array(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    elif array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    elif array.ndim != 3:
        raise ValueError(f"图像 shape 不合法: {array.shape}")

    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] != 3:
        raise ValueError(f"图像最后一维必须是 3 或 1，当前收到 {array.shape}")

    if array.dtype != np.uint8:
        if array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(array)


def to_tensor(data: Any, dtype: torch.dtype) -> torch.Tensor:
    """把输入数据转成指定 dtype 的 torch.Tensor。"""
    if torch.is_tensor(data):
        return data.to(dtype=dtype)
    return torch.tensor(np.asarray(data), dtype=dtype)


def normalize_index_sequence(indices: Any) -> list[int]:
    """把一条离散 token 序列归一化成 Python 整数列表。"""
    if indices is None:
        raise ValueError("离散 token 序列不能为空。")
    if torch.is_tensor(indices):
        return indices.detach().cpu().view(-1).to(torch.long).tolist()
    if isinstance(indices, np.ndarray):
        return indices.reshape(-1).astype(np.int64).tolist()
    if isinstance(indices, (list, tuple)):
        return [int(value) for value in indices]
    raise TypeError(f"不支持的索引类型: {type(indices)}")


def indices_to_token_strings(vocab_tokens: Sequence[str], indices: Sequence[int], kind: str) -> list[str]:
    """把离散索引序列映射成对应的 special token 字符串。"""
    token_strings: list[str] = []
    for index in indices:
        if index < 0 or index >= len(vocab_tokens):
            raise ValueError(f"{kind} 索引越界: index={index}, vocab_size={len(vocab_tokens)}")
        token_strings.append(vocab_tokens[index])
    return token_strings


def convert_future_bev_image_to_tokens(
    future_bev_image: Any,
    bev_patch_size: int,
    bev_codebook_size: int,
) -> list[int]:
    """把 future_bev_image 用简单 patch 平均量化成 future BEV token 序列。"""
    if isinstance(future_bev_image, (list, tuple)):
        frames = [_to_numpy_array(frame) for frame in future_bev_image]
    else:
        array = _to_numpy_array(future_bev_image)

        if array.ndim == 2:
            frames = [array]
        elif array.ndim == 3:
            if array.shape[0] in (1, 3):
                frames = [np.transpose(array, (1, 2, 0))]
            elif array.shape[-1] in (1, 3):
                frames = [array]
            else:
                frames = [frame for frame in array]
        elif array.ndim == 4:
            if array.shape[-1] in (1, 3):
                frames = [frame for frame in array]
            elif array.shape[1] in (1, 3):
                frames = [np.transpose(frame, (1, 2, 0)) for frame in array]
            else:
                raise ValueError(f"future_bev_image shape 不合法: {array.shape}")
        else:
            raise ValueError(f"future_bev_image shape 不合法: {array.shape}")

    token_ids: list[int] = []
    for frame in frames:
        frame_array = frame.astype(np.float32)
        if frame_array.ndim == 3:
            frame_gray = frame_array.mean(axis=-1)
        else:
            frame_gray = frame_array

        if frame_gray.max() > 1.0:
            frame_gray = frame_gray / 255.0

        height, width = frame_gray.shape
        if height % bev_patch_size != 0 or width % bev_patch_size != 0:
            raise ValueError("future_bev_image 的高宽必须能被 bev_patch_size 整除。")

        grid_h = height // bev_patch_size
        grid_w = width // bev_patch_size
        reshaped = frame_gray.reshape(grid_h, bev_patch_size, grid_w, bev_patch_size)
        pooled = reshaped.mean(axis=(1, 3))
        discrete = np.rint(pooled * (bev_codebook_size - 1)).astype(np.int64)
        token_ids.extend(discrete.reshape(-1).tolist())

    return token_ids


def build_text_token_ids(tokenizer: Any, text: str) -> list[int]:
    """把普通文本片段编码成 token id 列表。"""
    return tokenizer.encode(text, add_special_tokens=False)


def special_token_id(tokenizer: Any, token: str) -> int:
    """把 special token 字符串映射成对应的 token id。"""
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None:
        raise ValueError(f"tokenizer 无法识别 token: {token}")
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if unk_token_id is not None and token_id == unk_token_id:
        raise ValueError(f"tokenizer 把 token 识别成了 unk: {token}")
    return int(token_id)


def build_unified_token_sequence(
    tokenizer: Any,
    text_prompt: str,
    action_token_strings: Sequence[str],
    bev_token_strings: Sequence[str],
) -> tuple[list[int], list[int], list[int]]:
    """把单条样本组织成 unified-token 训练序列。

    输出：
    - input_ids: [L]
    - labels: [L]，只有 future action 与 future BEV 目标 token 位置保留监督，其余位置为 -100
    - token_types: [L]
    """
    image_token = "<image>"
    act_token = "<ACT>"
    bev_token = "<BEV>"

    input_ids: list[int] = []
    token_types: list[int] = []

    def append_text(text: str) -> None:
        """追加普通文本 token。"""
        ids = build_text_token_ids(tokenizer, text)
        input_ids.extend(ids)
        token_types.extend([int(TokenType.TEXT)] * len(ids))

    def append_special(token: str, token_type: TokenType) -> None:
        """追加单个 special token。"""
        input_ids.append(special_token_id(tokenizer, token))
        token_types.append(int(token_type))

    append_text(text_prompt.strip())
    append_text("\n前视图:")
    append_special(image_token, TokenType.IMAGE)
    append_text("\n当前 BEV:")
    append_special(image_token, TokenType.CURRENT_BEV)
    append_text("\n未来动作 token:")
    append_special(act_token, TokenType.TEXT)
    for token in action_token_strings:
        append_special(token, TokenType.FUTURE_ACTION)
    append_text("\n未来 BEV token:")
    append_special(bev_token, TokenType.TEXT)
    for token in bev_token_strings:
        append_special(token, TokenType.FUTURE_BEV)

    labels = [-100] * len(input_ids)
    for index, token_type in enumerate(token_types):
        if token_type in (int(TokenType.FUTURE_ACTION), int(TokenType.FUTURE_BEV)):
            labels[index] = input_ids[index]
    return input_ids, labels, token_types


def build_padded_tensor(sequences: Sequence[Sequence[int]], pad_value: int) -> torch.Tensor:
    """把变长一维序列 pad 成二维张量。"""
    max_length = max(len(sequence) for sequence in sequences)
    padded = [list(sequence) + [pad_value] * (max_length - len(sequence)) for sequence in sequences]
    return torch.tensor(padded, dtype=torch.long)


def build_batched_selective_attention_mask(token_types: torch.Tensor) -> torch.Tensor:
    """把 [B, L] 的 token_types 扩展成 [B, 1, L, L] 的 selective mask。"""
    masks = [build_selective_attention_mask(sample_token_types) for sample_token_types in token_types]
    return torch.cat(masks, dim=0)


def masked_cross_entropy(
    shift_logits: torch.Tensor,
    shift_labels: torch.Tensor,
    shift_target_mask: torch.Tensor,
) -> torch.Tensor:
    """只在指定目标位置上计算交叉熵损失。"""
    if int(shift_target_mask.sum().item()) == 0:
        return shift_logits.new_zeros(())

    selected_logits = shift_logits[shift_target_mask]
    selected_labels = shift_labels[shift_target_mask]
    return F.cross_entropy(selected_logits, selected_labels)


def build_train_batch(
    batch: Mapping[str, list[Any]],
    processor: Any,
    action_vocab_tokens: Sequence[str],
    bev_vocab_tokens: Sequence[str],
    bev_patch_size: int,
    bev_codebook_size: int,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, list[Any]]:
    """把原始 batch 组装成 Chameleon 前向所需输入，以及训练标签。"""
    input_id_sequences: list[list[int]] = []
    label_sequences: list[list[int]] = []
    token_type_sequences: list[list[int]] = []
    flat_images: list[Image.Image] = []
    sample_metadata: list[Any] = []

    batch_size = len(batch["text_prompt"])
    for batch_index in range(batch_size):
        text_prompt = str(batch["text_prompt"][batch_index])
        front_image = to_pil_image(batch["front_image"][batch_index])
        bev_image = to_pil_image(batch["bev_image"][batch_index])

        if "future_action_indices" in batch and batch["future_action_indices"][batch_index] is not None:
            action_indices = normalize_index_sequence(batch["future_action_indices"][batch_index])
        else:
            action_indices = []

        if "future_bev_tokens" in batch and batch["future_bev_tokens"][batch_index] is not None:
            bev_indices = normalize_index_sequence(batch["future_bev_tokens"][batch_index])
        elif "future_bev_image" in batch and batch["future_bev_image"][batch_index] is not None:
            bev_indices = convert_future_bev_image_to_tokens(
                future_bev_image=batch["future_bev_image"][batch_index],
                bev_patch_size=bev_patch_size,
                bev_codebook_size=bev_codebook_size,
            )
        else:
            raise ValueError("batch 里必须包含 future_bev_tokens 或 future_bev_image。")

        action_token_strings = indices_to_token_strings(action_vocab_tokens, action_indices, kind="action")
        bev_token_strings = indices_to_token_strings(bev_vocab_tokens, bev_indices, kind="future_bev")

        input_ids, labels, token_types = build_unified_token_sequence(
            tokenizer=processor.tokenizer,
            text_prompt=text_prompt,
            action_token_strings=action_token_strings,
            bev_token_strings=bev_token_strings,
        )

        input_id_sequences.append(input_ids)
        label_sequences.append(labels)
        token_type_sequences.append(token_types)
        flat_images.extend([front_image, bev_image])
        sample_metadata.append(
            {
                "action_indices": action_indices,
                "bev_indices": bev_indices,
            }
        )

    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_ids = build_padded_tensor(input_id_sequences, tokenizer.pad_token_id)
    labels = build_padded_tensor(label_sequences, -100)
    token_types = build_padded_tensor(token_type_sequences, int(TokenType.PAD))

    image_inputs = processor.image_processor(images=flat_images, return_tensors="pt")
    selective_mask = build_batched_selective_attention_mask(token_types)

    model_inputs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": selective_mask,
    }
    model_inputs.update(image_inputs)
    return model_inputs, labels, token_types, sample_metadata


def collect_future_actions(batch: Mapping[str, list[Any]], device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
    """从 batch 中收集连续未来轨迹。"""
    if "future_actions" not in batch:
        return None
    values = batch["future_actions"]
    if any(value is None for value in values):
        return None
    stacked = torch.stack([to_tensor(value, dtype=dtype) for value in values], dim=0)
    return stacked.to(device=device)


def resolve_action_indices(
    batch: Mapping[str, list[Any]],
    action_tokenizer_outputs: Mapping[str, torch.Tensor] | None,
) -> list[list[int]]:
    """优先使用 batch 自带的 future_action_indices，否则退回到 action tokenizer 当前编码结果。"""
    if "future_action_indices" in batch and all(value is not None for value in batch["future_action_indices"]):
        return [normalize_index_sequence(value) for value in batch["future_action_indices"]]

    if action_tokenizer_outputs is None:
        raise ValueError("缺少 future_action_indices，且 batch 中也没有 future_actions 可用于现算 action indices。")

    indices = action_tokenizer_outputs["indices"].detach().cpu()
    return [row.tolist() for row in indices]


def compute_losses(
    model: nn.Module,
    processor: Any,
    action_tokenizer: ActionTokenizer,
    batch: Mapping[str, list[Any]],
    action_vocab_tokens: Sequence[str],
    bev_vocab_tokens: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """对一个 batch 执行前向并计算三项损失。"""
    future_actions = collect_future_actions(batch, device=device, dtype=torch.float32)
    action_tokenizer_outputs: Mapping[str, torch.Tensor] | None = None
    if future_actions is not None:
        action_tokenizer_outputs = action_tokenizer(future_actions)

    if args.lambda_recon > 0.0 and future_actions is None:
        raise ValueError("当 lambda_recon > 0 时，batch 里必须包含 `future_actions`。")

    action_indices = resolve_action_indices(batch, action_tokenizer_outputs)
    batch_for_model = dict(batch)
    batch_for_model["future_action_indices"] = action_indices

    model_inputs, labels, token_types, _ = build_train_batch(
        batch=batch_for_model,
        processor=processor,
        action_vocab_tokens=action_vocab_tokens,
        bev_vocab_tokens=bev_vocab_tokens,
        bev_patch_size=args.bev_patch_size,
        bev_codebook_size=args.bev_codebook_size,
    )

    labels = labels.to(device=device)
    token_types = token_types.to(device=device)
    outputs = forward_batch(model, model_inputs, use_cache=False, return_dict=True)
    logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_token_types = token_types[:, 1:].contiguous()

    action_target_mask = shift_token_types.eq(int(TokenType.FUTURE_ACTION)) & shift_labels.ne(-100)
    bev_target_mask = shift_token_types.eq(int(TokenType.FUTURE_BEV)) & shift_labels.ne(-100)

    loss_action = masked_cross_entropy(shift_logits, shift_labels, action_target_mask)
    loss_bev = masked_cross_entropy(shift_logits, shift_labels, bev_target_mask)

    if action_tokenizer_outputs is not None:
        loss_recon = action_tokenizer_outputs["recon_loss"]
    else:
        loss_recon = logits.new_zeros(())

    total_loss = loss_action + args.lambda_bev * loss_bev + args.lambda_recon * loss_recon
    return {
        "loss": total_loss,
        "loss_action": loss_action,
        "loss_bev": loss_bev,
        "loss_recon": loss_recon,
    }


def save_checkpoint(
    checkpoint_dir: Path,
    model: nn.Module,
    processor: Any,
    action_tokenizer: ActionTokenizer,
    optimizer: AdamW,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    micro_step: int,
    optimizer_step: int,
    args: argparse.Namespace,
) -> None:
    """保存最小训练 checkpoint。"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "save_pretrained"):
        model.save_pretrained(checkpoint_dir / "model")
    else:
        torch.save(model.state_dict(), checkpoint_dir / "model.pt")

    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(checkpoint_dir / "processor")

    torch.save(action_tokenizer.state_dict(), checkpoint_dir / "action_tokenizer.pt")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "micro_step": micro_step,
            "optimizer_step": optimizer_step,
            "args": vars(args),
        },
        checkpoint_dir / "trainer_state.pt",
    )

    with (checkpoint_dir / "meta.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "epoch": epoch,
                "micro_step": micro_step,
                "optimizer_step": optimizer_step,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )


def get_primary_device(model: nn.Module) -> torch.device:
    """推断模型主要设备。"""
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cpu")


def count_trainable_parameters(modules: Iterable[nn.Module]) -> tuple[int, int]:
    """统计总参数量和可训练参数量。"""
    total_parameters = 0
    trainable_parameters = 0
    seen: set[int] = set()

    for module in modules:
        for parameter in module.parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            total_parameters += parameter.numel()
            if parameter.requires_grad:
                trainable_parameters += parameter.numel()
    return total_parameters, trainable_parameters


def main() -> None:
    """运行最小 SFT 训练主循环。"""
    args = parse_args()
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"未找到数据文件: {data_path}")

    action_vocab_tokens = build_dynamic_token_vocab("ACT", args.action_codebook_size)
    bev_vocab_tokens = build_dynamic_token_vocab("BEV", args.bev_codebook_size)
    all_special_tokens = list(DEFAULT_DRIVE_SPECIAL_TOKENS) + action_vocab_tokens + bev_vocab_tokens

    model, processor = build_model_and_processor(
        model_name=args.model_name,
        load_in_4bit=args.load_in_4bit,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        use_lora=True,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        add_drive_special_tokens=True,
        special_tokens=all_special_tokens,
        gradient_checkpointing=args.gradient_checkpointing,
        token=args.hf_token,
    )
    model.train()

    primary_device = get_primary_device(model)
    action_tokenizer = ActionTokenizer(
        trajectory_horizon=args.trajectory_horizon,
        action_dim=args.action_dim,
        hidden_dim=args.action_hidden_dim,
        latent_dim=args.action_latent_dim,
        num_latent_tokens=args.num_action_latent_tokens,
        codebook_size=args.action_codebook_size,
        commitment_cost=args.action_commitment_cost,
    ).to(primary_device)
    action_tokenizer.train()

    samples = load_samples(data_path)
    dataset = TensorListDataset(samples)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=simple_collate,
    )

    trainable_parameters = [
        parameter
        for parameter in chain(model.parameters(), action_tokenizer.parameters())
        if parameter.requires_grad
    ]
    optimizer = AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    amp_enabled = primary_device.type == "cuda" and args.amp_dtype != "none"
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(args.amp_dtype, torch.float32)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    total_parameters, total_trainable_parameters = count_trainable_parameters([model, action_tokenizer])
    print(f"[model] total_parameters={total_parameters:,}")
    print(f"[model] trainable_parameters={total_trainable_parameters:,}")
    print(f"[train] primary_device={primary_device}")
    print(f"[train] load_in_4bit={args.load_in_4bit}")

    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    optimizer_step = 0

    for epoch in range(args.num_epochs):
        for batch in dataloader:
            micro_step += 1

            with torch.autocast(
                device_type=primary_device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                loss_dict = compute_losses(
                    model=model,
                    processor=processor,
                    action_tokenizer=action_tokenizer,
                    batch=batch,
                    action_vocab_tokens=action_vocab_tokens,
                    bev_vocab_tokens=bev_vocab_tokens,
                    args=args,
                    device=primary_device,
                )
                scaled_loss = loss_dict["loss"] / args.gradient_accumulation_steps

            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if micro_step % args.gradient_accumulation_steps == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    list(chain(model.parameters(), action_tokenizer.parameters())),
                    args.max_grad_norm,
                )

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

                if args.save_every > 0 and optimizer_step % args.save_every == 0:
                    save_checkpoint(
                        checkpoint_dir=output_dir / f"checkpoint_step_{optimizer_step:06d}",
                        model=model,
                        processor=processor,
                        action_tokenizer=action_tokenizer,
                        optimizer=optimizer,
                        scaler=scaler,
                        epoch=epoch,
                        micro_step=micro_step,
                        optimizer_step=optimizer_step,
                        args=args,
                    )

            if args.log_every > 0 and micro_step % args.log_every == 0:
                print(
                    f"[train] epoch={epoch + 1} "
                    f"micro_step={micro_step} "
                    f"optimizer_step={optimizer_step} "
                    f"total_loss={loss_dict['loss'].detach().item():.6f} "
                    f"action_loss={loss_dict['loss_action'].detach().item():.6f} "
                    f"bev_loss={loss_dict['loss_bev'].detach().item():.6f} "
                    f"recon_loss={loss_dict['loss_recon'].detach().item():.6f}"
                )

    save_checkpoint(
        checkpoint_dir=output_dir / "checkpoint_last",
        model=model,
        processor=processor,
        action_tokenizer=action_tokenizer,
        optimizer=optimizer,
        scaler=scaler,
        epoch=args.num_epochs - 1,
        micro_step=micro_step,
        optimizer_step=optimizer_step,
        args=args,
    )
    print(f"[save] checkpoint_dir={output_dir / 'checkpoint_last'}")


if __name__ == "__main__":
    main()
