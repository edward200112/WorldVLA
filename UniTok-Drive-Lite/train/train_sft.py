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
from models.backbone_emu3 import (
    build_model_and_processor,
    forward_batch,
)
from unitok_drive_lite.token_registry import TokenRegistry


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

    parser.add_argument("--model_name", type=str, default="BAAI/Emu3-Chat-hf", help="Hugging Face 模型名。")
    parser.add_argument("--hf_token", type=str, default=None, help="访问 gated 模型时使用的 Hugging Face token。")
    parser.add_argument("--load_in_4bit", dest="load_in_4bit", action="store_true", help="是否使用 4-bit 量化加载 Emu3。")
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
    parser.add_argument("--summary_codebook_size", type=int, default=64, help="历史动作摘要 token 词表大小。")

    parser.add_argument("--bev_codebook_size", type=int, default=256, help="future BEV token 词表大小。")
    parser.add_argument("--bev_patch_size", type=int, default=16, help="从 future_bev_image 离散成 token 时的 patch 大小。")
    parser.add_argument("--future_bev_frames", type=int, default=3, help="未来 BEV 帧数。")

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


def build_token_registry(args: argparse.Namespace) -> TokenRegistry:
    """根据训练参数构造全局固定 token registry。"""
    return TokenRegistry.from_vocab_sizes(
        action_vocab_size=args.action_codebook_size,
        bev_vocab_size=args.bev_codebook_size,
        summary_vocab_size=args.summary_codebook_size,
    )


def build_minimal_batch_example() -> dict[str, list[Any]]:
    """返回一个最小 batch 示例，便于理解当前训练输入格式。"""
    return {
        "text_prompt": ["[NAV] LEFT | SPEED=30 | RISK=PED"],
        "front_image": [np.zeros((224, 224, 3), dtype=np.uint8)],
        "bev_image": [np.zeros((224, 224, 3), dtype=np.uint8)],
        "history_action_summary_indices": [[3]],
        "future_action_indices": [[5, 9, 12, 7]],
        "future_bev_tokens": [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]],
        "future_actions": [np.zeros((10, 3), dtype=np.float32)],
    }


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


def resolve_image_placeholder_id(processor: Any) -> int:
    """解析 Emu3 文本里的图像占位 token id。"""
    tokenizer = processor.tokenizer
    candidate_tokens: list[str] = []
    for token_name in ("image_token", "boi_token"):
        value = getattr(tokenizer, token_name, None)
        if isinstance(value, str):
            candidate_tokens.append(value)
    for token_name in ("image_token",):
        value = getattr(processor, token_name, None)
        if isinstance(value, str):
            candidate_tokens.append(value)
    candidate_tokens.extend(["<image>", "<|image|>", "<|extra_0|>"])

    seen: set[str] = set()
    for token in candidate_tokens:
        if token in seen:
            continue
        seen.add(token)
        token_id = tokenizer.convert_tokens_to_ids(token)
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        if token_id is not None and (unk_token_id is None or token_id != unk_token_id):
            return int(token_id)
    raise ValueError("无法解析 Emu3 的图像占位 token id。")


def resolve_summary_token_strings(
    batch: Mapping[str, list[Any]],
    batch_index: int,
    summary_vocab_tokens: Sequence[str],
) -> list[str]:
    """解析可选的历史动作摘要 token。"""
    if "history_action_summary_tokens" in batch and batch["history_action_summary_tokens"][batch_index] is not None:
        values = batch["history_action_summary_tokens"][batch_index]
        if isinstance(values, (list, tuple)) and all(isinstance(value, str) for value in values):
            return [str(value) for value in values]
        indices = normalize_index_sequence(values)
        return indices_to_token_strings(summary_vocab_tokens, indices, kind="history_action_summary")

    if "history_action_summary_indices" in batch and batch["history_action_summary_indices"][batch_index] is not None:
        indices = normalize_index_sequence(batch["history_action_summary_indices"][batch_index])
        return indices_to_token_strings(summary_vocab_tokens, indices, kind="history_action_summary")

    return []


def build_structured_chat_prompt(
    processor: Any,
    text_prompt: str,
    summary_token_strings: Sequence[str],
) -> str:
    """构造 Emu3 processor 风格的结构化上下文 prompt。"""
    summary_text = " ".join(summary_token_strings) if len(summary_token_strings) > 0 else "<ACT_SUMMARY>"
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "[TASK]\nUNI_DRIVE_SFT"}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"[NAV]\n{text_prompt.strip()}\n[FRONT]\n"},
                {"type": "image"},
                {"type": "text", "text": "\n[BEV_NOW]\n"},
                {"type": "image"},
                {"type": "text", "text": f"\n[ACT_SUM]\n{summary_text}"},
            ],
        },
    ]
    return processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_context_sequence_with_processor(
    processor: Any,
    text_prompt: str,
    front_image: Image.Image,
    bev_image: Image.Image,
    summary_token_strings: Sequence[str],
) -> tuple[list[int], list[int], torch.Tensor, torch.Tensor]:
    """先用 Emu3 processor 编码当前多模态上下文。"""
    prompt = build_structured_chat_prompt(
        processor=processor,
        text_prompt=text_prompt,
        summary_token_strings=summary_token_strings,
    )
    context_inputs = processor(
        text=[prompt],
        images=[front_image, bev_image],
        return_tensors="pt",
    )
    context_input_ids = context_inputs["input_ids"][0].tolist()
    token_types = [int(TokenType.TEXT)] * len(context_input_ids)

    image_placeholder_id = resolve_image_placeholder_id(processor)
    image_positions = [index for index, token_id in enumerate(context_input_ids) if int(token_id) == image_placeholder_id]
    if len(image_positions) < 2:
        raise ValueError("Emu3 上下文中必须至少出现两个图像占位 token。")
    token_types[image_positions[0]] = int(TokenType.IMAGE)
    token_types[image_positions[1]] = int(TokenType.CURRENT_BEV)

    image_sizes = context_inputs.get("image_sizes")
    if image_sizes is None:
        image_sizes = torch.tensor(
            [[front_image.height, front_image.width], [bev_image.height, bev_image.width]],
            dtype=torch.long,
        )
    return context_input_ids, token_types, context_inputs["pixel_values"], image_sizes


def append_text_segment(
    tokenizer: Any,
    input_ids: list[int],
    labels: list[int],
    token_types: list[int],
    text: str,
) -> None:
    """向训练序列尾部追加纯文本片段。"""
    text_ids = build_text_token_ids(tokenizer, text)
    input_ids.extend(text_ids)
    labels.extend([-100] * len(text_ids))
    token_types.extend([int(TokenType.TEXT)] * len(text_ids))


def append_query_targets(
    tokenizer: Any,
    input_ids: list[int],
    labels: list[int],
    token_types: list[int],
    query_token: str,
    target_token_strings: Sequence[str],
    target_token_type: TokenType,
) -> None:
    """追加 query placeholder，并在同一位置监督目标 token。"""
    query_token_id = special_token_id(tokenizer, query_token)
    for token in target_token_strings:
        input_ids.append(query_token_id)
        labels.append(special_token_id(tokenizer, token))
        token_types.append(int(target_token_type))


def build_target_query_suffix(
    tokenizer: Any,
    action_token_strings: Sequence[str],
    bev_token_strings: Sequence[str],
    future_bev_frames: int,
) -> tuple[list[int], list[int], list[int]]:
    """构造未来目标部分的 query placeholder 序列。"""
    input_ids: list[int] = []
    labels: list[int] = []
    token_types: list[int] = []

    append_text_segment(tokenizer, input_ids, labels, token_types, "\n[FUT_ACT]\n")
    append_query_targets(
        tokenizer=tokenizer,
        input_ids=input_ids,
        labels=labels,
        token_types=token_types,
        query_token="<ACT>",
        target_token_strings=action_token_strings,
        target_token_type=TokenType.FUTURE_ACTION,
    )

    if future_bev_frames > 0 and len(bev_token_strings) > 0 and len(bev_token_strings) % future_bev_frames == 0:
        tokens_per_frame = len(bev_token_strings) // future_bev_frames
        frame_slices = [
            list(bev_token_strings[frame_index * tokens_per_frame : (frame_index + 1) * tokens_per_frame])
            for frame_index in range(future_bev_frames)
        ]
    else:
        frame_slices = [list(bev_token_strings)]

    for frame_index, frame_token_strings in enumerate(frame_slices):
        append_text_segment(tokenizer, input_ids, labels, token_types, f"\n[FUT_BEV_{frame_index + 1}]\n")
        append_query_targets(
            tokenizer=tokenizer,
            input_ids=input_ids,
            labels=labels,
            token_types=token_types,
            query_token="<BEV>",
            target_token_strings=frame_token_strings,
            target_token_type=TokenType.FUTURE_BEV,
        )
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
    logits: torch.Tensor,
    labels: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """只在指定目标位置上计算交叉熵损失。"""
    if int(target_mask.sum().item()) == 0:
        return logits.new_zeros(())

    selected_logits = logits[target_mask]
    selected_labels = labels[target_mask]
    return F.cross_entropy(selected_logits, selected_labels)


def build_train_batch(
    batch: Mapping[str, list[Any]],
    processor: Any,
    action_vocab_tokens: Sequence[str],
    bev_vocab_tokens: Sequence[str],
    summary_vocab_tokens: Sequence[str],
    bev_patch_size: int,
    bev_codebook_size: int,
    future_bev_frames: int,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, list[Any]]:
    """把原始 batch 组装成 Emu3 前向所需输入，以及训练标签。"""
    input_id_sequences: list[list[int]] = []
    label_sequences: list[list[int]] = []
    token_type_sequences: list[list[int]] = []
    pixel_values_list: list[torch.Tensor] = []
    image_sizes_list: list[torch.Tensor] = []
    sample_metadata: list[Any] = []

    batch_size = len(batch["text_prompt"])
    for batch_index in range(batch_size):
        text_prompt = str(batch["text_prompt"][batch_index])
        front_image = to_pil_image(batch["front_image"][batch_index])
        bev_image = to_pil_image(batch["bev_image"][batch_index])
        summary_token_strings = resolve_summary_token_strings(
            batch=batch,
            batch_index=batch_index,
            summary_vocab_tokens=summary_vocab_tokens,
        )

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

        context_input_ids, context_token_types, pixel_values, image_sizes = build_context_sequence_with_processor(
            processor=processor,
            text_prompt=text_prompt,
            front_image=front_image,
            bev_image=bev_image,
            summary_token_strings=summary_token_strings,
        )
        target_input_ids, target_labels, target_token_types = build_target_query_suffix(
            tokenizer=processor.tokenizer,
            action_token_strings=action_token_strings,
            bev_token_strings=bev_token_strings,
            future_bev_frames=future_bev_frames,
        )
        input_ids = context_input_ids + target_input_ids
        labels = ([-100] * len(context_input_ids)) + target_labels
        token_types = context_token_types + target_token_types

        input_id_sequences.append(input_ids)
        label_sequences.append(labels)
        token_type_sequences.append(token_types)
        pixel_values_list.append(pixel_values)
        image_sizes_list.append(image_sizes)
        sample_metadata.append(
            {
                "action_indices": action_indices,
                "bev_indices": bev_indices,
                "summary_token_strings": summary_token_strings,
            }
        )

    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_ids = build_padded_tensor(input_id_sequences, tokenizer.pad_token_id)
    labels = build_padded_tensor(label_sequences, -100)
    token_types = build_padded_tensor(token_type_sequences, int(TokenType.PAD))

    selective_mask = build_batched_selective_attention_mask(token_types)

    model_inputs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": selective_mask,
        "pixel_values_list": pixel_values_list,
        "image_sizes_list": image_sizes_list,
    }
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
    token_registry: TokenRegistry,
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
        action_vocab_tokens=token_registry.action_tokens,
        bev_vocab_tokens=token_registry.bev_tokens,
        summary_vocab_tokens=token_registry.summary_tokens,
        bev_patch_size=args.bev_patch_size,
        bev_codebook_size=args.bev_codebook_size,
        future_bev_frames=args.future_bev_frames,
    )

    labels = labels.to(device=device)
    token_types = token_types.to(device=device)
    logits_list: list[torch.Tensor] = []
    batch_size = labels.shape[0]
    for batch_index in range(batch_size):
        sample_inputs = {
            "input_ids": model_inputs["input_ids"][batch_index : batch_index + 1].to(device),
            "attention_mask": model_inputs["attention_mask"][batch_index : batch_index + 1].to(device),
            "pixel_values": model_inputs["pixel_values_list"][batch_index],
            "image_sizes": model_inputs["image_sizes_list"][batch_index],
        }
        outputs = forward_batch(model, sample_inputs, use_cache=False, return_dict=True)
        logits_list.append(outputs.logits)
    logits = torch.cat(logits_list, dim=0)

    action_target_mask = token_types.eq(int(TokenType.FUTURE_ACTION)) & labels.ne(-100)
    bev_target_mask = token_types.eq(int(TokenType.FUTURE_BEV)) & labels.ne(-100)

    loss_action = masked_cross_entropy(logits, labels, action_target_mask)
    loss_bev = masked_cross_entropy(logits, labels, bev_target_mask)

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

    token_registry = build_token_registry(args)

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
        special_tokens=token_registry.all_special_tokens,
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
                    token_registry=token_registry,
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
