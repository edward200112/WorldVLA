from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.action_tokenizer import ActionTokenizer
from models.attention_mask import TokenType, build_selective_attention_mask
from models.backbone_chameleon import DEFAULT_DRIVE_SPECIAL_TOKENS, forward_batch


@dataclass
class PlannerConfig:
    """最小版 unified-token planner 的推理配置。"""

    num_candidates: int = 4
    future_bev_frames: int = 3
    num_bev_tokens_per_frame: int = 16
    bev_codebook_size: int = 256
    temperature: float = 1.0
    top_k: int = 8
    risk_weight: float = 0.4
    progress_weight: float = 0.4
    comfort_weight: float = 0.2


def build_dynamic_token_vocab(prefix: str, vocab_size: int) -> list[str]:
    """为 action 或 BEV 构造统一 token 字符串。"""
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
    """把输入图像统一转成 RGB PIL.Image。"""
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
        raise ValueError(f"图像最后一维必须为 3 或 1，当前收到 {array.shape}")

    if array.dtype != np.uint8:
        if array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(array)


def normalize_history_action_tokens(
    history_action_tokens: Sequence[int | str] | None,
    action_vocab_tokens: Sequence[str],
) -> list[str]:
    """把可选的历史动作 token 统一转成字符串 token 列表。"""
    if history_action_tokens is None:
        return []

    normalized_tokens: list[str] = []
    for token in history_action_tokens:
        if isinstance(token, str):
            normalized_tokens.append(token)
        else:
            token_index = int(token)
            if token_index < 0 or token_index >= len(action_vocab_tokens):
                raise ValueError(
                    f"历史动作 token 越界: index={token_index}, vocab_size={len(action_vocab_tokens)}"
                )
            normalized_tokens.append(action_vocab_tokens[token_index])
    return normalized_tokens


def tokenizer_special_id(tokenizer: Any, token: str) -> int:
    """把 special token 映射成 token id。"""
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None:
        raise ValueError(f"tokenizer 无法识别 token: {token}")
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if unk_token_id is not None and token_id == unk_token_id:
        raise ValueError(f"tokenizer 把 token 识别成了 unk: {token}")
    return int(token_id)


def build_text_token_ids(tokenizer: Any, text: str) -> list[int]:
    """把普通文本片段编码成 token id 列表。"""
    return tokenizer.encode(text, add_special_tokens=False)


def _sample_from_allowed_logits(
    logits: torch.Tensor,
    allowed_token_ids: Sequence[int],
    num_samples: int,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    """在限制词表子集上按温度和 top-k 采样。"""
    allowed_ids_tensor = torch.tensor(list(allowed_token_ids), device=logits.device, dtype=torch.long)
    allowed_logits = logits.index_select(dim=0, index=allowed_ids_tensor)

    if top_k > 0 and top_k < allowed_logits.numel():
        top_values, top_indices = torch.topk(allowed_logits, k=top_k, dim=0)
        candidate_logits = top_values / max(temperature, 1e-6)
        candidate_probs = F.softmax(candidate_logits, dim=0)
        sampled_top = torch.multinomial(candidate_probs, num_samples=num_samples, replacement=True)
        sampled_indices = top_indices.index_select(dim=0, index=sampled_top)
    else:
        candidate_logits = allowed_logits / max(temperature, 1e-6)
        candidate_probs = F.softmax(candidate_logits, dim=0)
        sampled_indices = torch.multinomial(candidate_probs, num_samples=num_samples, replacement=True)

    return allowed_ids_tensor.index_select(dim=0, index=sampled_indices)


def _restrict_argmax(logits: torch.Tensor, allowed_token_ids: Sequence[int]) -> int:
    """在允许 token 子集里做贪心选择。"""
    allowed_ids_tensor = torch.tensor(list(allowed_token_ids), device=logits.device, dtype=torch.long)
    allowed_logits = logits.index_select(dim=0, index=allowed_ids_tensor)
    best_local_index = int(torch.argmax(allowed_logits).item())
    return int(allowed_ids_tensor[best_local_index].item())


def _get_model_device(model: nn.Module) -> torch.device:
    """推断模型当前所在设备。"""
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cpu")


def _build_action_query_sequence(
    tokenizer: Any,
    navigation_text: str,
    history_action_token_strings: Sequence[str],
    num_action_tokens: int,
) -> tuple[list[int], list[int]]:
    """构造用于候选动作采样的 unified-token 输入序列。"""
    input_ids: list[int] = []
    token_types: list[int] = []

    def append_text(text: str) -> None:
        """追加普通文本 token。"""
        ids = build_text_token_ids(tokenizer, text)
        input_ids.extend(ids)
        token_types.extend([int(TokenType.TEXT)] * len(ids))

    def append_special(token: str, token_type: TokenType) -> None:
        """追加单个 special token。"""
        input_ids.append(tokenizer_special_id(tokenizer, token))
        token_types.append(int(token_type))

    append_text(navigation_text.strip())
    append_text("\n前视图:")
    append_special("<image>", TokenType.IMAGE)
    append_text("\n当前BEV:")
    append_special("<image>", TokenType.CURRENT_BEV)

    if len(history_action_token_strings) > 0:
        append_text("\n历史动作摘要:")
        append_special("<ACT_SUMMARY>", TokenType.HISTORY_ACTION_SUMMARY)
        for token in history_action_token_strings:
            append_special(token, TokenType.HISTORY_ACTION_SUMMARY)

    append_text("\n未来动作候选:")
    action_query_positions: list[int] = []
    for _ in range(num_action_tokens):
        action_query_positions.append(len(input_ids))
        append_special("<ACT>", TokenType.FUTURE_ACTION)
    return input_ids, token_types


def _build_bev_query_sequence(
    tokenizer: Any,
    navigation_text: str,
    history_action_token_strings: Sequence[str],
    action_token_strings: Sequence[str],
    future_bev_frames: int,
    num_bev_tokens_per_frame: int,
) -> tuple[list[int], list[int], list[int]]:
    """构造给定动作候选后的 future BEV token 预测序列。"""
    input_ids: list[int] = []
    token_types: list[int] = []

    def append_text(text: str) -> None:
        """追加普通文本 token。"""
        ids = build_text_token_ids(tokenizer, text)
        input_ids.extend(ids)
        token_types.extend([int(TokenType.TEXT)] * len(ids))

    def append_special(token: str, token_type: TokenType) -> None:
        """追加单个 special token。"""
        input_ids.append(tokenizer_special_id(tokenizer, token))
        token_types.append(int(token_type))

    append_text(navigation_text.strip())
    append_text("\n前视图:")
    append_special("<image>", TokenType.IMAGE)
    append_text("\n当前BEV:")
    append_special("<image>", TokenType.CURRENT_BEV)

    if len(history_action_token_strings) > 0:
        append_text("\n历史动作摘要:")
        append_special("<ACT_SUMMARY>", TokenType.HISTORY_ACTION_SUMMARY)
        for token in history_action_token_strings:
            append_special(token, TokenType.HISTORY_ACTION_SUMMARY)

    append_text("\n未来动作:")
    for token in action_token_strings:
        append_special(token, TokenType.FUTURE_ACTION)

    append_text("\n未来BEV:")
    bev_query_positions: list[int] = []
    total_bev_tokens = future_bev_frames * num_bev_tokens_per_frame
    for _ in range(total_bev_tokens):
        bev_query_positions.append(len(input_ids))
        append_special("<BEV>", TokenType.FUTURE_BEV)
    return input_ids, token_types, bev_query_positions


def _prepare_model_inputs(
    processor: Any,
    input_ids: Sequence[int],
    token_types: Sequence[int],
    front_image: Any,
    bev_image: Any,
) -> tuple[dict[str, Any], torch.Tensor]:
    """把 unified-token 序列和两张图像整理成模型输入。"""
    input_tensor = torch.tensor([list(input_ids)], dtype=torch.long)
    token_type_tensor = torch.tensor(token_types, dtype=torch.long)
    selective_mask = build_selective_attention_mask(token_type_tensor)

    images = [to_pil_image(front_image), to_pil_image(bev_image)]
    image_inputs = processor.image_processor(images=images, return_tensors="pt")
    model_inputs = {
        "input_ids": input_tensor,
        "attention_mask": selective_mask,
    }
    model_inputs.update(image_inputs)
    return model_inputs, token_type_tensor


@torch.no_grad()
def _predict_action_candidates(
    model: nn.Module,
    processor: Any,
    navigation_text: str,
    front_image: Any,
    bev_image: Any,
    history_action_token_strings: Sequence[str],
    action_vocab_tokens: Sequence[str],
    num_action_tokens: int,
    num_candidates: int,
    temperature: float,
    top_k: int,
) -> list[list[str]]:
    """使用 backbone 在 action token 子词表上采样 K 个候选动作序列。"""
    tokenizer = processor.tokenizer
    input_ids, token_types = _build_action_query_sequence(
        tokenizer=tokenizer,
        navigation_text=navigation_text,
        history_action_token_strings=history_action_token_strings,
        num_action_tokens=num_action_tokens,
    )
    model_inputs, _ = _prepare_model_inputs(
        processor=processor,
        input_ids=input_ids,
        token_types=token_types,
        front_image=front_image,
        bev_image=bev_image,
    )
    outputs = forward_batch(model, model_inputs, use_cache=False, return_dict=True)
    logits = outputs.logits[0]

    action_vocab_ids = [tokenizer_special_id(tokenizer, token) for token in action_vocab_tokens]
    action_positions = [index for index, token_type in enumerate(token_types) if token_type == int(TokenType.FUTURE_ACTION)]

    candidates: list[list[str]] = [[] for _ in range(num_candidates)]
    for position in action_positions:
        sampled_token_ids = _sample_from_allowed_logits(
            logits=logits[position],
            allowed_token_ids=action_vocab_ids,
            num_samples=num_candidates,
            temperature=temperature,
            top_k=top_k,
        )
        sampled_tokens = tokenizer.convert_ids_to_tokens(sampled_token_ids.tolist())
        for candidate_index in range(num_candidates):
            candidates[candidate_index].append(str(sampled_tokens[candidate_index]))
    return candidates


@torch.no_grad()
def _predict_future_bev_tokens(
    model: nn.Module,
    processor: Any,
    navigation_text: str,
    front_image: Any,
    bev_image: Any,
    history_action_token_strings: Sequence[str],
    candidate_action_tokens: Sequence[str],
    bev_vocab_tokens: Sequence[str],
    future_bev_frames: int,
    num_bev_tokens_per_frame: int,
) -> list[str]:
    """给定一个动作候选，预测未来 3 帧 BEV token。"""
    tokenizer = processor.tokenizer
    input_ids, token_types, bev_query_positions = _build_bev_query_sequence(
        tokenizer=tokenizer,
        navigation_text=navigation_text,
        history_action_token_strings=history_action_token_strings,
        action_token_strings=candidate_action_tokens,
        future_bev_frames=future_bev_frames,
        num_bev_tokens_per_frame=num_bev_tokens_per_frame,
    )
    model_inputs, _ = _prepare_model_inputs(
        processor=processor,
        input_ids=input_ids,
        token_types=token_types,
        front_image=front_image,
        bev_image=bev_image,
    )
    outputs = forward_batch(model, model_inputs, use_cache=False, return_dict=True)
    logits = outputs.logits[0]

    bev_vocab_ids = [tokenizer_special_id(tokenizer, token) for token in bev_vocab_tokens]
    predicted_tokens: list[str] = []
    for position in bev_query_positions:
        predicted_token_id = _restrict_argmax(logits[position], bev_vocab_ids)
        predicted_tokens.append(str(tokenizer.convert_ids_to_tokens(predicted_token_id)))
    return predicted_tokens


def _token_strings_to_indices(tokens: Sequence[str], vocab_tokens: Sequence[str]) -> list[int]:
    """把 token 字符串序列映射回离散索引。"""
    token_to_index = {token: index for index, token in enumerate(vocab_tokens)}
    indices: list[int] = []
    for token in tokens:
        if token not in token_to_index:
            raise ValueError(f"token 不在指定词表中: {token}")
        indices.append(token_to_index[token])
    return indices


def _score_candidate(
    trajectory: torch.Tensor,
    bev_indices: Sequence[int],
    bev_codebook_size: int,
    config: PlannerConfig,
) -> dict[str, float]:
    """用简单启发式规则给候选动作打分。"""
    trajectory = trajectory.detach().cpu()
    final_x = float(trajectory[-1, 0].item())
    progress_score = float(np.clip((final_x + 5.0) / 20.0, 0.0, 1.0))

    if len(bev_indices) == 0:
        occupancy_level = 0.0
    else:
        occupancy_level = float(np.mean(np.asarray(bev_indices, dtype=np.float32) / max(bev_codebook_size - 1, 1)))
    risk_score = float(np.clip(1.0 - occupancy_level, 0.0, 1.0))

    if trajectory.shape[0] >= 2:
        deltas = trajectory[1:] - trajectory[:-1]
        if deltas.shape[0] >= 2:
            jerk_like = torch.mean(torch.abs(deltas[1:] - deltas[:-1])).item()
        else:
            jerk_like = torch.mean(torch.abs(deltas)).item()
    else:
        jerk_like = 0.0
    comfort_score = float(np.clip(1.0 - jerk_like / 2.0, 0.0, 1.0))

    final_score = (
        config.risk_weight * risk_score
        + config.progress_weight * progress_score
        + config.comfort_weight * comfort_score
    )
    return {
        "risk_score": risk_score,
        "progress_score": progress_score,
        "comfort_score": comfort_score,
        "final_score": float(final_score),
    }


def plan_once(
    model: nn.Module,
    processor: Any,
    action_tokenizer: ActionTokenizer,
    navigation_text: str,
    front_image: Any,
    bev_image: Any,
    history_action_tokens: Sequence[int | str] | None = None,
    num_candidates: int = 4,
    future_bev_frames: int = 3,
    num_bev_tokens_per_frame: int = 16,
    bev_codebook_size: int = 256,
    temperature: float = 1.0,
    top_k: int = 8,
) -> dict[str, Any]:
    """执行一次 unified-token 自动驾驶最小版规划。

    输入：
    - navigation_text: 文本导航信息
    - front_image: 前视图图像
    - bev_image: 当前 BEV 图像
    - history_action_tokens: 可选的历史动作 token 序列，可以是索引或字符串

    输出：
    - 返回最佳候选以及所有候选的详细信息
    """
    model.eval()
    action_tokenizer.eval()

    planner_config = PlannerConfig(
        num_candidates=num_candidates,
        future_bev_frames=future_bev_frames,
        num_bev_tokens_per_frame=num_bev_tokens_per_frame,
        bev_codebook_size=bev_codebook_size,
        temperature=temperature,
        top_k=top_k,
    )

    action_vocab_tokens = build_dynamic_token_vocab("ACT", action_tokenizer.codebook_size)
    bev_vocab_tokens = build_dynamic_token_vocab("BEV", bev_codebook_size)
    history_action_token_strings = normalize_history_action_tokens(history_action_tokens, action_vocab_tokens)

    candidate_action_token_strings = _predict_action_candidates(
        model=model,
        processor=processor,
        navigation_text=navigation_text,
        front_image=front_image,
        bev_image=bev_image,
        history_action_token_strings=history_action_token_strings,
        action_vocab_tokens=action_vocab_tokens,
        num_action_tokens=action_tokenizer.num_latent_tokens,
        num_candidates=num_candidates,
        temperature=temperature,
        top_k=top_k,
    )

    action_tokenizer_device = next(action_tokenizer.parameters()).device
    all_candidates: list[dict[str, Any]] = []
    for candidate_tokens in candidate_action_token_strings:
        action_indices = _token_strings_to_indices(candidate_tokens, action_vocab_tokens)
        action_index_tensor = torch.tensor([action_indices], dtype=torch.long, device=action_tokenizer_device)
        recon_trajectory = action_tokenizer.decode_from_indices(action_index_tensor)[0]

        predicted_bev_tokens = _predict_future_bev_tokens(
            model=model,
            processor=processor,
            navigation_text=navigation_text,
            front_image=front_image,
            bev_image=bev_image,
            history_action_token_strings=history_action_token_strings,
            candidate_action_tokens=candidate_tokens,
            bev_vocab_tokens=bev_vocab_tokens,
            future_bev_frames=future_bev_frames,
            num_bev_tokens_per_frame=num_bev_tokens_per_frame,
        )
        predicted_bev_indices = _token_strings_to_indices(predicted_bev_tokens, bev_vocab_tokens)
        score_dict = _score_candidate(
            trajectory=recon_trajectory,
            bev_indices=predicted_bev_indices,
            bev_codebook_size=bev_codebook_size,
            config=planner_config,
        )

        all_candidates.append(
            {
                "action_token_strings": candidate_tokens,
                "action_token_indices": action_indices,
                "predicted_future_bev_tokens": predicted_bev_tokens,
                "predicted_future_bev_indices": predicted_bev_indices,
                "trajectory": recon_trajectory.detach().cpu(),
                **score_dict,
            }
        )

    best_candidate = max(all_candidates, key=lambda candidate: candidate["final_score"])
    return {
        "best_candidate": best_candidate,
        "all_candidates": all_candidates,
    }


class _MockTokenizer:
    """为 planner demo 提供一个最小 tokenizer。"""

    def __init__(self, special_tokens: Sequence[str]) -> None:
        """初始化 mock tokenizer。"""
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self._special_to_id: dict[str, int] = {
            "<pad>": self.pad_token_id,
            "<eos>": self.eos_token_id,
            "<unk>": self.unk_token_id,
            "<image>": 3,
        }
        cursor = 4
        for token in special_tokens:
            if token not in self._special_to_id:
                self._special_to_id[token] = cursor
                cursor += 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """把文本粗糙地编码成稳定 id。"""
        del add_special_tokens
        return [1000 + (ord(char) % 256) for char in text]

    def convert_tokens_to_ids(self, token: str) -> int:
        """把 token 字符串映射到 id。"""
        return self._special_to_id.get(token, self.unk_token_id)

    def convert_ids_to_tokens(self, ids: int | Sequence[int]) -> str | list[str]:
        """把 id 映射回 token 字符串。"""
        id_to_token = {value: key for key, value in self._special_to_id.items()}
        if isinstance(ids, int):
            return id_to_token.get(ids, "<unk>")
        return [id_to_token.get(int(token_id), "<unk>") for token_id in ids]


class _MockImageProcessor:
    """为 planner demo 提供一个最小 image_processor。"""

    def __call__(self, images: Sequence[Any], return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        """把图像转成一个简单的 tensor。"""
        del return_tensors
        tensors = []
        for image in images:
            pil_image = to_pil_image(image).resize((32, 32))
            array = np.asarray(pil_image).astype(np.float32) / 255.0
            tensors.append(torch.tensor(array).permute(2, 0, 1))
        return {"pixel_values": torch.stack(tensors, dim=0)}


class _MockProcessor:
    """为 planner demo 提供一个最小 processor。"""

    def __init__(self, special_tokens: Sequence[str]) -> None:
        """初始化 mock processor。"""
        self.tokenizer = _MockTokenizer(special_tokens)
        self.image_processor = _MockImageProcessor()


class _MockBackbone(nn.Module):
    """为 planner demo 提供一个最小 backbone。"""

    def __init__(self, vocab_size: int) -> None:
        """初始化 mock backbone。"""
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 2048, 64)
        self.head = nn.Linear(64, vocab_size + 2048)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **_: Any) -> Any:
        """返回随机但可重复的 logits。"""
        del attention_mask
        hidden = self.embedding(input_ids)
        logits = self.head(hidden)
        return SimpleNamespace(logits=logits)


if __name__ == "__main__":
    """运行一个伪造输入的最小 demo。"""
    torch.manual_seed(7)
    np.random.seed(7)

    action_tokenizer = ActionTokenizer(
        trajectory_horizon=10,
        action_dim=3,
        hidden_dim=128,
        latent_dim=32,
        num_latent_tokens=4,
        codebook_size=32,
    )

    special_tokens = list(DEFAULT_DRIVE_SPECIAL_TOKENS)
    special_tokens += build_dynamic_token_vocab("ACT", action_tokenizer.codebook_size)
    special_tokens += build_dynamic_token_vocab("BEV", 64)
    processor = _MockProcessor(special_tokens)
    model = _MockBackbone(vocab_size=5000)

    fake_front_image = Image.fromarray(np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8))
    fake_bev_image = Image.fromarray(np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8))

    result = plan_once(
        model=model,
        processor=processor,
        action_tokenizer=action_tokenizer,
        navigation_text="前方路口左转，注意礼让行人并保持低速。",
        front_image=fake_front_image,
        bev_image=fake_bev_image,
        history_action_tokens=[1, 3],
        num_candidates=3,
        future_bev_frames=3,
        num_bev_tokens_per_frame=8,
        bev_codebook_size=64,
        temperature=1.0,
        top_k=6,
    )

    best_candidate = result["best_candidate"]
    print("best action token indices:", best_candidate["action_token_indices"])
    print("best future bev token indices:", best_candidate["predicted_future_bev_indices"][:12], "...")
    print("scores:", {
        "risk_score": round(best_candidate["risk_score"], 4),
        "progress_score": round(best_candidate["progress_score"], 4),
        "comfort_score": round(best_candidate["comfort_score"], 4),
        "final_score": round(best_candidate["final_score"], 4),
    })
    print("trajectory shape:", tuple(best_candidate["trajectory"].shape))
