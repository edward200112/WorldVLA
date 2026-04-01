"""Deprecated: 顶层 planner 实验实现。

当前仓库的唯一权威运行主链路是 `scripts/` + `unitok_drive_lite/`。
本文件仅保留给规划算法实验、mock demo 和兼容旧调用使用，不再作为官方推理入口维护。
"""

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
from models.attention_mask import TokenType, build_generation_attention_mask, build_selective_attention_mask
from models.backbone_emu3 import forward_batch
from unitok_drive_lite.token_registry import TokenRegistry


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


def build_token_registry(action_tokenizer: ActionTokenizer, bev_codebook_size: int) -> TokenRegistry:
    """根据 action tokenizer 和 BEV 词表大小构造全局固定 token registry。"""
    return TokenRegistry.from_vocab_sizes(
        action_vocab_size=action_tokenizer.codebook_size,
        bev_vocab_size=bev_codebook_size,
    )


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


def _resolve_image_placeholder_id(processor: Any) -> int:
    """解析 Emu3 prompt 中图像占位 token 的 id。"""
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


def _build_structured_planner_prompt(
    processor: Any,
    navigation_text: str,
    history_action_token_strings: Sequence[str],
    stage_tag: str,
) -> str:
    """构造适配 Emu3 chat template 的结构化 prompt。"""
    summary_tokens = ["<ACT_SUMMARY>"] + list(history_action_token_strings)
    summary_text = " ".join(summary_tokens) if len(summary_tokens) > 0 else "<ACT_SUMMARY>"
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "[TASK]\nUNI_DRIVE_PLAN"}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"[NAV]\n{navigation_text.strip()}\n[FRONT]\n"},
                {"type": "image"},
                {"type": "text", "text": "\n[BEV_NOW]\n"},
                {"type": "image"},
                {"type": "text", "text": f"\n[ACT_SUM]\n{summary_text}\n[STAGE]\n{stage_tag}"},
            ],
        },
    ]
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
    return (
        "[TASK]\nUNI_DRIVE_PLAN\n"
        f"[NAV]\n{navigation_text.strip()}\n"
        "[FRONT]\n<image>\n"
        "[BEV_NOW]\n<image>\n"
        f"[ACT_SUM]\n{summary_text}\n"
        f"[STAGE]\n{stage_tag}\nassistant"
    )


def _encode_multimodal_context(
    processor: Any,
    navigation_text: str,
    front_image: Any,
    bev_image: Any,
    history_action_token_strings: Sequence[str],
    stage_tag: str,
) -> tuple[list[int], list[int], torch.Tensor, torch.Tensor]:
    """把当前文本、前视图、当前 BEV 编码成 Emu3 上下文。"""
    tokenizer = processor.tokenizer
    front_pil = to_pil_image(front_image)
    bev_pil = to_pil_image(bev_image)
    prompt = _build_structured_planner_prompt(
        processor=processor,
        navigation_text=navigation_text,
        history_action_token_strings=history_action_token_strings,
        stage_tag=stage_tag,
    )

    if callable(processor):
        context_inputs = processor(
            text=[prompt],
            images=[front_pil, bev_pil],
            return_tensors="pt",
        )
    else:
        context_inputs = {
            "input_ids": torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], dtype=torch.long),
        }
        context_inputs.update(processor.image_processor(images=[front_pil, bev_pil], return_tensors="pt"))

    context_input_ids = context_inputs["input_ids"][0].tolist()
    token_types = [int(TokenType.TEXT)] * len(context_input_ids)

    image_placeholder_id = _resolve_image_placeholder_id(processor)
    image_positions = [index for index, token_id in enumerate(context_input_ids) if int(token_id) == image_placeholder_id]
    if len(image_positions) < 2:
        raise ValueError("planner 上下文中必须至少包含两个图像占位 token。")
    token_types[image_positions[0]] = int(TokenType.IMAGE)
    token_types[image_positions[1]] = int(TokenType.CURRENT_BEV)

    summary_token_ids: set[int] = set()
    for token in ["<ACT_SUMMARY>", *history_action_token_strings]:
        token_id = tokenizer.convert_tokens_to_ids(token)
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        if token_id is not None and (unk_token_id is None or token_id != unk_token_id):
            summary_token_ids.add(int(token_id))
    for index, token_id in enumerate(context_input_ids):
        if int(token_id) in summary_token_ids:
            token_types[index] = int(TokenType.HISTORY_ACTION_SUMMARY)

    image_sizes = context_inputs.get("image_sizes")
    if image_sizes is None:
        image_sizes = torch.tensor(
            [[front_pil.height, front_pil.width], [bev_pil.height, bev_pil.width]],
            dtype=torch.long,
        )
    return context_input_ids, token_types, context_inputs["pixel_values"], image_sizes


def _prepare_model_inputs(
    input_ids: Sequence[int],
    token_types: Sequence[int],
    pixel_values: torch.Tensor,
    image_sizes: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor]:
    """把 unified-token 序列和图像张量整理成 Emu3 输入。"""
    input_tensor = torch.tensor([list(input_ids)], dtype=torch.long)
    token_type_tensor = torch.tensor(token_types, dtype=torch.long)
    selective_mask = build_selective_attention_mask(token_type_tensor)
    generation_attention_mask = build_generation_attention_mask(token_type_tensor)

    model_inputs = {
        "input_ids": input_tensor,
        "attention_mask": selective_mask,
        "generation_attention_mask": generation_attention_mask,
        "pixel_values": pixel_values,
        "image_sizes": image_sizes,
    }
    return model_inputs, token_type_tensor


def _append_text_segment(
    tokenizer: Any,
    input_ids: list[int],
    token_types: list[int],
    text: str,
) -> None:
    """向序列尾部追加结构化文本片段。"""
    text_ids = build_text_token_ids(tokenizer, text)
    input_ids.extend(text_ids)
    token_types.extend([int(TokenType.TEXT)] * len(text_ids))


def _append_special_token(
    tokenizer: Any,
    input_ids: list[int],
    token_types: list[int],
    token: str,
    token_type: TokenType,
) -> None:
    """向序列尾部追加一个 special token。"""
    input_ids.append(tokenizer_special_id(tokenizer, token))
    token_types.append(int(token_type))


def _build_action_query_sequence(
    tokenizer: Any,
    context_input_ids: Sequence[int],
    context_token_types: Sequence[int],
    num_action_tokens: int,
) -> tuple[list[int], list[int], list[int]]:
    """在上下文后追加动作 query placeholder。"""
    input_ids = list(context_input_ids)
    token_types = list(context_token_types)
    _append_text_segment(tokenizer, input_ids, token_types, "\n[FUT_ACT]\n")
    action_query_positions: list[int] = []
    for _ in range(num_action_tokens):
        action_query_positions.append(len(input_ids))
        _append_special_token(tokenizer, input_ids, token_types, "<ACT>", TokenType.FUTURE_ACTION)
    return input_ids, token_types, action_query_positions


def _build_bev_query_sequence(
    tokenizer: Any,
    context_input_ids: Sequence[int],
    context_token_types: Sequence[int],
    action_token_strings: Sequence[str],
    future_bev_frames: int,
    num_bev_tokens_per_frame: int,
) -> tuple[list[int], list[int], list[int]]:
    """在上下文和动作候选后追加 future BEV query placeholder。"""
    input_ids = list(context_input_ids)
    token_types = list(context_token_types)
    _append_text_segment(tokenizer, input_ids, token_types, "\n[FUT_ACT]\n")
    for token in action_token_strings:
        _append_special_token(tokenizer, input_ids, token_types, token, TokenType.FUTURE_ACTION)

    bev_query_positions: list[int] = []
    for frame_index in range(future_bev_frames):
        _append_text_segment(tokenizer, input_ids, token_types, f"\n[FUT_BEV_{frame_index + 1}]\n")
        for _ in range(num_bev_tokens_per_frame):
            bev_query_positions.append(len(input_ids))
            _append_special_token(tokenizer, input_ids, token_types, "<BEV>", TokenType.FUTURE_BEV)
    return input_ids, token_types, bev_query_positions


@torch.no_grad()
def generate_action_candidates(
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
    """第一阶段：生成 K 个未来 action token 候选。

    说明：
    - 这里仍然使用 unified-token query placeholder 思路。
    - 由于 selective attention 会阻断 future action token 之间的可见性，
      因此一次前向即可在每个动作 query 位置独立采样。
    """
    tokenizer = processor.tokenizer
    context_input_ids, context_token_types, pixel_values, image_sizes = _encode_multimodal_context(
        processor=processor,
        navigation_text=navigation_text,
        front_image=front_image,
        bev_image=bev_image,
        history_action_token_strings=history_action_token_strings,
        stage_tag="GEN_ACT",
    )
    input_ids, token_types, action_query_positions = _build_action_query_sequence(
        tokenizer=tokenizer,
        context_input_ids=context_input_ids,
        context_token_types=context_token_types,
        num_action_tokens=num_action_tokens,
    )
    model_inputs, _ = _prepare_model_inputs(
        input_ids=input_ids,
        token_types=token_types,
        pixel_values=pixel_values,
        image_sizes=image_sizes,
    )
    outputs = forward_batch(model, model_inputs, use_cache=False, return_dict=True)
    logits = outputs.logits[0]

    action_vocab_ids = [tokenizer_special_id(tokenizer, token) for token in action_vocab_tokens]
    candidates: list[list[str]] = [[] for _ in range(num_candidates)]
    for position in action_query_positions:
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
def rollout_future_bev(
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
    """第二阶段：给定动作候选，rollout future BEV token。"""
    tokenizer = processor.tokenizer
    context_input_ids, context_token_types, pixel_values, image_sizes = _encode_multimodal_context(
        processor=processor,
        navigation_text=navigation_text,
        front_image=front_image,
        bev_image=bev_image,
        history_action_token_strings=history_action_token_strings,
        stage_tag="ROLL_BEV",
    )
    input_ids, token_types, bev_query_positions = _build_bev_query_sequence(
        tokenizer=tokenizer,
        context_input_ids=context_input_ids,
        context_token_types=context_token_types,
        action_token_strings=candidate_action_tokens,
        future_bev_frames=future_bev_frames,
        num_bev_tokens_per_frame=num_bev_tokens_per_frame,
    )
    bev_vocab_ids = [tokenizer_special_id(tokenizer, token) for token in bev_vocab_tokens]
    rollout_input_ids = list(input_ids)
    predicted_tokens: list[str] = []
    for position in bev_query_positions:
        model_inputs, _ = _prepare_model_inputs(
            input_ids=rollout_input_ids,
            token_types=token_types,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
        )
        outputs = forward_batch(model, model_inputs, use_cache=False, return_dict=True)
        predicted_token_id = _restrict_argmax(outputs.logits[0, position], bev_vocab_ids)
        rollout_input_ids[position] = predicted_token_id
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


def _run_candidate_scorer(
    trajectory: torch.Tensor,
    bev_indices: Sequence[int],
    planner_config: PlannerConfig,
    scorer: Any | None,
) -> dict[str, float]:
    """执行候选打分器，默认回退到启发式 scorer。"""
    if scorer is None:
        return _score_candidate(
            trajectory=trajectory,
            bev_indices=bev_indices,
            bev_codebook_size=planner_config.bev_codebook_size,
            config=planner_config,
        )

    if isinstance(scorer, nn.Module):
        scorer.eval()
        with torch.no_grad():
            raw_score = scorer(
                trajectory=trajectory.unsqueeze(0),
                bev_indices=torch.tensor([list(bev_indices)], dtype=torch.long, device=trajectory.device),
                bev_codebook_size=planner_config.bev_codebook_size,
                config=planner_config,
            )
    else:
        raw_score = scorer(
            trajectory=trajectory,
            bev_indices=bev_indices,
            bev_codebook_size=planner_config.bev_codebook_size,
            config=planner_config,
        )

    if not isinstance(raw_score, Mapping):
        raise TypeError("自定义 scorer 必须返回包含分数字段的 dict。")

    normalized_score: dict[str, float] = {}
    for key, value in raw_score.items():
        if torch.is_tensor(value):
            normalized_score[key] = float(value.detach().reshape(-1)[0].item())
        else:
            normalized_score[key] = float(value)

    if "final_score" not in normalized_score:
        required_keys = {"risk_score", "progress_score", "comfort_score"}
        if not required_keys.issubset(normalized_score):
            raise ValueError("自定义 scorer 至少要返回 `final_score`，或同时返回 risk/progress/comfort 三项。")
        normalized_score["final_score"] = (
            planner_config.risk_weight * normalized_score["risk_score"]
            + planner_config.progress_weight * normalized_score["progress_score"]
            + planner_config.comfort_weight * normalized_score["comfort_score"]
        )

    normalized_score.setdefault("risk_score", 0.0)
    normalized_score.setdefault("progress_score", 0.0)
    normalized_score.setdefault("comfort_score", 0.0)
    return normalized_score


def score_candidates(
    action_tokenizer: ActionTokenizer,
    candidate_action_token_strings: Sequence[Sequence[str]],
    candidate_future_bev_token_strings: Sequence[Sequence[str]],
    token_registry: TokenRegistry,
    planner_config: PlannerConfig,
    scorer: Any | None = None,
) -> list[dict[str, Any]]:
    """第三阶段：对候选动作进行打分。

    说明：
    - 默认使用当前启发式 scorer。
    - 通过 `scorer` 参数可替换成 learned scorer，只要返回分数字典即可。
    """
    if len(candidate_action_token_strings) != len(candidate_future_bev_token_strings):
        raise ValueError("动作候选数量和 future BEV rollout 数量必须一致。")

    action_tokenizer_device = next(action_tokenizer.parameters()).device
    scored_candidates: list[dict[str, Any]] = []
    for candidate_tokens, bev_tokens in zip(candidate_action_token_strings, candidate_future_bev_token_strings):
        action_indices = _token_strings_to_indices(candidate_tokens, token_registry.action_tokens)
        action_index_tensor = torch.tensor([action_indices], dtype=torch.long, device=action_tokenizer_device)
        trajectory = action_tokenizer.decode_from_indices(action_index_tensor)[0]
        predicted_bev_indices = _token_strings_to_indices(bev_tokens, token_registry.bev_tokens)
        score_dict = _run_candidate_scorer(
            trajectory=trajectory,
            bev_indices=predicted_bev_indices,
            planner_config=planner_config,
            scorer=scorer,
        )
        scored_candidates.append(
            {
                "action_token_strings": list(candidate_tokens),
                "action_token_indices": action_indices,
                "predicted_future_bev_tokens": list(bev_tokens),
                "predicted_future_bev_indices": predicted_bev_indices,
                "trajectory": trajectory.detach().cpu(),
                **score_dict,
            }
        )
    return scored_candidates


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
    scorer: Any | None = None,
) -> dict[str, Any]:
    """执行一次两阶段 unified-token 自动驾驶最小版规划。

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

    token_registry = build_token_registry(action_tokenizer, bev_codebook_size)
    history_action_token_strings = normalize_history_action_tokens(history_action_tokens, token_registry.action_tokens)

    candidate_action_token_strings = generate_action_candidates(
        model=model,
        processor=processor,
        navigation_text=navigation_text,
        front_image=front_image,
        bev_image=bev_image,
        history_action_token_strings=history_action_token_strings,
        action_vocab_tokens=token_registry.action_tokens,
        num_action_tokens=action_tokenizer.num_latent_tokens,
        num_candidates=num_candidates,
        temperature=temperature,
        top_k=top_k,
    )

    candidate_future_bev_token_strings: list[list[str]] = []
    for candidate_tokens in candidate_action_token_strings:
        predicted_bev_tokens = rollout_future_bev(
            model=model,
            processor=processor,
            navigation_text=navigation_text,
            front_image=front_image,
            bev_image=bev_image,
            history_action_token_strings=history_action_token_strings,
            candidate_action_tokens=candidate_tokens,
            bev_vocab_tokens=token_registry.bev_tokens,
            future_bev_frames=future_bev_frames,
            num_bev_tokens_per_frame=num_bev_tokens_per_frame,
        )
        candidate_future_bev_token_strings.append(predicted_bev_tokens)

    all_candidates = score_candidates(
        action_tokenizer=action_tokenizer,
        candidate_action_token_strings=candidate_action_token_strings,
        candidate_future_bev_token_strings=candidate_future_bev_token_strings,
        token_registry=token_registry,
        planner_config=planner_config,
        scorer=scorer,
    )
    best_candidate = max(all_candidates, key=lambda candidate: candidate["final_score"])
    return {
        "best_candidate": best_candidate,
        "all_candidates": all_candidates,
        "action_candidates": candidate_action_token_strings,
        "bev_rollouts": candidate_future_bev_token_strings,
    }


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
    """兼容旧调用：转发到新的 `generate_action_candidates(...)`。"""
    return generate_action_candidates(
        model=model,
        processor=processor,
        navigation_text=navigation_text,
        front_image=front_image,
        bev_image=bev_image,
        history_action_token_strings=history_action_token_strings,
        action_vocab_tokens=action_vocab_tokens,
        num_action_tokens=num_action_tokens,
        num_candidates=num_candidates,
        temperature=temperature,
        top_k=top_k,
    )


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
    """兼容旧调用：转发到新的 `rollout_future_bev(...)`。"""
    return rollout_future_bev(
        model=model,
        processor=processor,
        navigation_text=navigation_text,
        front_image=front_image,
        bev_image=bev_image,
        history_action_token_strings=history_action_token_strings,
        candidate_action_tokens=candidate_action_tokens,
        bev_vocab_tokens=bev_vocab_tokens,
        future_bev_frames=future_bev_frames,
        num_bev_tokens_per_frame=num_bev_tokens_per_frame,
    )


class _MockTokenizer:
    """为 planner demo 提供一个最小 tokenizer。"""

    def __init__(self, special_tokens: Sequence[str]) -> None:
        """初始化 mock tokenizer。"""
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        self.image_token = "<image>"
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
        self._text_to_id: dict[str, int] = {}
        self._next_text_id = cursor + 1024

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """把文本粗糙地编码成稳定 id。"""
        del add_special_tokens
        normalized_text = text.replace("\n", " \n ")
        for token in sorted(self._special_to_id, key=len, reverse=True):
            normalized_text = normalized_text.replace(token, f" {token} ")

        token_ids: list[int] = []
        for piece in normalized_text.split():
            if piece in self._special_to_id:
                token_ids.append(self._special_to_id[piece])
                continue
            if piece not in self._text_to_id:
                self._text_to_id[piece] = self._next_text_id
                self._next_text_id += 1
            token_ids.append(self._text_to_id[piece])
        return token_ids

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
        return {
            "pixel_values": torch.stack(tensors, dim=0),
            "image_sizes": torch.tensor([[32, 32] for _ in images], dtype=torch.long),
        }


class _MockProcessor:
    """为 planner demo 提供一个最小 processor。"""

    def __init__(self, special_tokens: Sequence[str]) -> None:
        """初始化 mock processor。"""
        self.tokenizer = _MockTokenizer(special_tokens)
        self.image_processor = _MockImageProcessor()
        self.image_token = "<image>"

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, Any]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        """把多模态对话模板转成简单字符串。"""
        del tokenize
        parts: list[str] = []
        for message in conversation:
            parts.append(message["role"])
            for item in message["content"]:
                if item["type"] == "text":
                    parts.append(str(item["text"]))
                elif item["type"] == "image":
                    parts.append("<image>")
        if add_generation_prompt:
            parts.append("assistant")
        return " ".join(parts)

    def __call__(
        self,
        text: Sequence[str],
        images: Sequence[Any],
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        """模拟 Emu3Processor 的 `text + images` 多模态输入接口。"""
        del return_tensors
        if len(text) != 1:
            raise ValueError("mock processor 当前只支持单条文本输入。")
        input_ids = torch.tensor([self.tokenizer.encode(text[0], add_special_tokens=False)], dtype=torch.long)
        image_inputs = self.image_processor(images=images, return_tensors="pt")
        return {
            "input_ids": input_ids,
            "pixel_values": image_inputs["pixel_values"],
            "image_sizes": image_inputs["image_sizes"],
        }


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

    token_registry = build_token_registry(action_tokenizer, bev_codebook_size=64)
    processor = _MockProcessor(token_registry.all_special_tokens)
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
