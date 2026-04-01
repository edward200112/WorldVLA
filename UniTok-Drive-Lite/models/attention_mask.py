from __future__ import annotations

from enum import IntEnum

import torch


class TokenType(IntEnum):
    """统一 token 序列里各类 token 的类别编号。"""

    TEXT = 0
    IMAGE = 1
    CURRENT_BEV = 2
    HISTORY_ACTION_SUMMARY = 3
    FUTURE_ACTION = 4
    FUTURE_BEV = 5
    PAD = 6


TOKEN_TYPE_TO_NAME = {
    TokenType.TEXT: "TEXT",
    TokenType.IMAGE: "IMAGE",
    TokenType.CURRENT_BEV: "CUR_BEV",
    TokenType.HISTORY_ACTION_SUMMARY: "HIST_ACT",
    TokenType.FUTURE_ACTION: "FUT_ACT",
    TokenType.FUTURE_BEV: "FUT_BEV",
    TokenType.PAD: "PAD",
}


def _normalize_token_types(token_types: torch.Tensor) -> torch.Tensor:
    """把 token_types 归一化成 [L] long 张量。"""
    if token_types.ndim == 2:
        if token_types.shape[0] != 1:
            raise ValueError(f"当前只支持单条样本的 token_types，收到 {tuple(token_types.shape)}")
        token_types = token_types[0]
    if token_types.ndim != 1:
        raise ValueError(f"token_types 的 shape 应为 [L]，当前收到 {tuple(token_types.shape)}")
    return token_types.to(dtype=torch.long)


def _normalize_valid_tokens(
    token_types: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """把可见 token 掩码归一化成 [L] bool 张量。"""
    if attention_mask is None:
        return token_types.ne(int(TokenType.PAD))

    if attention_mask.ndim == 2:
        if attention_mask.shape[0] != 1:
            raise ValueError(f"当前只支持单条样本的 attention_mask，收到 {tuple(attention_mask.shape)}")
        attention_mask = attention_mask[0]
    if attention_mask.ndim != 1:
        raise ValueError(f"attention_mask 的 shape 应为 [L]，当前收到 {tuple(attention_mask.shape)}")
    if attention_mask.shape[0] != token_types.shape[0]:
        raise ValueError("attention_mask 与 token_types 的长度必须一致。")
    return attention_mask.to(torch.bool) & token_types.ne(int(TokenType.PAD))


def _build_base_causal_visibility(
    token_types: torch.Tensor,
    valid_tokens: torch.Tensor,
) -> torch.Tensor:
    """构造基础的二维 causal 可见性矩阵。

    输入：
    - token_types: [L]
    - valid_tokens: [L]

    输出：
    - allowed: [L, L]，其中 allowed[i, j] 表示第 i 个 query 是否可以看到第 j 个 key
    """
    sequence_length = token_types.shape[0]
    positions = torch.arange(sequence_length, device=token_types.device)

    # 行表示 query 位置 i，列表示 key 位置 j。
    # causal 规则要求 j <= i。
    allowed = positions.unsqueeze(0) <= positions.unsqueeze(1)

    # PAD token 不参与注意力。
    allowed = allowed & valid_tokens.unsqueeze(0) & valid_tokens.unsqueeze(1)
    return allowed


def _build_future_action_context_mask(token_types: torch.Tensor) -> torch.Tensor:
    """为 future action query 构造允许访问的 key 类型掩码。

    规则：
    - future action token 只允许看到：
      1. 文本 token
      2. 当前图像 token
      3. 当前 BEV token
      4. 历史动作 summary token
      5. 自己当前位置的对角线
    - 因此它不能看到本 chunk 中更早的 raw action token，也不会看到其他未来 token。

    输入：
    - token_types: [L]

    输出：
    - action_specific_allowed: [L, L]
    """
    sequence_length = token_types.shape[0]
    device = token_types.device
    positions = torch.arange(sequence_length, device=device)

    is_future_action_query = token_types.eq(int(TokenType.FUTURE_ACTION)).unsqueeze(1)
    is_text_key = token_types.eq(int(TokenType.TEXT))
    is_image_key = token_types.eq(int(TokenType.IMAGE))
    is_current_bev_key = token_types.eq(int(TokenType.CURRENT_BEV))
    is_history_action_summary_key = token_types.eq(int(TokenType.HISTORY_ACTION_SUMMARY))

    # 允许 future action query 看到的上下文 key 类型。
    allowed_context_keys = (
        is_text_key
        | is_image_key
        | is_current_bev_key
        | is_history_action_summary_key
    ).unsqueeze(0)

    # 对角线允许每个 future action query 看到自己当前位置。
    diagonal = positions.unsqueeze(0) == positions.unsqueeze(1)

    action_specific_allowed = (~is_future_action_query) | allowed_context_keys | diagonal
    return action_specific_allowed


def build_selective_attention_visibility(
    token_types: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """构造 selective attention 的二维可见性矩阵。

    输入：
    - token_types: [L]，每个位置是一个整数类别编号，类别定义见 TokenType
    - attention_mask: [L] 或 [1, L]，1 表示有效 token，0 表示 padding；可选

    输出：
    - visibility: [L, L]，True 表示可见，False 表示不可见

    注意：
    - 行表示 query，列表示 key
    """
    token_types = _normalize_token_types(token_types)
    valid_tokens = _normalize_valid_tokens(token_types, attention_mask)
    base_causal_allowed = _build_base_causal_visibility(token_types, valid_tokens)
    future_action_specific_allowed = _build_future_action_context_mask(token_types)
    return base_causal_allowed & future_action_specific_allowed


def build_selective_attention_mask(
    token_types: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float32,
    expand_batch_dim: bool = True,
) -> torch.Tensor:
    """构造适用于 Hugging Face causal LM 的 additive selective attention mask。

    输入：
    - token_types: [L]，每个位置是一个整数类别编号，类别定义见 TokenType
    - attention_mask: [L] 或 [1, L]，1 表示有效 token，0 表示 padding；可选
    - dtype: 输出 additive mask 的浮点类型
    - expand_batch_dim: 是否扩成 [1, 1, L, L]

    输出：
    - 若 expand_batch_dim=True，返回 [1, 1, L, L]
    - 否则返回 [L, L]

    注意：
    - 这是训练 / 全序列前向最稳妥的接法，可以直接送进 Emu3 的 `forward(...)`
    - 对 `generate(...)` 而言，Transformers 通常更稳定地支持 2D padding mask，因此不建议直接复用 4D mask
    """
    token_types = _normalize_token_types(token_types)
    final_allowed = build_selective_attention_visibility(token_types, attention_mask=attention_mask)

    additive_mask = torch.zeros(
        (token_types.shape[0], token_types.shape[0]),
        dtype=dtype,
        device=token_types.device,
    )
    additive_mask = additive_mask.masked_fill(~final_allowed, torch.finfo(dtype).min)
    if expand_batch_dim:
        additive_mask = additive_mask.unsqueeze(0).unsqueeze(0)
    return additive_mask


def build_generation_attention_mask(
    token_types: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """构造给 `generate(...)` 使用的 2D padding mask。

    说明：
    - 这里只保留“有效 token / padding”信息，不表达 selective action 约束。
    - 如果推理阶段必须严格保持 future action 互不可见，应改用 `forward(...)` 手工逐步解码，
      或像当前 planner 一样直接做 query 位置打分，而不是依赖 `model.generate(...)`。
    """
    token_types = _normalize_token_types(token_types)
    valid_tokens = _normalize_valid_tokens(token_types, attention_mask)
    return valid_tokens.to(dtype=torch.long).unsqueeze(0)


def infer_padding_mask_from_additive_attention_mask(mask: torch.Tensor) -> torch.Tensor:
    """从 additive 4D mask 反推出 2D padding mask。

    说明：
    - 该函数只提取“哪些 token 有效”，不会保留 selective visibility 细节。
    - 主要用于 Emu3 `generate(...)` 的近似回退路径。
    """
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"mask 必须是 [B, 1, L, L]，当前收到 {tuple(mask.shape)}")
    diagonal = torch.diagonal(mask[:, 0], dim1=-2, dim2=-1)
    return diagonal.eq(0).to(dtype=torch.long)


def _mask_to_boolean_visibility(mask: torch.Tensor) -> torch.Tensor:
    """把 4D additive mask 转回布尔可见矩阵，便于调试打印。"""
    if mask.ndim == 4:
        if mask.shape[0] != 1 or mask.shape[1] != 1:
            raise ValueError(f"当前只支持 [1, 1, L, L] 的可视化，收到 {tuple(mask.shape)}")
        return mask[0, 0].eq(0)
    if mask.ndim == 2:
        return mask.eq(0)
    raise ValueError(f"mask 必须是 [L, L] 或 [1, 1, L, L]，当前收到 {tuple(mask.shape)}")


def print_attention_mask_visualization(
    token_types: torch.Tensor,
    mask: torch.Tensor | None = None,
    max_tokens: int = 48,
) -> None:
    """打印 attention mask 的可视化结果。

    打印约定：
    - `.` 表示该 query 可以看到该 key
    - `x` 表示该 query 不能看到该 key
    """
    if token_types.ndim != 1:
        raise ValueError(f"token_types 的 shape 应为 [L]，当前收到 {tuple(token_types.shape)}")

    if mask is None:
        mask = build_selective_attention_mask(token_types)

    visibility = _mask_to_boolean_visibility(mask)
    sequence_length = token_types.shape[0]
    display_length = min(sequence_length, max_tokens)

    print("Token Types:")
    for index in range(display_length):
        token_type = TokenType(int(token_types[index].item()))
        print(f"  idx={index:02d} type={TOKEN_TYPE_TO_NAME[token_type]}")

    if display_length < sequence_length:
        print(f"  ... 其余 {sequence_length - display_length} 个 token 省略")

    print("\nMask Visualization:")
    header = "qry\\key " + " ".join(f"{index:02d}" for index in range(display_length))
    print(header)

    for query_index in range(display_length):
        row_symbols = ["." if bool(visibility[query_index, key_index].item()) else "x" for key_index in range(display_length)]
        print(f"{query_index:02d}      " + "  ".join(row_symbols))


if __name__ == "__main__":
    """运行一个最小示例，展示 selective mask 的可视化效果。"""
    demo_token_types = torch.tensor(
        [
            TokenType.TEXT,
            TokenType.TEXT,
            TokenType.IMAGE,
            TokenType.IMAGE,
            TokenType.CURRENT_BEV,
            TokenType.CURRENT_BEV,
            TokenType.HISTORY_ACTION_SUMMARY,
            TokenType.FUTURE_ACTION,
            TokenType.FUTURE_ACTION,
            TokenType.FUTURE_ACTION,
            TokenType.FUTURE_BEV,
            TokenType.FUTURE_BEV,
        ],
        dtype=torch.long,
    )

    demo_mask = build_selective_attention_mask(demo_token_types)
    print("mask shape:", tuple(demo_mask.shape))
    print_attention_mask_visualization(demo_token_types, demo_mask)
