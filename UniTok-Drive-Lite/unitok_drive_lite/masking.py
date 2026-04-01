from __future__ import annotations

from models.attention_mask import (
    TOKEN_TYPE_TO_NAME,
    TokenType,
    build_generation_attention_mask as _build_generation_attention_mask,
    build_selective_attention_mask as _build_selective_attention_mask,
    build_selective_attention_visibility,
    print_attention_mask_visualization,
)
import torch


def build_selective_attention_mask(
    attention_mask: torch.Tensor,
    token_types: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """构造 selective 4D additive attention mask。

    说明：
    - `unitok_drive_lite` 主链路历史上把 padding mask 和 token_types 分开传入。
    - 为了避免与顶层 `models/attention_mask.py` 继续分叉，这里统一委托给共享实现。
    """
    return _build_selective_attention_mask(
        token_types=token_types,
        attention_mask=attention_mask,
        dtype=dtype,
        expand_batch_dim=True,
    )


def build_generation_attention_mask(
    attention_mask: torch.Tensor,
    token_types: torch.Tensor,
) -> torch.Tensor:
    """构造 Emu3 `generate(...)` 使用的近似 2D padding mask。"""
    return _build_generation_attention_mask(
        token_types=token_types,
        attention_mask=attention_mask,
    )


__all__ = [
    "TOKEN_TYPE_TO_NAME",
    "TokenType",
    "build_generation_attention_mask",
    "build_selective_attention_mask",
    "build_selective_attention_visibility",
    "print_attention_mask_visualization",
]
