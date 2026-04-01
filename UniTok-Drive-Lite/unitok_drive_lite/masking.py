from __future__ import annotations

import torch


def build_selective_attention_mask(
    attention_mask: torch.Tensor,
    role_ids: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """构造 selective 4D additive attention mask。"""
    batch_size, seq_length = attention_mask.shape
    device = attention_mask.device

    positions = torch.arange(seq_length, device=device)
    causal = positions.unsqueeze(0) <= positions.unsqueeze(1)
    causal = causal.unsqueeze(0).expand(batch_size, -1, -1)

    valid = attention_mask.to(torch.bool)
    valid_pairs = valid.unsqueeze(1) & valid.unsqueeze(2)
    allowed = causal & valid_pairs

    is_raw_action = role_ids.eq(1)
    raw_action_pairs = is_raw_action.unsqueeze(1) & is_raw_action.unsqueeze(2)
    allowed = allowed & ~raw_action_pairs

    additive_mask = torch.zeros((batch_size, 1, seq_length, seq_length), dtype=dtype, device=device)
    additive_mask = additive_mask.masked_fill(~allowed.unsqueeze(1), torch.finfo(dtype).min)
    return additive_mask
