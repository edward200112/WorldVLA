"""Rationale text losses."""

from __future__ import annotations

import torch
from torch import nn


class MaskedRationaleTextLoss(nn.Module):
    """Token-level cross-entropy over fixed-length rationale predictions."""

    def __init__(self, pad_token_id: int = 0) -> None:
        super().__init__()
        self.pad_token_id = pad_token_id

    def forward(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        token_mask: torch.Tensor,
        sample_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute a masked token-level rationale loss."""

        if logits.ndim != 3:
            raise ValueError(f"Expected logits with shape [B, L, V], got {tuple(logits.shape)}")
        if not sample_mask.any():
            return logits.new_zeros(())
        vocab_size = logits.shape[-1]
        flat_logits = logits.reshape(-1, vocab_size)
        flat_targets = target_ids.reshape(-1)
        loss = torch.nn.functional.cross_entropy(flat_logits, flat_targets, reduction="none")
        loss = loss.reshape(target_ids.shape)
        effective_mask = token_mask & sample_mask.unsqueeze(-1)
        if not effective_mask.any():
            return logits.new_zeros(())
        return loss[effective_mask].mean()
