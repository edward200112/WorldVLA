"""Behavior classification losses."""

from __future__ import annotations

import torch
from torch import nn


class MaskedBehaviorCrossEntropyLoss(nn.Module):
    """Cross-entropy over behavior labels with sample masking."""

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute a masked behavior classification loss."""

        if logits.ndim != 2:
            raise ValueError(f"Expected logits with shape [B, C], got {tuple(logits.shape)}")
        if not valid_mask.any():
            return logits.new_zeros(())
        masked_logits = logits[valid_mask]
        masked_targets = targets[valid_mask]
        return torch.nn.functional.cross_entropy(masked_logits, masked_targets)
