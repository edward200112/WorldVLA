"""Preference/ranking loss stub reserved for future trajectory preference data."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class PreferenceRankingLoss(nn.Module):
    """Optional ranking loss placeholder for future preference-supervised extensions."""

    def forward(
        self,
        scores: torch.Tensor,
        preference_targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return zero when unused, otherwise raise until the extension is implemented."""

        if preference_targets is None:
            return scores.new_zeros(())
        raise NotImplementedError(
            "TODO: implement trajectory preference/ranking loss when rater-scored alternatives are available."
        )
