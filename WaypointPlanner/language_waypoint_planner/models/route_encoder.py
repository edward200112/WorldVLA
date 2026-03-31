"""Route command token encoder."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from language_waypoint_planner.constants import ROUTE_COMMANDS


class RouteCommandEncoder(nn.Module):
    """Encode a discrete route command into a single token."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(len(ROUTE_COMMANDS), output_dim)

    def forward(self, route_command_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Return one route token per batch element."""

        if route_command_ids.ndim != 1:
            raise ValueError(f"Expected route_command_ids with shape [B], got {tuple(route_command_ids.shape)}")
        tokens = self.embedding(route_command_ids).unsqueeze(1)
        valid_mask = torch.ones(route_command_ids.shape[0], 1, dtype=torch.bool, device=route_command_ids.device)
        return tokens, valid_mask, {"route_ids": route_command_ids}
