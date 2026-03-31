"""Prediction heads for trajectory, behavior, and rationale tasks."""

from __future__ import annotations

import torch
from torch import nn

from language_waypoint_planner.constants import BEHAVIOR_LABELS


class WaypointHead(nn.Module):
    """Predict a fixed number of future 2D waypoints."""

    def __init__(self, hidden_dim: int, future_steps: int) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, future_steps * 2),
        )

    def forward(self, global_token: torch.Tensor) -> torch.Tensor:
        """Map the fused global token to future 2D waypoints."""

        output = self.network(global_token)
        return output.reshape(global_token.shape[0], self.future_steps, 2)


class BehaviorHead(nn.Module):
    """Predict one of the discrete driving behavior labels."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(BEHAVIOR_LABELS)),
        )

    def forward(self, global_token: torch.Tensor) -> torch.Tensor:
        """Return behavior logits."""

        return self.network(global_token)


class RationaleHead(nn.Module):
    """Generate fixed-length rationale token logits from the global token."""

    def __init__(self, hidden_dim: int, vocab_size: int, max_length: int) -> None:
        super().__init__()
        self.max_length = max_length
        self.query_embedding = nn.Parameter(torch.zeros(1, max_length, hidden_dim))
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, global_token: torch.Tensor) -> torch.Tensor:
        """Return token logits shaped `[B, max_length, vocab_size]`."""

        hidden = global_token.unsqueeze(1) + self.query_embedding
        return self.network(hidden)
