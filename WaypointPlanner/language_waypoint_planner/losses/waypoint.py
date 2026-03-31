"""Waypoint regression losses."""

from __future__ import annotations

import torch
from torch import nn


class TimeWeightedWaypointLoss(nn.Module):
    """Smooth L1 waypoint loss with increasing weight over future time."""

    def __init__(self, future_steps: int, final_step_weight: float = 3.0) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.final_step_weight = final_step_weight

    def forward(
        self,
        pred_waypoints: torch.Tensor,
        target_waypoints: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute a masked time-weighted waypoint loss."""

        if pred_waypoints.shape != target_waypoints.shape:
            raise ValueError(
                f"pred_waypoints and target_waypoints must match, got {pred_waypoints.shape} vs {target_waypoints.shape}"
            )
        step_weights = torch.linspace(
            1.0,
            self.final_step_weight,
            steps=self.future_steps,
            device=pred_waypoints.device,
            dtype=pred_waypoints.dtype,
        ).view(1, self.future_steps, 1)
        errors = torch.nn.functional.smooth_l1_loss(
            pred_waypoints,
            target_waypoints,
            reduction="none",
        )
        weighted = errors * step_weights
        sample_loss = weighted.mean(dim=(1, 2))
        if not valid_mask.any():
            return pred_waypoints.new_zeros(())
        return sample_loss[valid_mask].mean()
