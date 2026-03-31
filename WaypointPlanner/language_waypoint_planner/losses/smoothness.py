"""Trajectory smoothness regularizer."""

from __future__ import annotations

import torch
from torch import nn


class TrajectorySmoothnessLoss(nn.Module):
    """Penalize high second-order differences in predicted trajectories."""

    def forward(self, pred_waypoints: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Compute a masked smoothness regularization loss."""

        if pred_waypoints.ndim != 3 or pred_waypoints.shape[-1] != 2:
            raise ValueError(f"Expected pred_waypoints with shape [B, T, 2], got {tuple(pred_waypoints.shape)}")
        if pred_waypoints.shape[1] < 3 or not valid_mask.any():
            return pred_waypoints.new_zeros(())
        second_difference = pred_waypoints[:, 2:] - 2.0 * pred_waypoints[:, 1:-1] + pred_waypoints[:, :-2]
        sample_loss = second_difference.pow(2).mean(dim=(1, 2))
        return sample_loss[valid_mask].mean()
