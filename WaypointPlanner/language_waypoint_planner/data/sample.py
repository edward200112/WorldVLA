"""Base sample dataclass for heterogeneous driving datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class BaseDrivingSample:
    """Container for a single multimodal driving sample."""

    images: torch.Tensor
    ego_hist: torch.Tensor
    velocity: torch.Tensor
    acceleration: torch.Tensor
    route_command: str
    language_input: str
    target_waypoints: Optional[torch.Tensor]
    target_behavior: Optional[int]
    target_rationale: Optional[str]
    valid_masks: Dict[str, bool]

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.images.ndim != 5:
            raise ValueError(f"images must have shape [T, N, C, H, W], got {tuple(self.images.shape)}")
        if self.images.shape[2] != 3:
            raise ValueError("images must be RGB tensors with 3 channels")
        if self.ego_hist.ndim != 2 or self.ego_hist.shape[-1] != 2:
            raise ValueError(f"ego_hist must have shape [L, 2], got {tuple(self.ego_hist.shape)}")
        if self.velocity.ndim not in (1, 2):
            raise ValueError(f"velocity must have shape [L] or [L, 1], got {tuple(self.velocity.shape)}")
        if self.acceleration.ndim not in (1, 2):
            raise ValueError(
                f"acceleration must have shape [L] or [L, 1], got {tuple(self.acceleration.shape)}"
            )
        if not isinstance(self.route_command, str) or not self.route_command:
            raise ValueError("route_command must be a non-empty string")
        if not isinstance(self.language_input, str):
            raise ValueError("language_input must be a string")
        if self.target_waypoints is not None:
            if self.target_waypoints.ndim != 2 or self.target_waypoints.shape[-1] != 2:
                raise ValueError(
                    f"target_waypoints must have shape [future_steps, 2], got {tuple(self.target_waypoints.shape)}"
                )
        if self.target_behavior is not None and not isinstance(self.target_behavior, int):
            raise ValueError("target_behavior must be an int index or None")
        if self.target_rationale is not None and not isinstance(self.target_rationale, str):
            raise ValueError("target_rationale must be a string or None")
        for key in ("waypoints", "behavior", "rationale"):
            if key not in self.valid_masks:
                raise ValueError(f"valid_masks must contain key '{key}'")
