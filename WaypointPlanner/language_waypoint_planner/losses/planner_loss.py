"""Composite loss computation for multitask training."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from language_waypoint_planner.configs.schema import LossConfig
from language_waypoint_planner.models.planner_model import PlannerOutput

from .behavior import MaskedBehaviorCrossEntropyLoss
from .preference import PreferenceRankingLoss
from .rationale import MaskedRationaleTextLoss
from .smoothness import TrajectorySmoothnessLoss
from .waypoint import TimeWeightedWaypointLoss


class PlannerLossComputer(nn.Module):
    """Combine all task losses with configurable weights."""

    def __init__(self, config: LossConfig, future_steps: int, pad_token_id: int = 0) -> None:
        super().__init__()
        self.config = config
        self.waypoint_loss = TimeWeightedWaypointLoss(
            future_steps=future_steps,
            final_step_weight=config.waypoint_final_step_weight,
        )
        self.behavior_loss = MaskedBehaviorCrossEntropyLoss()
        self.rationale_loss = MaskedRationaleTextLoss(pad_token_id=pad_token_id)
        self.smoothness_loss = TrajectorySmoothnessLoss()
        self.preference_loss = PreferenceRankingLoss()

    def forward(self, outputs: PlannerOutput, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute a weighted sum of all task losses."""

        waypoint_component = self.waypoint_loss(
            pred_waypoints=outputs.pred_waypoints,
            target_waypoints=batch["target_waypoints"],
            valid_mask=batch["valid_masks"]["waypoints"],
        )
        behavior_component = self.behavior_loss(
            logits=outputs.pred_behavior_logits,
            targets=batch["target_behavior"],
            valid_mask=batch["valid_masks"]["behavior"],
        )
        rationale_component = self.rationale_loss(
            logits=outputs.pred_rationale_logits_or_tokens,
            target_ids=batch["target_rationale_ids"],
            token_mask=batch["target_rationale_token_mask"],
            sample_mask=batch["valid_masks"]["rationale"],
        )
        smoothness_component = self.smoothness_loss(
            pred_waypoints=outputs.pred_waypoints,
            valid_mask=batch["valid_masks"]["waypoints"],
        )
        preference_component = self.preference_loss(outputs.pred_behavior_logits)
        total = (
            self.config.waypoint_weight * waypoint_component
            + self.config.behavior_weight * behavior_component
            + self.config.rationale_weight * rationale_component
            + self.config.smoothness_weight * smoothness_component
            + self.config.preference_weight * preference_component
        )
        return {
            "total": total,
            "waypoint": waypoint_component,
            "behavior": behavior_component,
            "rationale": rationale_component,
            "smoothness": smoothness_component,
            "preference": preference_component,
        }
