"""Loss computation tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from language_waypoint_planner.configs import load_config
from language_waypoint_planner.data import WaymoE2EDataset, build_multitask_collate_fn
from language_waypoint_planner.losses import PlannerLossComputer
from language_waypoint_planner.models import PlannerModel


class LossComputationTest(unittest.TestCase):
    """Verify multitask losses are finite."""

    def test_loss_values_are_finite(self) -> None:
        config = load_config(PROJECT_ROOT / "language_waypoint_planner" / "configs" / "minimal_debug.yaml")
        dataset = WaymoE2EDataset(config.data.train_sources[0])
        batch = build_multitask_collate_fn(rationale_max_length=config.model.rationale_max_length)(
            [dataset[0], dataset[1]]
        )
        model = PlannerModel.from_config(
            config=config.model,
            image_size=config.data.train_sources[0].image_size,
            temporal_window=config.data.train_sources[0].temporal_window,
            num_cameras=len(config.data.train_sources[0].cameras),
            history_steps=config.data.train_sources[0].history_steps,
            future_steps=config.data.train_sources[0].future_steps,
        )
        outputs = model(
            images=batch["images"],
            ego_hist=batch["ego_hist"],
            velocity=batch["velocity"],
            acceleration=batch["acceleration"],
            route_command_ids=batch["route_command_ids"],
            language_input=batch["language_input"],
        )
        loss_computer = PlannerLossComputer(config.losses, future_steps=20)
        losses = loss_computer(outputs, batch)
        for value in losses.values():
            self.assertTrue(value.isfinite().item())


if __name__ == "__main__":
    unittest.main()
