"""One-batch training step test."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from language_waypoint_planner.configs import load_config
from language_waypoint_planner.train.engine import run_training


class TrainStepTest(unittest.TestCase):
    """Exercise the end-to-end one-batch training path."""

    def test_one_batch_training_run(self) -> None:
        config = load_config(PROJECT_ROOT / "language_waypoint_planner" / "configs" / "minimal_debug.yaml")
        with tempfile.TemporaryDirectory() as tmpdir:
            config.logging.output_dir = tmpdir
            metrics = run_training(config)
        self.assertIn("ade", metrics)
        self.assertIn("behavior_accuracy", metrics)


if __name__ == "__main__":
    unittest.main()
