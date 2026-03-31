"""Dataset smoke tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from language_waypoint_planner.configs.schema import DatasetSourceConfig
from language_waypoint_planner.data import (
    DriveLMDataset,
    Talk2CarDataset,
    WaymoE2EDataset,
    build_multitask_collate_fn,
)


class DatasetSmokeTest(unittest.TestCase):
    """Smoke test all datasets in synthetic mode."""

    def _make_source(self, name: str) -> DatasetSourceConfig:
        return DatasetSourceConfig(
            name=name,
            use_synthetic=True,
            synthetic_length=4,
            image_size=(64, 64),
            temporal_window=4,
            future_steps=20,
            history_steps=16,
        )

    def test_waymo_dataset(self) -> None:
        dataset = WaymoE2EDataset(self._make_source("waymo"))
        sample = dataset[0]
        self.assertEqual(tuple(sample.images.shape), (4, 3, 3, 64, 64))
        self.assertTrue(sample.valid_masks["waypoints"])

    def test_drivelm_dataset(self) -> None:
        dataset = DriveLMDataset(self._make_source("drivelm"))
        sample = dataset[0]
        self.assertTrue(sample.valid_masks["rationale"])
        self.assertIsNotNone(sample.target_rationale)

    def test_talk2car_dataset_and_collate(self) -> None:
        dataset = Talk2CarDataset(self._make_source("talk2car"))
        batch = build_multitask_collate_fn(rationale_max_length=12)([dataset[0], dataset[1]])
        self.assertEqual(tuple(batch["images"].shape), (2, 4, 3, 3, 64, 64))
        self.assertEqual(tuple(batch["target_rationale_ids"].shape), (2, 12))


if __name__ == "__main__":
    unittest.main()
