"""Dataset interfaces for Waymo, DriveLM, and Talk2Car style samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from language_waypoint_planner.configs.schema import DatasetSourceConfig
from language_waypoint_planner.constants import DEFAULT_RATIONALE_TEMPLATES, ROUTE_COMMANDS

from .label_utils import (
    BehaviorHeuristicsConfig,
    behavior_name_to_index,
    build_behavior_label_from_trajectory,
    build_missing_label_masks,
    normalize_rationale,
)
from .sample import BaseDrivingSample


class BaseManifestDrivingDataset(Dataset[BaseDrivingSample]):
    """Common manifest-backed dataset with a deterministic synthetic fallback."""

    dataset_name: str = "base"

    def __init__(
        self,
        source_config: DatasetSourceConfig,
        heuristics_config: Optional[BehaviorHeuristicsConfig] = None,
    ) -> None:
        self.source_config = source_config
        self.heuristics_config = heuristics_config or BehaviorHeuristicsConfig()
        self.data_root = Path(source_config.data_root).expanduser()
        self.image_size = tuple(source_config.image_size)
        self.cameras = list(source_config.cameras)
        self.temporal_window = source_config.temporal_window
        self.future_steps = source_config.future_steps
        self.history_steps = source_config.history_steps
        self.records = self._load_records()

    def _load_records(self) -> List[Any]:
        if self.source_config.annotation_file:
            manifest_path = Path(self.source_config.annotation_file).expanduser()
            if not manifest_path.is_absolute():
                manifest_path = self.data_root / manifest_path
            if not manifest_path.exists():
                raise FileNotFoundError(f"Manifest not found: {manifest_path}")
            return self._read_manifest(manifest_path)
        if self.source_config.use_synthetic:
            return list(range(self.source_config.synthetic_length))
        raise FileNotFoundError(
            f"{self.dataset_name} requires annotation_file or use_synthetic=True for offline debug runs"
        )

    def _read_manifest(self, manifest_path: Path) -> List[Dict[str, Any]]:
        text = manifest_path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Manifest is empty: {manifest_path}")
        if manifest_path.suffix == ".jsonl":
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        loaded = json.loads(text)
        if not isinstance(loaded, list):
            raise TypeError(f"Expected list manifest, got {type(loaded)!r}")
        return loaded

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> BaseDrivingSample:
        record = self.records[index]
        if isinstance(record, int):
            return self._build_synthetic_sample(index=record)
        if not isinstance(record, dict):
            raise TypeError(f"Dataset record must be dict or int, got {type(record)!r}")
        return self._build_manifest_sample(record)

    def _load_image(self, image_path: str) -> torch.Tensor:
        path = Path(image_path)
        if not path.is_absolute():
            path = self.data_root / path
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        image = Image.open(path).convert("RGB").resize(self.image_size)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return tensor

    def _load_image_sequence(self, image_paths: Sequence[Sequence[str]]) -> torch.Tensor:
        if len(image_paths) != self.temporal_window:
            raise ValueError(
                f"Expected {self.temporal_window} temporal frames, got {len(image_paths)} in {self.dataset_name}"
            )
        frame_tensors: List[torch.Tensor] = []
        for frame_paths in image_paths:
            if len(frame_paths) != len(self.cameras):
                raise ValueError(
                    f"Expected {len(self.cameras)} cameras per frame, got {len(frame_paths)} in {self.dataset_name}"
                )
            frame = torch.stack([self._load_image(path) for path in frame_paths], dim=0)
            frame_tensors.append(frame)
        return torch.stack(frame_tensors, dim=0)

    def _build_manifest_sample(self, record: Dict[str, Any]) -> BaseDrivingSample:
        images_raw = record.get("images")
        ego_hist = torch.tensor(record["ego_hist"], dtype=torch.float32)
        velocity = torch.tensor(record["velocity"], dtype=torch.float32)
        acceleration = torch.tensor(record["acceleration"], dtype=torch.float32)
        route_command = str(record.get("route_command", "straight"))
        language_input = str(record.get("language_input", ""))

        if images_raw is None:
            raise KeyError("Manifest record must contain 'images'")
        images = self._load_image_sequence(images_raw)

        target_waypoints = None
        if record.get("target_waypoints") is not None:
            target_waypoints = torch.tensor(record["target_waypoints"], dtype=torch.float32)
        target_behavior = record.get("target_behavior")
        if isinstance(target_behavior, str):
            target_behavior = behavior_name_to_index(target_behavior)
        elif target_behavior is not None:
            target_behavior = int(target_behavior)
        if target_behavior is None and target_waypoints is not None and self.source_config.auto_behavior_from_waypoints:
            target_behavior = build_behavior_label_from_trajectory(
                target_waypoints,
                config=self.heuristics_config,
            )

        rationale_raw = record.get("target_rationale")
        target_rationale = None if rationale_raw is None else normalize_rationale(str(rationale_raw))
        valid_masks = self._resolve_valid_masks(
            has_waypoints=target_waypoints is not None,
            has_behavior=target_behavior is not None,
            has_rationale=target_rationale is not None,
            explicit_masks=record.get("valid_masks"),
        )

        return BaseDrivingSample(
            images=images,
            ego_hist=ego_hist,
            velocity=velocity,
            acceleration=acceleration,
            route_command=route_command,
            language_input=language_input,
            target_waypoints=target_waypoints,
            target_behavior=target_behavior,
            target_rationale=target_rationale,
            valid_masks=valid_masks,
        )

    def _resolve_valid_masks(
        self,
        has_waypoints: bool,
        has_behavior: bool,
        has_rationale: bool,
        explicit_masks: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, bool]:
        masks = build_missing_label_masks(
            has_waypoints=has_waypoints,
            has_behavior=has_behavior,
            has_rationale=has_rationale,
        )
        if explicit_masks is None:
            return masks
        for key in masks:
            if key in explicit_masks:
                masks[key] = bool(explicit_masks[key])
        return masks

    def default_label_availability(self) -> Dict[str, bool]:
        """Return default task availability flags for the dataset."""

        return {"waypoints": True, "behavior": True, "rationale": False}

    def _build_synthetic_sample(self, index: int) -> BaseDrivingSample:
        raise NotImplementedError("Subclasses must implement synthetic sample generation")

    def _build_synthetic_images(self, index: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(index + 17)
        images = torch.rand(
            self.temporal_window,
            len(self.cameras),
            3,
            self.image_size[1],
            self.image_size[0],
            generator=generator,
        )
        camera_offsets = torch.linspace(0.0, 0.2, steps=len(self.cameras)).view(1, -1, 1, 1, 1)
        time_offsets = torch.linspace(0.0, 0.1, steps=self.temporal_window).view(-1, 1, 1, 1, 1)
        images = (images * 0.8 + camera_offsets + time_offsets).clamp(0.0, 1.0)
        return images

    def _build_synthetic_ego_history(self, behavior_name: str) -> Dict[str, torch.Tensor]:
        history_x = torch.linspace(-4.0, -0.25, steps=self.history_steps)
        lateral_scale = {
            "lane_change_left": 0.4,
            "lane_change_right": -0.4,
            "turn_left": 0.6,
            "turn_right": -0.6,
        }.get(behavior_name, 0.0)
        history_y = torch.linspace(lateral_scale, 0.0, steps=self.history_steps)
        ego_hist = torch.stack([history_x, history_y], dim=-1)
        xy_delta = torch.zeros_like(ego_hist)
        xy_delta[1:] = ego_hist[1:] - ego_hist[:-1]
        velocity = torch.linalg.norm(xy_delta, dim=-1, keepdim=True) / 0.25
        acceleration = torch.zeros_like(velocity)
        acceleration[1:] = (velocity[1:] - velocity[:-1]) / 0.25
        return {"ego_hist": ego_hist, "velocity": velocity, "acceleration": acceleration}

    def _behavior_to_waypoints(self, behavior_name: str) -> torch.Tensor:
        t = torch.linspace(0.25, 5.0, steps=self.future_steps)
        if behavior_name == "stop":
            x = torch.cumsum(torch.linspace(0.45, 0.02, steps=self.future_steps), dim=0)
            y = torch.zeros_like(x)
        elif behavior_name == "yield":
            x = torch.cumsum(torch.linspace(0.6, 0.18, steps=self.future_steps), dim=0)
            y = 0.1 * torch.sin(t)
        elif behavior_name == "lane_change_left":
            x = torch.linspace(0.5, 10.0, steps=self.future_steps)
            y = 1.8 * torch.tanh((t - 2.5) * 1.2)
        elif behavior_name == "lane_change_right":
            x = torch.linspace(0.5, 10.0, steps=self.future_steps)
            y = -1.8 * torch.tanh((t - 2.5) * 1.2)
        elif behavior_name == "turn_left":
            theta = torch.linspace(0.0, np.pi / 3.0, steps=self.future_steps)
            radius = 8.0
            x = radius * torch.sin(theta)
            y = radius * (1.0 - torch.cos(theta))
        elif behavior_name == "turn_right":
            theta = torch.linspace(0.0, np.pi / 3.0, steps=self.future_steps)
            radius = 8.0
            x = radius * torch.sin(theta)
            y = -radius * (1.0 - torch.cos(theta))
        else:
            x = torch.linspace(0.5, 10.0, steps=self.future_steps)
            y = 0.08 * torch.sin(t * 0.5)
        return torch.stack([x, y], dim=-1)

    def _synthetic_route_command(self, behavior_name: str) -> str:
        if behavior_name in ("turn_left", "lane_change_left"):
            return "left"
        if behavior_name in ("turn_right", "lane_change_right"):
            return "right"
        return "straight"

    def _synthetic_language_input(self, behavior_name: str) -> str:
        return ""

    def _synthetic_rationale(self, behavior_name: str) -> Optional[str]:
        return None

    def _synthetic_masks(self, behavior_name: str) -> Dict[str, bool]:
        del behavior_name
        return self.default_label_availability()

    def _common_synthetic_sample(self, index: int) -> BaseDrivingSample:
        behavior_name = list(DEFAULT_RATIONALE_TEMPLATES.keys())[index % len(DEFAULT_RATIONALE_TEMPLATES)]
        waypoints = self._behavior_to_waypoints(behavior_name)
        ego = self._build_synthetic_ego_history(behavior_name)
        rationale = self._synthetic_rationale(behavior_name)
        if rationale is not None:
            rationale = normalize_rationale(rationale, behavior_name)
        target_behavior = behavior_name_to_index(behavior_name)
        valid_masks = self._synthetic_masks(behavior_name)
        if not valid_masks["waypoints"]:
            waypoints = None
        if not valid_masks["behavior"]:
            target_behavior = None
        if not valid_masks["rationale"]:
            rationale = None

        return BaseDrivingSample(
            images=self._build_synthetic_images(index),
            ego_hist=ego["ego_hist"],
            velocity=ego["velocity"],
            acceleration=ego["acceleration"],
            route_command=self._synthetic_route_command(behavior_name),
            language_input=self._synthetic_language_input(behavior_name),
            target_waypoints=waypoints,
            target_behavior=target_behavior,
            target_rationale=rationale,
            valid_masks=valid_masks,
        )


class WaymoE2EDataset(BaseManifestDrivingDataset):
    """Waymo Open Dataset style end-to-end driving dataset."""

    dataset_name = "waymo"

    def default_label_availability(self) -> Dict[str, bool]:
        return {"waypoints": True, "behavior": True, "rationale": False}

    def _build_synthetic_sample(self, index: int) -> BaseDrivingSample:
        return self._common_synthetic_sample(index)


class DriveLMDataset(BaseManifestDrivingDataset):
    """DriveLM-style reasoning dataset with sparse or missing waypoint labels."""

    dataset_name = "drivelm"

    def default_label_availability(self) -> Dict[str, bool]:
        return {"waypoints": False, "behavior": True, "rationale": True}

    def _synthetic_language_input(self, behavior_name: str) -> str:
        return f"Explain the next action for a {behavior_name.replace('_', ' ')} maneuver."

    def _synthetic_rationale(self, behavior_name: str) -> Optional[str]:
        return DEFAULT_RATIONALE_TEMPLATES[behavior_name]

    def _build_synthetic_sample(self, index: int) -> BaseDrivingSample:
        return self._common_synthetic_sample(index)


class Talk2CarDataset(BaseManifestDrivingDataset):
    """Talk2Car-style command grounding dataset with language and route cues."""

    dataset_name = "talk2car"

    def default_label_availability(self) -> Dict[str, bool]:
        return {"waypoints": True, "behavior": False, "rationale": True}

    def _synthetic_language_input(self, behavior_name: str) -> str:
        route = self._synthetic_route_command(behavior_name)
        if route == "left":
            return "Please move left and continue past the obstruction."
        if route == "right":
            return "Please move right and continue after the parked vehicle."
        return "Keep going straight while staying safe."

    def _synthetic_rationale(self, behavior_name: str) -> Optional[str]:
        return DEFAULT_RATIONALE_TEMPLATES[behavior_name]

    def _build_synthetic_sample(self, index: int) -> BaseDrivingSample:
        return self._common_synthetic_sample(index)


def build_dataset(source_config: DatasetSourceConfig) -> BaseManifestDrivingDataset:
    """Instantiate a dataset from its source configuration."""

    name = source_config.name.lower()
    if name == "waymo":
        return WaymoE2EDataset(source_config)
    if name == "drivelm":
        return DriveLMDataset(source_config)
    if name == "talk2car":
        return Talk2CarDataset(source_config)
    raise ValueError(f"Unsupported dataset name: {source_config.name}. Supported: waymo, drivelm, talk2car")
