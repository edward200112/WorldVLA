"""Behavior heuristics, rationale normalization, and masking helpers."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Optional

import torch

from language_waypoint_planner.constants import (
    BEHAVIOR_TO_INDEX,
    DEFAULT_RATIONALE_TEMPLATES,
    INDEX_TO_BEHAVIOR,
)


@dataclass
class BehaviorHeuristicsConfig:
    """Thresholds for behavior label generation from future trajectories."""

    dt_seconds: float = 0.25
    stop_terminal_speed_threshold: float = 0.35
    yield_decel_threshold: float = 0.75
    turn_heading_threshold_deg: float = 30.0
    lane_change_lateral_threshold: float = 1.2
    lane_change_heading_threshold_deg: float = 15.0
    lane_change_curvature_threshold_deg: float = 12.0


def _heading_from_delta(delta: torch.Tensor) -> torch.Tensor:
    return torch.atan2(delta[..., 1], delta[..., 0].clamp(min=1e-6))


def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def behavior_name_to_index(name: str) -> int:
    """Convert a behavior name to its integer label."""

    if name not in BEHAVIOR_TO_INDEX:
        raise KeyError(f"Unknown behavior label: {name}")
    return BEHAVIOR_TO_INDEX[name]


def behavior_index_to_name(index: int) -> str:
    """Convert a behavior index to its string label."""

    if index not in INDEX_TO_BEHAVIOR:
        raise KeyError(f"Unknown behavior index: {index}")
    return INDEX_TO_BEHAVIOR[index]


def build_behavior_label_from_trajectory(
    waypoints: torch.Tensor,
    config: Optional[BehaviorHeuristicsConfig] = None,
) -> int:
    """Infer a discrete behavior label from future 2D waypoints."""

    if waypoints.ndim != 2 or waypoints.shape[-1] != 2:
        raise ValueError(f"Expected waypoints with shape [T, 2], got {tuple(waypoints.shape)}")
    if waypoints.shape[0] < 3:
        raise ValueError("At least 3 future waypoints are required to infer behavior")

    cfg = config or BehaviorHeuristicsConfig()
    deltas = waypoints[1:] - waypoints[:-1]
    speeds = torch.linalg.norm(deltas, dim=-1) / cfg.dt_seconds
    headings = _heading_from_delta(deltas)
    terminal_speed = float(speeds[-1])
    heading_change = float(_wrap_angle(headings[-1] - headings[0]))
    lateral_displacement = float(waypoints[-1, 1] - waypoints[0, 1])
    heading_diffs = _wrap_angle(headings[1:] - headings[:-1])
    curvature_deg = float(torch.rad2deg(torch.abs(heading_diffs).mean())) if len(heading_diffs) > 0 else 0.0
    speed_drop = float(speeds[0] - speeds[-1])
    turn_threshold = math.radians(cfg.turn_heading_threshold_deg)
    lane_change_heading_threshold = math.radians(cfg.lane_change_heading_threshold_deg)

    if terminal_speed <= cfg.stop_terminal_speed_threshold:
        return behavior_name_to_index("stop")
    if abs(heading_change) >= turn_threshold:
        return behavior_name_to_index("turn_left" if heading_change > 0.0 else "turn_right")
    if (
        abs(lateral_displacement) >= cfg.lane_change_lateral_threshold
        and abs(heading_change) <= lane_change_heading_threshold
        and curvature_deg <= cfg.lane_change_curvature_threshold_deg
    ):
        return behavior_name_to_index(
            "lane_change_left" if lateral_displacement > 0.0 else "lane_change_right"
        )
    if speed_drop >= cfg.yield_decel_threshold and terminal_speed > cfg.stop_terminal_speed_threshold:
        return behavior_name_to_index("yield")
    return behavior_name_to_index("keep_lane")


def normalize_rationale(text: str, behavior: Optional[str] = None) -> str:
    """Normalize rationale text into a concise template-friendly sentence."""

    normalized = text.lower().strip()
    normalized = re.sub(r"[^a-z0-9\s']", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized and behavior is not None:
        return DEFAULT_RATIONALE_TEMPLATES[behavior]
    if "pedestrian" in normalized or "crosswalk" in normalized:
        return "slowing because a pedestrian may enter the crosswalk"
    if "yield" in normalized:
        return "decelerating to yield to other traffic"
    if "turn left" in normalized or "left turn" in normalized:
        return DEFAULT_RATIONALE_TEMPLATES["turn_left"]
    if "turn right" in normalized or "right turn" in normalized:
        return DEFAULT_RATIONALE_TEMPLATES["turn_right"]
    if "lane change" in normalized or "merge" in normalized:
        if "left" in normalized:
            return DEFAULT_RATIONALE_TEMPLATES["lane_change_left"]
        if "right" in normalized:
            return DEFAULT_RATIONALE_TEMPLATES["lane_change_right"]
    if normalized:
        return normalized
    if behavior is None:
        return ""
    return DEFAULT_RATIONALE_TEMPLATES[behavior]


def build_missing_label_masks(
    has_waypoints: bool,
    has_behavior: bool,
    has_rationale: bool,
) -> Dict[str, bool]:
    """Build the standard missing-label mask dictionary."""

    return {
        "waypoints": bool(has_waypoints),
        "behavior": bool(has_behavior),
        "rationale": bool(has_rationale),
    }
