"""Shared constants for the language waypoint planner."""

from typing import Dict, List

BEHAVIOR_LABELS: List[str] = [
    "keep_lane",
    "stop",
    "yield",
    "lane_change_left",
    "lane_change_right",
    "turn_left",
    "turn_right",
]

BEHAVIOR_TO_INDEX: Dict[str, int] = {label: idx for idx, label in enumerate(BEHAVIOR_LABELS)}
INDEX_TO_BEHAVIOR: Dict[int, str] = {idx: label for label, idx in BEHAVIOR_TO_INDEX.items()}

ROUTE_COMMANDS: List[str] = ["left", "straight", "right"]
ROUTE_TO_INDEX: Dict[str, int] = {label: idx for idx, label in enumerate(ROUTE_COMMANDS)}
INDEX_TO_ROUTE: Dict[int, str] = {idx: label for label, idx in ROUTE_TO_INDEX.items()}

DEFAULT_RATIONALE_TEMPLATES: Dict[str, str] = {
    "keep_lane": "maintaining lane while following the planned route",
    "stop": "slowing down to stop for the scene ahead",
    "yield": "decelerating to yield while preserving safety margin",
    "lane_change_left": "moving left to follow the route or avoid blockage",
    "lane_change_right": "moving right to follow the route or avoid blockage",
    "turn_left": "turning left to follow the route command",
    "turn_right": "turning right to follow the route command",
}
