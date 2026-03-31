"""TODO bridge for future CARLA / Bench2Drive integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class CarlaBench2DriveBridge:
    """Placeholder interface for future closed-loop simulator integration."""

    planner_frequency_hz: int = 4

    def from_sensor_packet(self, sensor_packet: Dict[str, Any]) -> Dict[str, Any]:
        """TODO: map simulator sensor packets into planner inputs."""

        raise NotImplementedError(
            "TODO: implement CARLA sensor packet conversion once the offline planner interface is stabilized."
        )

    def to_control(self, planner_output: Dict[str, Any]) -> Dict[str, float]:
        """TODO: map planner outputs to simulator control commands."""

        raise NotImplementedError(
            "TODO: implement CARLA / Bench2Drive control conversion during closed-loop integration."
        )
