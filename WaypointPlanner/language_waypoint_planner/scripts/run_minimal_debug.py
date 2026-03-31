"""Run the minimal debug training configuration."""

from __future__ import annotations

from pathlib import Path

from language_waypoint_planner.configs import load_config
from language_waypoint_planner.train.engine import run_training


def main() -> None:
    """Launch the minimal debug run."""

    config_path = Path(__file__).resolve().parents[1] / "configs" / "minimal_debug.yaml"
    config = load_config(config_path)
    run_training(config)


if __name__ == "__main__":
    main()
