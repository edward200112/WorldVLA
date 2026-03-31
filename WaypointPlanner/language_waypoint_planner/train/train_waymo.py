"""Waypoint + behavior pretraining entrypoint for Waymo-style data."""

from __future__ import annotations

import argparse
from pathlib import Path

from language_waypoint_planner.configs import load_config
from language_waypoint_planner.train.engine import run_training


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    default_config = Path(__file__).resolve().parents[1] / "configs" / "waymo_pretrain.yaml"
    parser = argparse.ArgumentParser(description="Train the planner on Waymo-style waypoint supervision.")
    parser.add_argument("--config", type=str, default=str(default_config), help="Path to config file.")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    config = load_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()
