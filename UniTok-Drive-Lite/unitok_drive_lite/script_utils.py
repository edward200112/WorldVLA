from __future__ import annotations

import argparse
from typing import Any

from .config import TokenConfig
from .data import build_dataset


def add_dataset_selection_args(
    parser: argparse.ArgumentParser,
    *,
    include_dataset_size: bool = True,
    include_focus_scene_token: bool = False,
) -> None:
    """为最小脚本统一添加 dataset 选择参数。"""
    parser.add_argument("--dataset_type", type=str, default="toy", choices=("toy", "nuscenes"))
    if include_dataset_size:
        parser.add_argument("--dataset_size", type=int, default=8)
    parser.add_argument("--nuscenes_root", type=str, default=None)
    parser.add_argument("--nuscenes_version", type=str, default="v1.0-mini")
    parser.add_argument("--nuscenes_split", type=str, default="mini_train")
    parser.add_argument("--max_samples", type=int, default=None)
    if include_focus_scene_token:
        parser.add_argument("--focus_scene_token", type=str, default=None)


def build_dataset_from_args(
    args: Any,
    token_config: TokenConfig,
    *,
    seed: int,
    dataset_size: int | None = None,
) -> Any:
    """从脚本参数构建 toy / nuScenes 数据集。"""
    resolved_dataset_size = dataset_size
    if resolved_dataset_size is None:
        resolved_dataset_size = getattr(args, "dataset_size", None)

    return build_dataset(
        dataset_type=args.dataset_type,
        token_config=token_config,
        seed=seed,
        dataset_size=resolved_dataset_size,
        nuscenes_root=getattr(args, "nuscenes_root", None),
        nuscenes_version=getattr(args, "nuscenes_version", "v1.0-mini"),
        nuscenes_split=getattr(args, "nuscenes_split", "mini_train"),
        max_samples=getattr(args, "max_samples", None),
        focus_scene_token=getattr(args, "focus_scene_token", None),
    )
