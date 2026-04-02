from __future__ import annotations

import argparse
from typing import Any, Mapping

from .config import ACTION_QUANTIZATION_MODES, TokenConfig
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


def add_action_quantization_args(parser: argparse.ArgumentParser) -> None:
    """为训练 / 推理 / 分析脚本统一添加 action 量化参数。"""
    parser.add_argument(
        "--action_quantization_mode",
        type=str,
        default=None,
        choices=ACTION_QUANTIZATION_MODES,
    )
    parser.add_argument(
        "--action_longitudinal_quantization_mode",
        type=str,
        default=None,
        choices=ACTION_QUANTIZATION_MODES,
    )
    parser.add_argument(
        "--action_lateral_quantization_mode",
        type=str,
        default=None,
        choices=ACTION_QUANTIZATION_MODES,
    )
    parser.add_argument("--action_zero_deadband", type=float, default=None)
    parser.add_argument("--action_longitudinal_zero_dense_power", type=float, default=None)
    parser.add_argument("--action_lateral_zero_dense_power", type=float, default=None)
    parser.add_argument("--action_lateral_near_zero_threshold", type=float, default=None)


def apply_action_quantization_args(token_config: TokenConfig, args: Any) -> None:
    """把命令行量化参数写回 TokenConfig。"""
    if getattr(args, "action_quantization_mode", None) is not None:
        token_config.action_quantization_mode = args.action_quantization_mode
    if getattr(args, "action_longitudinal_quantization_mode", None) is not None:
        token_config.action_longitudinal_quantization_mode = args.action_longitudinal_quantization_mode
    if getattr(args, "action_lateral_quantization_mode", None) is not None:
        token_config.action_lateral_quantization_mode = args.action_lateral_quantization_mode
    if getattr(args, "action_zero_deadband", None) is not None:
        token_config.action_zero_deadband = float(args.action_zero_deadband)
    if getattr(args, "action_longitudinal_zero_dense_power", None) is not None:
        token_config.action_longitudinal_zero_dense_power = float(args.action_longitudinal_zero_dense_power)
    if getattr(args, "action_lateral_zero_dense_power", None) is not None:
        token_config.action_lateral_zero_dense_power = float(args.action_lateral_zero_dense_power)
    if getattr(args, "action_lateral_near_zero_threshold", None) is not None:
        token_config.action_lateral_near_zero_threshold = float(args.action_lateral_near_zero_threshold)


def print_action_quantization_summary(summary: Mapping[str, Any]) -> None:
    """打印当前 action token 的量化语义摘要。"""
    print(
        "[action_quantization] "
        f"mode_default={summary['default_mode']} "
        f"deadband={summary['action_zero_deadband']:.4f} "
        f"value_range={summary['action_value_range']:.4f} "
        f"bins_per_dim={summary['action_bins_per_dim']} "
        f"lateral_near_zero_threshold={summary['lateral_near_zero_threshold']:.4f}"
    )
    for dim_name, dim_summary in summary["per_dim"].items():
        centers = ", ".join(f"{value:.4f}" for value in dim_summary["centers"])
        print(
            "[action_quantization] "
            f"dim={dim_name} mode={dim_summary['mode']} "
            f"zero_dense_power={dim_summary['zero_dense_power']:.4f} "
            f"centers=[{centers}]"
        )


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
