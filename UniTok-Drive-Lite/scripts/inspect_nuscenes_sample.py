"""检查单个 nuScenes 样本的最小脚本。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unitok_drive_lite import build_default_config
from unitok_drive_lite.data import build_dataset
from unitok_drive_lite.eval_utils import bev_occupancy_stats, tensor_to_list


def parse_args() -> argparse.Namespace:
    """解析样本检查参数。"""
    parser = argparse.ArgumentParser(description="检查单个 nuScenes 样本的 metadata / action / BEV 统计。")
    parser.add_argument("--nuscenes_root", type=str, required=True)
    parser.add_argument("--nuscenes_version", type=str, default="v1.0-mini")
    parser.add_argument("--nuscenes_split", type=str, default="mini_train")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=16)
    parser.add_argument("--focus_scene_token", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def _action_stats(actions: torch.Tensor) -> dict[str, object]:
    """返回历史/未来动作序列的最小统计。"""
    actions_cpu = actions.detach().cpu().to(torch.float32)
    return {
        "shape": list(actions_cpu.shape),
        "mean": tensor_to_list(actions_cpu.mean(dim=0)),
        "min": tensor_to_list(actions_cpu.min(dim=0).values),
        "max": tensor_to_list(actions_cpu.max(dim=0).values),
        "final_cumsum": tensor_to_list(torch.cumsum(actions_cpu, dim=0)[-1]),
    }


def main() -> None:
    """检查一个 nuScenes 样本。"""
    args = parse_args()
    config = build_default_config(PROJECT_ROOT)

    try:
        dataset = build_dataset(
            dataset_type="nuscenes",
            token_config=config.tokens,
            seed=config.train.seed,
            nuscenes_root=args.nuscenes_root,
            nuscenes_version=args.nuscenes_version,
            nuscenes_split=args.nuscenes_split,
            max_samples=args.max_samples,
            focus_scene_token=args.focus_scene_token,
        )
    except (ImportError, FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"[data] {error}") from error

    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise SystemExit(
            f"[data] sample_index={args.sample_index} 超出范围，当前数据集大小为 {len(dataset)}"
        )

    sample = dataset[args.sample_index]
    future_bev_frame_stats = sample.metadata.get("future_bev_frame_stats", [])
    summary = {
        "sample_index": args.sample_index,
        "metadata": sample.metadata,
        "history_actions_stats": _action_stats(sample.history_actions),
        "future_actions_stats": _action_stats(sample.future_actions),
        "bev_now_stats": bev_occupancy_stats(sample.bev_now),
        "future_bevs_stats": bev_occupancy_stats(sample.future_bevs),
        "future_bev_frame_stats": future_bev_frame_stats,
    }

    print("[inspect] metadata=")
    print(json.dumps(sample.metadata, ensure_ascii=False, indent=2))
    print("[inspect] history_actions_stats=", summary["history_actions_stats"])
    print("[inspect] future_actions_stats=", summary["future_actions_stats"])
    print("[inspect] bev_now_stats=", summary["bev_now_stats"])
    print("[inspect] future_bevs_stats=", summary["future_bevs_stats"])
    print("[inspect] future_bev_source=", sample.metadata.get("future_bev_source"))
    print("[inspect] future_sample_tokens=", sample.metadata.get("future_sample_tokens"))

    if args.output_dir:
        output_dir = PROJECT_ROOT / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"nuscenes_sample_{args.sample_index:04d}.pt"
        torch.save(
            {
                "metadata": sample.metadata,
                "front_image": sample.front_image.detach().cpu(),
                "bev_now": sample.bev_now.detach().cpu(),
                "history_actions": sample.history_actions.detach().cpu(),
                "future_actions": sample.future_actions.detach().cpu(),
                "future_bevs": sample.future_bevs.detach().cpu(),
                "summary": summary,
            },
            output_path,
        )
        print(f"[inspect] output_pt={output_path}")


if __name__ == "__main__":
    main()
