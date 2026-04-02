"""最小 action token 分布分析脚本。"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unitok_drive_lite import UnifiedDriveModel, build_default_config
from unitok_drive_lite.script_utils import add_dataset_selection_args, build_dataset_from_args
from unitok_drive_lite.train_utils import greedy_rollout, seed_everything


def parse_args() -> argparse.Namespace:
    """解析 action 分布分析参数。"""
    parser = argparse.ArgumentParser(description="分析最小主链路的 predicted action token 分布。")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="outputs/unitok_drive_lite/checkpoint_last",
    )
    add_dataset_selection_args(parser)
    parser.add_argument("--max_eval_samples", type=int, default=16)
    parser.add_argument("--sample_stride", type=int, default=1)
    parser.add_argument(
        "--output_json",
        type=str,
        default="outputs/unitok_drive_lite/action_distribution_summary.json",
    )
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--load_in_4bit", action="store_true")
    return parser.parse_args()


def _counter_to_top_k(counter: Counter, top_k: int, denominator: int) -> List[Dict[str, Any]]:
    """把 Counter 转成可 JSON 序列化的 top-k 摘要。"""
    top_items: List[Dict[str, Any]] = []
    for token_id, count in counter.most_common(top_k):
        top_items.append(
            {
                "token_id": int(token_id),
                "count": int(count),
                "frequency": float(count / max(denominator, 1)),
            }
        )
    return top_items


def _entropy_nats(counter: Counter) -> float:
    """计算 token 分布熵，单位为 nats。"""
    total = sum(counter.values())
    if total <= 0:
        return 0.0

    entropy = 0.0
    for count in counter.values():
        probability = count / total
        entropy -= probability * math.log(probability)
    return float(entropy)


def _sample_indices(length: int, stride: int, limit: int) -> List[int]:
    """生成分析用样本索引。"""
    if stride <= 0:
        raise ValueError(f"--sample_stride 必须大于 0，当前收到 {stride}")
    if limit <= 0:
        raise ValueError(f"--max_eval_samples 必须大于 0，当前收到 {limit}")

    indices = list(range(0, length, stride))
    return indices[:limit]


def _build_example_record(
    sample_index: int,
    metadata: Dict[str, Any],
    predicted_action_tokens: List[int],
    predicted_trajectory: torch.Tensor,
) -> Dict[str, Any]:
    """构造少量样例轨迹记录。"""
    return {
        "sample_index": sample_index,
        "metadata": metadata,
        "predicted_action_tokens": [int(token_id) for token_id in predicted_action_tokens],
        "predicted_trajectory": predicted_trajectory.tolist(),
    }


def main() -> None:
    """运行最小 action 分布分析。"""
    args = parse_args()
    checkpoint_dir = PROJECT_ROOT / args.checkpoint_dir
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"未找到检查点目录: {checkpoint_dir}")

    config = build_default_config(PROJECT_ROOT)
    config.model.load_in_4bit = args.load_in_4bit
    seed_everything(config.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        dataset = build_dataset_from_args(
            args=args,
            token_config=config.tokens,
            seed=config.train.seed,
        )
    except (ImportError, FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"[data] {error}") from error

    indices = _sample_indices(len(dataset), args.sample_stride, args.max_eval_samples)
    if not indices:
        raise RuntimeError("没有可分析的样本，请检查 dataset 参数或 --max_eval_samples / --sample_stride。")

    model = UnifiedDriveModel(config).to(device)
    model.load_checkpoint(checkpoint_dir)
    print(f"[model] backbone={config.model.model_name}")
    print(f"[model] load_in_4bit={config.model.load_in_4bit}")
    print(f"[data] dataset_type={args.dataset_type} eval_samples={len(indices)}")

    flat_counter: Counter = Counter()
    per_position_counters = [
        Counter() for _ in range(config.tokens.future_action_horizon)
    ]
    example_records: List[Dict[str, Any]] = []

    for sample_index in indices:
        sample = dataset[sample_index]
        output = greedy_rollout(model, sample, device)
        predicted_action_tokens = [int(token_id) for token_id in output["predicted_action_tokens"]]
        predicted_trajectory = output["predicted_trajectory"].detach().cpu()

        for position, token_id in enumerate(predicted_action_tokens):
            flat_counter[token_id] += 1
            if position < len(per_position_counters):
                per_position_counters[position][token_id] += 1

        if len(example_records) < 3:
            example_records.append(
                _build_example_record(
                    sample_index=sample_index,
                    metadata=sample.metadata,
                    predicted_action_tokens=predicted_action_tokens,
                    predicted_trajectory=predicted_trajectory,
                )
            )

    total_predicted_tokens = sum(flat_counter.values())
    unique_token_count = len(flat_counter)
    top_tokens = _counter_to_top_k(flat_counter, args.top_k, total_predicted_tokens)
    per_position_summary: List[Dict[str, Any]] = []
    for position, counter in enumerate(per_position_counters):
        per_position_summary.append(
            {
                "position": position,
                "unique_token_count": len(counter),
                "top_tokens": _counter_to_top_k(counter, min(args.top_k, 5), sum(counter.values())),
            }
        )

    summary = {
        "checkpoint_dir": str(checkpoint_dir),
        "dataset_type": args.dataset_type,
        "evaluated_sample_count": len(indices),
        "sample_indices": indices,
        "unique_predicted_action_token_count": unique_token_count,
        "total_predicted_action_token_count": total_predicted_tokens,
        "entropy_nats": _entropy_nats(flat_counter),
        "top_predicted_action_tokens": top_tokens,
        "per_position_distribution": per_position_summary,
        "example_decoded_trajectories": example_records,
    }

    output_json = PROJECT_ROOT / args.output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(f"[analysis] evaluated_sample_count={summary['evaluated_sample_count']}")
    print(f"[analysis] unique_predicted_action_token_count={unique_token_count}")
    print(f"[analysis] entropy_nats={summary['entropy_nats']:.4f}")
    print("[analysis] top_predicted_action_tokens=")
    for item in top_tokens:
        print(
            f"  token_id={item['token_id']} count={item['count']} "
            f"frequency={item['frequency']:.4f}"
        )
    print("[analysis] example_decoded_trajectories=")
    for record in example_records:
        final_point = record["predicted_trajectory"][-1]
        print(
            f"  sample_index={record['sample_index']} tokens={record['predicted_action_tokens']} "
            f"final_point={final_point}"
        )
    print(f"[analysis] summary_json={output_json}")


if __name__ == "__main__":
    main()
