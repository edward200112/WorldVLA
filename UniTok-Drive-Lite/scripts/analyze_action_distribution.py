"""最小 GT-vs-Pred action 评估脚本。"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unitok_drive_lite import UnifiedDriveModel, build_default_config
from unitok_drive_lite.eval_utils import (
    build_per_position_distribution,
    counter_to_top_k,
    entropy_nats,
    future_bev_difference_summary,
    token_match_summary,
    trajectory_stats,
    tensor_to_list,
)
from unitok_drive_lite.script_utils import add_dataset_selection_args, build_dataset_from_args
from unitok_drive_lite.train_utils import greedy_rollout, seed_everything


def parse_args() -> argparse.Namespace:
    """解析 GT-vs-Pred 评估参数。"""
    parser = argparse.ArgumentParser(description="分析最小主链路的 GT-vs-Pred action token 表现。")
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


def _sample_indices(length: int, stride: int, limit: int) -> List[int]:
    """生成分析用样本索引。"""
    if stride <= 0:
        raise ValueError(f"--sample_stride 必须大于 0，当前收到 {stride}")
    if limit <= 0:
        raise ValueError(f"--max_eval_samples 必须大于 0，当前收到 {limit}")
    return list(range(0, length, stride))[:limit]


def _stack_predicted_future_bevs(predicted_future_bevs: List[torch.Tensor]) -> torch.Tensor:
    """把 rollout 返回的 future BEV 列表堆叠成 [F, 1, H, W]。"""
    return torch.stack([frame.detach().cpu().to(torch.float32) for frame in predicted_future_bevs], dim=0)


def _scene_key(metadata: Dict[str, Any]) -> Tuple[str, str]:
    """提取 scene token / name。"""
    return str(metadata.get("scene_token", "unknown")), str(metadata.get("scene_name", "unknown"))


def _mean_from_records(records: List[Dict[str, Any]], field_name: str) -> float:
    """从一组 sample record 中取某个标量字段均值。"""
    if not records:
        return 0.0
    return float(sum(float(record[field_name]) for record in records) / len(records))


def _print_example_mismatches(sample_records: List[Dict[str, Any]], max_examples: int = 3) -> None:
    """打印少量 GT-vs-Pred 不匹配样例。"""
    mismatch_records = [record for record in sample_records if not record["exact_sequence_match"]]
    if not mismatch_records:
        print("[analysis] 所有评估样本都达到了 exact sequence match。")
        return

    print("[analysis] example_mismatches=")
    for record in mismatch_records[:max_examples]:
        print(
            f"  sample_index={record['sample_index']} scene={record['metadata'].get('scene_name', 'unknown')} "
            f"token_match_ratio={record['token_match_ratio']:.3f}"
        )
        print(f"    gt={record['gt_action_tokens']}")
        print(f"    pred={record['predicted_action_tokens']}")


def main() -> None:
    """运行 GT-vs-Pred action 评估。"""
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

    gt_counter: Counter = Counter()
    predicted_counter: Counter = Counter()
    gt_per_position_counters = [Counter() for _ in range(config.tokens.future_action_horizon)]
    predicted_per_position_counters = [Counter() for _ in range(config.tokens.future_action_horizon)]

    total_token_correct = 0
    total_token_count = 0
    exact_sequence_match_count = 0
    per_position_correct_count = [0 for _ in range(config.tokens.future_action_horizon)]

    scene_aggregates: Dict[Tuple[str, str], Dict[str, Any]] = {}
    sample_records: List[Dict[str, Any]] = []
    quantized_stats_records: List[Dict[str, Any]] = []
    raw_stats_records: List[Dict[str, Any]] = []

    for sample_index in indices:
        sample = dataset[sample_index]
        rollout_output = greedy_rollout(model, sample, device)

        gt_action_tokens = [int(token_id) for token_id in model.discretizer.encode_future_actions(sample.future_actions)]
        predicted_action_tokens = [int(token_id) for token_id in rollout_output["predicted_action_tokens"]]
        match = token_match_summary(gt_action_tokens, predicted_action_tokens)

        gt_quantized_trajectory = model.discretizer.decode_action_tokens_to_trajectory(gt_action_tokens)
        predicted_quantized_trajectory = rollout_output["predicted_trajectory"].detach().cpu().to(torch.float32)
        gt_raw_trajectory = torch.cumsum(sample.future_actions.detach().cpu().to(torch.float32), dim=0)
        # 当前模型只输出离散 action token，因此“predicted raw trajectory”与量化解码轨迹相同。
        predicted_raw_trajectory = predicted_quantized_trajectory.clone()

        quantized_stats = trajectory_stats(predicted_quantized_trajectory, gt_quantized_trajectory)
        raw_stats = trajectory_stats(predicted_raw_trajectory, gt_raw_trajectory)
        gt_future_bevs = sample.future_bevs.detach().cpu().to(torch.float32)
        predicted_future_bevs = _stack_predicted_future_bevs(rollout_output["predicted_future_bevs"])
        future_bev_stats = future_bev_difference_summary(predicted_future_bevs, gt_future_bevs)

        for position, (gt_token_id, predicted_token_id) in enumerate(zip(gt_action_tokens, predicted_action_tokens)):
            gt_counter[gt_token_id] += 1
            predicted_counter[predicted_token_id] += 1
            gt_per_position_counters[position][gt_token_id] += 1
            predicted_per_position_counters[position][predicted_token_id] += 1
            per_position_correct_count[position] += int(gt_token_id == predicted_token_id)

        total_token_correct += match["token_match_count"]
        total_token_count += len(gt_action_tokens)
        exact_sequence_match_count += int(match["exact_sequence_match"])
        quantized_stats_records.append(quantized_stats)
        raw_stats_records.append(raw_stats)

        scene_token, scene_name = _scene_key(sample.metadata)
        scene_entry = scene_aggregates.setdefault(
            (scene_token, scene_name),
            {
                "scene_token": scene_token,
                "scene_name": scene_name,
                "sample_count": 0,
                "token_correct": 0,
                "token_total": 0,
                "exact_sequence_match_count": 0,
            },
        )
        scene_entry["sample_count"] += 1
        scene_entry["token_correct"] += match["token_match_count"]
        scene_entry["token_total"] += len(gt_action_tokens)
        scene_entry["exact_sequence_match_count"] += int(match["exact_sequence_match"])

        sample_records.append(
            {
                "sample_index": sample_index,
                "metadata": sample.metadata,
                "gt_action_tokens": gt_action_tokens,
                "predicted_action_tokens": predicted_action_tokens,
                "token_match_count": match["token_match_count"],
                "token_match_ratio": match["token_match_ratio"],
                "exact_sequence_match": match["exact_sequence_match"],
                "per_position_token_correctness": match["per_position_token_correctness"],
                "gt_quantized_trajectory": tensor_to_list(gt_quantized_trajectory),
                "predicted_quantized_trajectory": tensor_to_list(predicted_quantized_trajectory),
                "gt_raw_trajectory": tensor_to_list(gt_raw_trajectory),
                "predicted_raw_trajectory": tensor_to_list(predicted_raw_trajectory),
                "predicted_raw_trajectory_source": "decoded_action_tokens",
                "quantized_trajectory_stats": quantized_stats,
                "raw_trajectory_stats": raw_stats,
                "future_bev_difference_summary": future_bev_stats,
            }
        )

    per_scene_summary: List[Dict[str, Any]] = []
    for scene_summary in scene_aggregates.values():
        token_total = max(int(scene_summary["token_total"]), 1)
        sample_count = max(int(scene_summary["sample_count"]), 1)
        per_scene_summary.append(
            {
                "scene_token": scene_summary["scene_token"],
                "scene_name": scene_summary["scene_name"],
                "sample_count": int(scene_summary["sample_count"]),
                "token_accuracy_overall": float(scene_summary["token_correct"] / token_total),
                "exact_sequence_match_ratio": float(scene_summary["exact_sequence_match_count"] / sample_count),
            }
        )
    per_scene_summary.sort(key=lambda item: item["sample_count"], reverse=True)

    token_accuracy_overall = float(total_token_correct / max(total_token_count, 1))
    token_accuracy_per_position = [
        float(correct_count / max(len(indices), 1)) for correct_count in per_position_correct_count
    ]

    summary = {
        "checkpoint_dir": str(checkpoint_dir),
        "dataset_type": args.dataset_type,
        "evaluated_sample_count": len(indices),
        "sample_indices": indices,
        "unique_scene_count": len(per_scene_summary),
        "per_scene_summary": per_scene_summary,
        "gt_unique_action_token_count": len(gt_counter),
        "pred_unique_action_token_count": len(predicted_counter),
        "gt_entropy_nats": entropy_nats(gt_counter),
        "pred_entropy_nats": entropy_nats(predicted_counter),
        "top_gt_action_tokens": counter_to_top_k(gt_counter, args.top_k, sum(gt_counter.values())),
        "top_pred_action_tokens": counter_to_top_k(
            predicted_counter,
            args.top_k,
            sum(predicted_counter.values()),
        ),
        "per_position_gt_distribution": build_per_position_distribution(gt_per_position_counters, args.top_k),
        "per_position_pred_distribution": build_per_position_distribution(predicted_per_position_counters, args.top_k),
        "token_accuracy_overall": token_accuracy_overall,
        "token_accuracy_per_position": token_accuracy_per_position,
        "exact_sequence_match_count": exact_sequence_match_count,
        "exact_sequence_match_ratio": float(exact_sequence_match_count / max(len(indices), 1)),
        "quantized_trajectory_mae_mean": _mean_from_records(quantized_stats_records, "mean_abs_error"),
        "quantized_final_step_l2_mean": _mean_from_records(quantized_stats_records, "final_step_l2"),
        "raw_trajectory_mae_mean": _mean_from_records(raw_stats_records, "mean_abs_error"),
        "raw_final_step_l2_mean": _mean_from_records(raw_stats_records, "final_step_l2"),
        "per_sample_results": sample_records,
    }

    output_json = PROJECT_ROOT / args.output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(f"[analysis] evaluated_sample_count={summary['evaluated_sample_count']}")
    print(
        f"[analysis] gt_entropy_nats={summary['gt_entropy_nats']:.4f} "
        f"pred_entropy_nats={summary['pred_entropy_nats']:.4f}"
    )
    print(
        f"[analysis] token_accuracy_overall={summary['token_accuracy_overall']:.4f} "
        f"exact_sequence_match_ratio={summary['exact_sequence_match_ratio']:.4f}"
    )
    print(
        f"[analysis] quantized_final_step_l2_mean={summary['quantized_final_step_l2_mean']:.4f} "
        f"raw_final_step_l2_mean={summary['raw_final_step_l2_mean']:.4f}"
    )
    print(f"[analysis] unique_scene_count={summary['unique_scene_count']}")
    print("[analysis] top_gt_action_tokens=")
    for item in summary["top_gt_action_tokens"]:
        print(
            f"  token_id={item['token_id']} count={item['count']} "
            f"frequency={item['frequency']:.4f}"
        )
    print("[analysis] top_pred_action_tokens=")
    for item in summary["top_pred_action_tokens"]:
        print(
            f"  token_id={item['token_id']} count={item['count']} "
            f"frequency={item['frequency']:.4f}"
        )
    if per_scene_summary:
        print("[analysis] per_scene_summary=")
        for item in per_scene_summary[:5]:
            print(
                f"  scene={item['scene_name']} sample_count={item['sample_count']} "
                f"token_accuracy={item['token_accuracy_overall']:.4f}"
            )
    _print_example_mismatches(sample_records)
    print(f"[analysis] summary_json={output_json}")


if __name__ == "__main__":
    main()
