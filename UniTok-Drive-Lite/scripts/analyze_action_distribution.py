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
    entropy_nats,
    future_bev_difference_summary,
    quantization_trajectory_stats,
    summarize_scalar_distribution,
    token_match_summary,
    trajectory_stats,
    tensor_to_list,
)
from unitok_drive_lite.script_utils import (
    add_action_quantization_args,
    add_dataset_selection_args,
    apply_action_quantization_args,
    build_dataset_from_args,
    print_action_quantization_summary,
)
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
    add_action_quantization_args(parser)
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


def _build_lateral_bin_top_k(
    counter: Counter,
    model: UnifiedDriveModel,
    top_k: int,
) -> List[Dict[str, Any]]:
    """为 lateral bin 计数补充 center 语义。"""
    denominator = max(sum(counter.values()), 1)
    centers = model.discretizer.action_bin_centers[1].tolist()
    summary: List[Dict[str, Any]] = []
    for bin_index, count in counter.most_common(top_k):
        summary.append(
            {
                "bin_index": int(bin_index),
                "center": float(centers[int(bin_index)]),
                "count": int(count),
                "frequency": float(count / denominator),
            }
        )
    return summary


def _build_joint_action_top_k(
    counter: Counter,
    model: UnifiedDriveModel,
    top_k: int,
) -> List[Dict[str, Any]]:
    """为 joint action token 计数补充 per-dim bin / center 语义。"""
    denominator = max(sum(counter.values()), 1)
    summary: List[Dict[str, Any]] = []
    for token_id, count in counter.most_common(top_k):
        bin_indices = model.discretizer.decode_action_token_id_to_bins(int(token_id))
        summary.append(
            {
                "token_id": int(token_id),
                "count": int(count),
                "frequency": float(count / denominator),
                "bin_indices": bin_indices,
                "centers": {
                    dim_name: float(
                        model.discretizer.action_bin_centers[
                            0 if dim_name == "longitudinal" else 1,
                            bin_indices[dim_name],
                        ].item()
                    )
                    for dim_name in ("longitudinal", "lateral")
                },
            }
        )
    return summary


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
    apply_action_quantization_args(config.tokens, args)
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
    print_action_quantization_summary(model.discretizer.get_action_quantization_summary())
    print(f"[data] dataset_type={args.dataset_type} eval_samples={len(indices)}")

    gt_counter: Counter = Counter()
    predicted_counter: Counter = Counter()
    gt_lateral_bin_counter: Counter = Counter()
    predicted_lateral_bin_counter: Counter = Counter()
    near_zero_lateral_bin_counter: Counter = Counter()
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
    gt_quantization_stats_records: List[Dict[str, Any]] = []
    raw_lateral_values: List[float] = []
    gt_quantized_lateral_values: List[float] = []
    predicted_quantized_lateral_values: List[float] = []

    for sample_index in indices:
        sample = dataset[sample_index]
        rollout_output = greedy_rollout(model, sample, device)

        raw_actions = sample.future_actions.detach().cpu().to(torch.float32)
        gt_action_tokens = [int(token_id) for token_id in model.discretizer.encode_future_actions(raw_actions)]
        predicted_action_tokens = [int(token_id) for token_id in rollout_output["predicted_action_tokens"]]
        match = token_match_summary(gt_action_tokens, predicted_action_tokens)

        gt_action_bin_indices = model.discretizer.quantize_future_action_bin_indices(raw_actions).detach().cpu()
        predicted_action_bin_indices = torch.tensor(
            [
                [
                    model.discretizer.decode_action_token_id_to_bins(token_id)["longitudinal"],
                    model.discretizer.decode_action_token_id_to_bins(token_id)["lateral"],
                ]
                for token_id in predicted_action_tokens
            ],
            dtype=torch.long,
        )
        gt_quantized_action_deltas = model.discretizer.decode_action_token_ids(gt_action_tokens)
        predicted_quantized_action_deltas = model.discretizer.decode_action_token_ids(predicted_action_tokens)
        gt_quantized_trajectory = model.discretizer.decode_action_tokens_to_trajectory(gt_action_tokens)
        predicted_quantized_trajectory = rollout_output["predicted_trajectory"].detach().cpu().to(torch.float32)
        gt_raw_trajectory = torch.cumsum(raw_actions, dim=0)
        # 当前模型只输出离散 action token，因此“predicted raw trajectory”与量化解码轨迹相同。
        predicted_raw_trajectory = predicted_quantized_trajectory.clone()

        gt_quantization_stats = quantization_trajectory_stats(
            gt_quantized_trajectory,
            gt_raw_trajectory,
            near_zero_lateral_threshold=config.tokens.action_lateral_near_zero_threshold,
        )
        quantized_stats = trajectory_stats(predicted_quantized_trajectory, gt_quantized_trajectory)
        raw_stats = trajectory_stats(predicted_raw_trajectory, gt_raw_trajectory)
        gt_future_bevs = sample.future_bevs.detach().cpu().to(torch.float32)
        predicted_future_bevs = _stack_predicted_future_bevs(rollout_output["predicted_future_bevs"])
        future_bev_stats = future_bev_difference_summary(predicted_future_bevs, gt_future_bevs)

        raw_lateral = raw_actions[:, 1]
        gt_quantized_lateral = gt_quantized_action_deltas[:, 1]
        predicted_quantized_lateral = predicted_quantized_action_deltas[:, 1]
        near_zero_lateral_mask = raw_lateral.abs().le(config.tokens.action_lateral_near_zero_threshold)
        raw_lateral_values.extend(float(value) for value in raw_lateral.tolist())
        gt_quantized_lateral_values.extend(float(value) for value in gt_quantized_lateral.tolist())
        predicted_quantized_lateral_values.extend(float(value) for value in predicted_quantized_lateral.tolist())

        for position, (gt_token_id, predicted_token_id) in enumerate(zip(gt_action_tokens, predicted_action_tokens)):
            gt_counter[gt_token_id] += 1
            predicted_counter[predicted_token_id] += 1
            gt_per_position_counters[position][gt_token_id] += 1
            predicted_per_position_counters[position][predicted_token_id] += 1
            per_position_correct_count[position] += int(gt_token_id == predicted_token_id)
            gt_lateral_bin_counter[int(gt_action_bin_indices[position, 1].item())] += 1
            predicted_lateral_bin_counter[int(predicted_action_bin_indices[position, 1].item())] += 1
            if bool(near_zero_lateral_mask[position].item()):
                near_zero_lateral_bin_counter[int(gt_action_bin_indices[position, 1].item())] += 1

        total_token_correct += match["token_match_count"]
        total_token_count += len(gt_action_tokens)
        exact_sequence_match_count += int(match["exact_sequence_match"])
        gt_quantization_stats_records.append(gt_quantization_stats)
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
                "gt_action_bin_indices": tensor_to_list(gt_action_bin_indices),
                "predicted_action_bin_indices": tensor_to_list(predicted_action_bin_indices),
                "gt_quantized_action_deltas": tensor_to_list(gt_quantized_action_deltas),
                "predicted_quantized_action_deltas": tensor_to_list(predicted_quantized_action_deltas),
                "token_match_count": match["token_match_count"],
                "token_match_ratio": match["token_match_ratio"],
                "exact_sequence_match": match["exact_sequence_match"],
                "per_position_token_correctness": match["per_position_token_correctness"],
                "gt_quantized_trajectory": tensor_to_list(gt_quantized_trajectory),
                "predicted_quantized_trajectory": tensor_to_list(predicted_quantized_trajectory),
                "gt_raw_trajectory": tensor_to_list(gt_raw_trajectory),
                "predicted_raw_trajectory": tensor_to_list(predicted_raw_trajectory),
                "predicted_raw_trajectory_source": "decoded_action_tokens_only_no_continuous_head",
                "gt_quantization_trajectory_stats": gt_quantization_stats,
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
    raw_lateral_distribution = summarize_scalar_distribution(
        torch.tensor(raw_lateral_values, dtype=torch.float32),
        near_zero_threshold=config.tokens.action_lateral_near_zero_threshold,
    )
    gt_quantized_lateral_distribution = summarize_scalar_distribution(
        torch.tensor(gt_quantized_lateral_values, dtype=torch.float32),
        near_zero_threshold=config.tokens.action_lateral_near_zero_threshold,
    )
    predicted_quantized_lateral_distribution = summarize_scalar_distribution(
        torch.tensor(predicted_quantized_lateral_values, dtype=torch.float32),
        near_zero_threshold=config.tokens.action_lateral_near_zero_threshold,
    )
    top_gt_joint_action_tokens = _build_joint_action_top_k(gt_counter, model, args.top_k)
    top_pred_joint_action_tokens = _build_joint_action_top_k(predicted_counter, model, args.top_k)

    summary = {
        "checkpoint_dir": str(checkpoint_dir),
        "dataset_type": args.dataset_type,
        "evaluated_sample_count": len(indices),
        "sample_indices": indices,
        "action_quantization": model.discretizer.get_action_quantization_summary(),
        "unique_scene_count": len(per_scene_summary),
        "per_scene_summary": per_scene_summary,
        "gt_unique_action_token_count": len(gt_counter),
        "pred_unique_action_token_count": len(predicted_counter),
        "gt_entropy_nats": entropy_nats(gt_counter),
        "pred_entropy_nats": entropy_nats(predicted_counter),
        "top_gt_action_tokens": top_gt_joint_action_tokens,
        "top_pred_action_tokens": top_pred_joint_action_tokens,
        "top_gt_joint_action_tokens": top_gt_joint_action_tokens,
        "top_pred_joint_action_tokens": top_pred_joint_action_tokens,
        "top_gt_lateral_bins": _build_lateral_bin_top_k(gt_lateral_bin_counter, model, args.top_k),
        "top_pred_lateral_bins": _build_lateral_bin_top_k(predicted_lateral_bin_counter, model, args.top_k),
        "near_zero_lateral_bin_usage": _build_lateral_bin_top_k(near_zero_lateral_bin_counter, model, args.top_k),
        "raw_lateral_distribution": raw_lateral_distribution,
        "gt_quantized_lateral_distribution": gt_quantized_lateral_distribution,
        "predicted_quantized_lateral_distribution": predicted_quantized_lateral_distribution,
        "per_position_gt_distribution": build_per_position_distribution(gt_per_position_counters, args.top_k),
        "per_position_pred_distribution": build_per_position_distribution(predicted_per_position_counters, args.top_k),
        "token_accuracy_overall": token_accuracy_overall,
        "token_accuracy_per_position": token_accuracy_per_position,
        "exact_sequence_match_count": exact_sequence_match_count,
        "exact_sequence_match_ratio": float(exact_sequence_match_count / max(len(indices), 1)),
        "gt_quantization_trajectory_mae_mean": _mean_from_records(gt_quantization_stats_records, "mean_abs_error"),
        "gt_quantization_final_step_l2_mean": _mean_from_records(gt_quantization_stats_records, "final_step_l2"),
        "gt_quantization_longitudinal_mean_abs_error_mean": _mean_from_records(
            gt_quantization_stats_records,
            "longitudinal_mean_abs_error",
        ),
        "gt_quantization_lateral_mean_abs_error_mean": _mean_from_records(
            gt_quantization_stats_records,
            "lateral_mean_abs_error",
        ),
        "gt_quantization_longitudinal_final_step_abs_error_mean": _mean_from_records(
            gt_quantization_stats_records,
            "longitudinal_final_step_abs_error",
        ),
        "gt_quantization_lateral_final_step_abs_error_mean": _mean_from_records(
            gt_quantization_stats_records,
            "lateral_final_step_abs_error",
        ),
        "gt_quantization_near_zero_lateral_step_ratio_mean": _mean_from_records(
            gt_quantization_stats_records,
            "near_zero_lateral_step_ratio",
        ),
        "gt_quantization_lateral_sign_flip_ratio_mean": _mean_from_records(
            gt_quantization_stats_records,
            "lateral_sign_flip_ratio",
        ),
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
        f"[analysis] gt_quantization_final_step_l2_mean={summary['gt_quantization_final_step_l2_mean']:.4f} "
        f"quantized_final_step_l2_mean={summary['quantized_final_step_l2_mean']:.4f} "
        f"raw_final_step_l2_mean={summary['raw_final_step_l2_mean']:.4f}"
    )
    print(
        f"[analysis] gt_quantization_longitudinal_mae_mean="
        f"{summary['gt_quantization_longitudinal_mean_abs_error_mean']:.4f} "
        f"gt_quantization_lateral_mae_mean={summary['gt_quantization_lateral_mean_abs_error_mean']:.4f} "
        f"gt_quantization_lateral_sign_flip_ratio_mean={summary['gt_quantization_lateral_sign_flip_ratio_mean']:.4f}"
    )
    print(
        f"[analysis] raw_lateral_near_zero_ratio={summary['raw_lateral_distribution']['near_zero_ratio']:.4f} "
        f"gt_quantized_lateral_near_zero_ratio={summary['gt_quantized_lateral_distribution']['near_zero_ratio']:.4f} "
        f"pred_quantized_lateral_near_zero_ratio={summary['predicted_quantized_lateral_distribution']['near_zero_ratio']:.4f}"
    )
    print(f"[analysis] unique_scene_count={summary['unique_scene_count']}")
    print("[analysis] top_gt_action_tokens=")
    for item in summary["top_gt_action_tokens"]:
        print(
            f"  token_id={item['token_id']} count={item['count']} "
            f"frequency={item['frequency']:.4f} "
            f"longitudinal_bin={item['bin_indices']['longitudinal']} "
            f"lateral_bin={item['bin_indices']['lateral']} "
            f"centers={item['centers']}"
        )
    print("[analysis] top_pred_action_tokens=")
    for item in summary["top_pred_action_tokens"]:
        print(
            f"  token_id={item['token_id']} count={item['count']} "
            f"frequency={item['frequency']:.4f} "
            f"longitudinal_bin={item['bin_indices']['longitudinal']} "
            f"lateral_bin={item['bin_indices']['lateral']} "
            f"centers={item['centers']}"
        )
    print("[analysis] top_gt_lateral_bins=")
    for item in summary["top_gt_lateral_bins"]:
        print(
            f"  bin_index={item['bin_index']} center={item['center']:.4f} "
            f"count={item['count']} frequency={item['frequency']:.4f}"
        )
    print("[analysis] top_pred_lateral_bins=")
    for item in summary["top_pred_lateral_bins"]:
        print(
            f"  bin_index={item['bin_index']} center={item['center']:.4f} "
            f"count={item['count']} frequency={item['frequency']:.4f}"
        )
    print("[analysis] near_zero_lateral_bin_usage=")
    for item in summary["near_zero_lateral_bin_usage"]:
        print(
            f"  bin_index={item['bin_index']} center={item['center']:.4f} "
            f"count={item['count']} frequency={item['frequency']:.4f}"
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
