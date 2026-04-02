from __future__ import annotations

from collections import Counter
import math
from typing import Any, Dict, Iterable, Sequence

import torch

ACTION_DIM_NAMES: tuple[str, str] = ("longitudinal", "lateral")
ACTION_DIM_TO_INDEX = {name: index for index, name in enumerate(ACTION_DIM_NAMES)}


def tensor_to_list(value: torch.Tensor) -> list[Any]:
    """把 tensor 安全转成 Python 列表。"""
    return value.detach().cpu().tolist()


def counter_to_top_k(counter: Counter, top_k: int, denominator: int) -> list[Dict[str, Any]]:
    """把 Counter 转成可 JSON 序列化的 top-k 摘要。"""
    top_items: list[Dict[str, Any]] = []
    for token_id, count in counter.most_common(top_k):
        top_items.append(
            {
                "token_id": int(token_id),
                "count": int(count),
                "frequency": float(count / max(denominator, 1)),
            }
        )
    return top_items


def entropy_nats(counter: Counter) -> float:
    """计算 token 分布熵，单位为 nats。"""
    total = sum(counter.values())
    if total <= 0:
        return 0.0

    entropy = 0.0
    for count in counter.values():
        probability = count / total
        entropy -= probability * math.log(probability)
    return float(entropy)


def token_match_summary(
    gt_tokens: Sequence[int],
    predicted_tokens: Sequence[int],
) -> Dict[str, Any]:
    """统计 token 级别的 GT-vs-Pred 匹配情况。"""
    if len(gt_tokens) != len(predicted_tokens):
        raise ValueError(
            "GT token 与预测 token 长度不一致: "
            f"gt={len(gt_tokens)} predicted={len(predicted_tokens)}"
        )

    correctness = [int(int(predicted) == int(target)) for target, predicted in zip(gt_tokens, predicted_tokens)]
    match_count = sum(correctness)
    horizon = len(gt_tokens)
    return {
        "token_match_count": int(match_count),
        "token_match_ratio": float(match_count / max(horizon, 1)),
        "per_position_token_correctness": correctness,
        "exact_sequence_match": bool(match_count == horizon),
    }


def trajectory_stats(predicted: torch.Tensor, target: torch.Tensor) -> Dict[str, Any]:
    """返回一组简洁的轨迹对比指标。"""
    predicted_cpu = predicted.detach().cpu().to(torch.float32)
    target_cpu = target.detach().cpu().to(torch.float32)
    if predicted_cpu.shape != target_cpu.shape:
        raise ValueError(
            f"预测轨迹与目标轨迹 shape 不一致: predicted={tuple(predicted_cpu.shape)} "
            f"target={tuple(target_cpu.shape)}"
        )

    difference = predicted_cpu - target_cpu
    absolute_difference = difference.abs()
    final_delta = predicted_cpu[-1] - target_cpu[-1]
    summary = {
        "predicted_shape": list(predicted_cpu.shape),
        "target_shape": list(target_cpu.shape),
        "mean_abs_error": float(absolute_difference.mean().item()),
        "max_abs_error": float(absolute_difference.max().item()),
        "final_step_l2": float(torch.linalg.vector_norm(final_delta).item()),
        "predicted_final_point": tensor_to_list(predicted_cpu[-1]),
        "target_final_point": tensor_to_list(target_cpu[-1]),
    }
    if predicted_cpu.ndim >= 2 and predicted_cpu.shape[-1] >= len(ACTION_DIM_NAMES):
        for dim_name, dim_index in ACTION_DIM_TO_INDEX.items():
            summary[f"{dim_name}_mean_abs_error"] = float(absolute_difference[:, dim_index].mean().item())
            summary[f"{dim_name}_final_step_abs_error"] = float(final_delta[dim_index].abs().item())
    return summary


def summarize_scalar_distribution(
    values: torch.Tensor,
    *,
    near_zero_threshold: float | None = None,
) -> Dict[str, Any]:
    """返回一维连续值分布的摘要。"""
    values_cpu = values.detach().cpu().to(torch.float32).flatten()
    if values_cpu.numel() == 0:
        summary = {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "abs_mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p05": 0.0,
            "p50": 0.0,
            "p95": 0.0,
        }
    else:
        summary = {
            "count": int(values_cpu.numel()),
            "mean": float(values_cpu.mean().item()),
            "std": float(values_cpu.std(unbiased=False).item()),
            "abs_mean": float(values_cpu.abs().mean().item()),
            "min": float(values_cpu.min().item()),
            "max": float(values_cpu.max().item()),
            "p05": float(torch.quantile(values_cpu, 0.05).item()),
            "p50": float(torch.quantile(values_cpu, 0.50).item()),
            "p95": float(torch.quantile(values_cpu, 0.95).item()),
        }

    if near_zero_threshold is not None:
        if values_cpu.numel() == 0:
            near_zero_mask = torch.zeros_like(values_cpu, dtype=torch.bool)
        else:
            near_zero_mask = values_cpu.abs().le(float(near_zero_threshold))
        summary.update(
            {
                "near_zero_threshold": float(near_zero_threshold),
                "near_zero_count": int(near_zero_mask.sum().item()),
                "near_zero_ratio": float(near_zero_mask.float().mean().item()) if values_cpu.numel() > 0 else 0.0,
            }
        )
    return summary


def quantization_trajectory_stats(
    quantized_trajectory: torch.Tensor,
    raw_trajectory: torch.Tensor,
    *,
    near_zero_lateral_threshold: float,
) -> Dict[str, Any]:
    """统计 raw -> quantized 轨迹误差，重点检查 near-zero lateral 是否被扭坏。"""
    summary = trajectory_stats(quantized_trajectory, raw_trajectory)
    raw_cpu = raw_trajectory.detach().cpu().to(torch.float32)
    quantized_cpu = quantized_trajectory.detach().cpu().to(torch.float32)
    if raw_cpu.shape != quantized_cpu.shape:
        raise ValueError(
            "raw trajectory 与 quantized trajectory shape 不一致: "
            f"raw={tuple(raw_cpu.shape)} quantized={tuple(quantized_cpu.shape)}"
        )
    if raw_cpu.ndim != 2 or raw_cpu.shape[-1] < 2:
        raise ValueError(f"动作轨迹必须是 [T, 2]，当前收到 {tuple(raw_cpu.shape)}")

    raw_lateral = raw_cpu[:, ACTION_DIM_TO_INDEX["lateral"]]
    quantized_lateral = quantized_cpu[:, ACTION_DIM_TO_INDEX["lateral"]]
    near_zero_mask = raw_lateral.abs().le(float(near_zero_lateral_threshold))
    sign_flip_mask = raw_lateral * quantized_lateral < 0.0

    near_zero_count = int(near_zero_mask.sum().item())
    summary.update(
        {
            "near_zero_lateral_threshold": float(near_zero_lateral_threshold),
            "near_zero_lateral_step_count": near_zero_count,
            "near_zero_lateral_step_ratio": float(near_zero_mask.float().mean().item()),
            "lateral_sign_flip_count": int(sign_flip_mask.sum().item()),
            "lateral_sign_flip_ratio": float(sign_flip_mask.float().mean().item()),
            "near_zero_lateral_sign_flip_count": int((sign_flip_mask & near_zero_mask).sum().item()),
            "near_zero_lateral_sign_flip_ratio": (
                float((sign_flip_mask & near_zero_mask).sum().item() / near_zero_count)
                if near_zero_count > 0
                else 0.0
            ),
        }
    )
    return summary


def bev_occupancy_stats(bev: torch.Tensor) -> Dict[str, Any]:
    """返回单帧或多帧 BEV 的占用统计。"""
    bev_cpu = bev.detach().cpu().to(torch.float32)
    nonzero_mask = bev_cpu.gt(0.0)
    occupied_mask = bev_cpu.gt(0.5)
    return {
        "shape": list(bev_cpu.shape),
        "sum": float(bev_cpu.sum().item()),
        "mean": float(bev_cpu.mean().item()),
        "max": float(bev_cpu.max().item()),
        "min": float(bev_cpu.min().item()),
        "nonzero_fraction": float(nonzero_mask.float().mean().item()),
        "occupied_fraction": float(occupied_mask.float().mean().item()),
    }


def future_bev_difference_summary(
    predicted_future_bevs: torch.Tensor,
    gt_future_bevs: torch.Tensor,
) -> Dict[str, Any]:
    """比较 GT 与预测 future BEV 的占用差异。"""
    predicted_cpu = predicted_future_bevs.detach().cpu().to(torch.float32)
    gt_cpu = gt_future_bevs.detach().cpu().to(torch.float32)
    if predicted_cpu.shape != gt_cpu.shape:
        raise ValueError(
            "GT future BEV 与预测 future BEV shape 不一致: "
            f"gt={tuple(gt_cpu.shape)} predicted={tuple(predicted_cpu.shape)}"
        )

    absolute_difference = (predicted_cpu - gt_cpu).abs()
    per_frame: list[Dict[str, Any]] = []
    identical_frame_count = 0
    for frame_index in range(gt_cpu.shape[0]):
        gt_frame = gt_cpu[frame_index]
        predicted_frame = predicted_cpu[frame_index]
        frame_difference = absolute_difference[frame_index]
        if torch.allclose(gt_frame, predicted_frame):
            identical_frame_count += 1
        per_frame.append(
            {
                "frame_index": frame_index,
                "mae": float(frame_difference.mean().item()),
                "max_abs_diff": float(frame_difference.max().item()),
                "gt_occupancy": bev_occupancy_stats(gt_frame),
                "predicted_occupancy": bev_occupancy_stats(predicted_frame),
            }
        )

    return {
        "shape": list(gt_cpu.shape),
        "overall_mean_abs_diff": float(absolute_difference.mean().item()),
        "overall_max_abs_diff": float(absolute_difference.max().item()),
        "identical_frame_count": int(identical_frame_count),
        "per_frame": per_frame,
    }


def build_per_position_distribution(
    counters: Iterable[Counter],
    top_k: int,
) -> list[Dict[str, Any]]:
    """构造每个时间位置的 token 分布摘要。"""
    summary: list[Dict[str, Any]] = []
    for position, counter in enumerate(counters):
        summary.append(
            {
                "position": position,
                "unique_token_count": len(counter),
                "top_tokens": counter_to_top_k(counter, min(top_k, 5), sum(counter.values())),
            }
        )
    return summary
