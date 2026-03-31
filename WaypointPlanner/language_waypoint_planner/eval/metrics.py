"""Evaluation metrics for trajectories, behavior, and rationale text."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import torch

from language_waypoint_planner.constants import BEHAVIOR_LABELS
from language_waypoint_planner.data.label_utils import normalize_rationale


def compute_ade_fde(
    pred_waypoints: torch.Tensor,
    target_waypoints: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Dict[str, float]:
    """Compute Average Displacement Error and Final Displacement Error."""

    if not valid_mask.any():
        return {"ade": 0.0, "fde": 0.0}
    errors = torch.linalg.norm(pred_waypoints - target_waypoints, dim=-1)
    ade = errors.mean(dim=1)[valid_mask].mean().item()
    fde = errors[:, -1][valid_mask].mean().item()
    return {"ade": float(ade), "fde": float(fde)}


def _macro_f1(confusion: torch.Tensor) -> float:
    per_class: List[float] = []
    for class_index in range(confusion.shape[0]):
        tp = confusion[class_index, class_index]
        fp = confusion[:, class_index].sum() - tp
        fn = confusion[class_index].sum() - tp
        denom_precision = tp + fp
        denom_recall = tp + fn
        precision = float(tp / denom_precision) if denom_precision > 0 else 0.0
        recall = float(tp / denom_recall) if denom_recall > 0 else 0.0
        if precision + recall == 0.0:
            per_class.append(0.0)
        else:
            per_class.append(2.0 * precision * recall / (precision + recall))
    return sum(per_class) / len(per_class)


def compute_behavior_metrics(
    logits: torch.Tensor,
    target_behavior: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Dict[str, float]:
    """Compute behavior accuracy and macro F1."""

    if not valid_mask.any():
        return {"behavior_accuracy": 0.0, "behavior_f1": 0.0}
    preds = logits.argmax(dim=-1)[valid_mask]
    targets = target_behavior[valid_mask]
    accuracy = float((preds == targets).float().mean().item())
    confusion = torch.zeros(len(BEHAVIOR_LABELS), len(BEHAVIOR_LABELS), dtype=torch.float32)
    for pred, target in zip(preds.tolist(), targets.tolist()):
        confusion[target, pred] += 1.0
    return {
        "behavior_accuracy": accuracy,
        "behavior_f1": _macro_f1(confusion),
    }


def compute_rationale_placeholder_metric(
    predictions: Sequence[str],
    targets: Sequence[str],
    valid_mask: torch.Tensor,
) -> Dict[str, float]:
    """Simple normalized exact-match placeholder for rationale generation."""

    if not valid_mask.any():
        return {"rationale_exact_match": 0.0}
    normalized_pairs = [
        (normalize_rationale(pred), normalize_rationale(target))
        for pred, target, is_valid in zip(predictions, targets, valid_mask.tolist())
        if is_valid
    ]
    exact_match = sum(int(pred == target) for pred, target in normalized_pairs) / len(normalized_pairs)
    return {"rationale_exact_match": float(exact_match)}
