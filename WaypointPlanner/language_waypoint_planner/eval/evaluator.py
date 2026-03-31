"""Offline evaluation loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from language_waypoint_planner.models import PlannerModel

from .metrics import compute_ade_fde, compute_behavior_metrics, compute_rationale_placeholder_metric
from .visualize import save_prediction_visualizations


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif isinstance(value, dict):
            moved[key] = _move_batch_to_device(value, device)
        else:
            moved[key] = value
    return moved


@torch.no_grad()
def run_evaluation(
    model: PlannerModel,
    dataloader,
    device: torch.device,
    output_dir: Path,
    save_visualizations: bool = True,
    num_visualizations: int = 8,
) -> Dict[str, float]:
    """Evaluate the model on a dataloader and optionally save qualitative visualizations."""

    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_history = {"ade": [], "fde": [], "behavior_accuracy": [], "behavior_f1": [], "rationale_exact_match": []}
    visualizations_saved = 0

    for batch in dataloader:
        cpu_batch = batch
        batch = _move_batch_to_device(batch, device)
        outputs = model(
            images=batch["images"],
            ego_hist=batch["ego_hist"],
            velocity=batch["velocity"],
            acceleration=batch["acceleration"],
            route_command_ids=batch["route_command_ids"],
            language_input=batch["language_input"],
        )
        pred_rationales = model.decode_rationale_logits(outputs.pred_rationale_logits_or_tokens)

        trajectory_metrics = compute_ade_fde(
            pred_waypoints=outputs.pred_waypoints,
            target_waypoints=batch["target_waypoints"],
            valid_mask=batch["valid_masks"]["waypoints"],
        )
        behavior_metrics = compute_behavior_metrics(
            logits=outputs.pred_behavior_logits,
            target_behavior=batch["target_behavior"],
            valid_mask=batch["valid_masks"]["behavior"],
        )
        rationale_metrics = compute_rationale_placeholder_metric(
            predictions=pred_rationales,
            targets=cpu_batch["target_rationale_text"],
            valid_mask=cpu_batch["valid_masks"]["rationale"],
        )
        for key, value in {**trajectory_metrics, **behavior_metrics, **rationale_metrics}.items():
            metric_history[key].append(value)

        if save_visualizations and visualizations_saved < num_visualizations:
            remaining = num_visualizations - visualizations_saved
            saved = save_prediction_visualizations(
                batch=cpu_batch,
                pred_waypoints=outputs.pred_waypoints.cpu(),
                pred_rationales=pred_rationales,
                output_dir=output_dir / "visualizations",
                max_items=remaining,
            )
            visualizations_saved += len(saved)

    return {
        key: float(sum(values) / len(values)) if values else 0.0
        for key, values in metric_history.items()
    }
