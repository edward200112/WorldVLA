"""Multitask collate function with masking for heterogeneous supervision."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

from language_waypoint_planner.constants import ROUTE_TO_INDEX

from .sample import BaseDrivingSample
from .tokenizer import HashTextTokenizer


def _route_command_to_index(route_command: str) -> int:
    canonical = route_command.lower().strip()
    if canonical == "follow_lane":
        canonical = "straight"
    if canonical not in ROUTE_TO_INDEX:
        raise KeyError(f"Unknown route command: {route_command}")
    return ROUTE_TO_INDEX[canonical]


def multitask_collate(
    samples: Sequence[BaseDrivingSample],
    tokenizer: Optional[HashTextTokenizer] = None,
    rationale_max_length: int = 16,
) -> Dict[str, Any]:
    """Collate heterogeneous driving samples while preserving missing-label masks."""

    if not samples:
        raise ValueError("Cannot collate an empty batch")
    tokenizer = tokenizer or HashTextTokenizer()

    images = torch.stack([sample.images for sample in samples], dim=0)
    ego_hist = torch.stack([sample.ego_hist for sample in samples], dim=0)
    velocity = torch.stack(
        [sample.velocity if sample.velocity.ndim == 2 else sample.velocity.unsqueeze(-1) for sample in samples],
        dim=0,
    )
    acceleration = torch.stack(
        [
            sample.acceleration if sample.acceleration.ndim == 2 else sample.acceleration.unsqueeze(-1)
            for sample in samples
        ],
        dim=0,
    )
    route_command_ids = torch.tensor(
        [_route_command_to_index(sample.route_command) for sample in samples],
        dtype=torch.long,
    )
    language_input = [sample.language_input for sample in samples]

    valid_waypoints = torch.tensor([sample.valid_masks["waypoints"] for sample in samples], dtype=torch.bool)
    valid_behavior = torch.tensor([sample.valid_masks["behavior"] for sample in samples], dtype=torch.bool)
    valid_rationale = torch.tensor([sample.valid_masks["rationale"] for sample in samples], dtype=torch.bool)

    future_steps = max(
        sample.target_waypoints.shape[0] if sample.target_waypoints is not None else 0 for sample in samples
    )
    if future_steps == 0:
        future_steps = 20
    target_waypoints: List[torch.Tensor] = []
    for sample in samples:
        if sample.target_waypoints is None:
            target_waypoints.append(torch.zeros(future_steps, 2, dtype=torch.float32))
        else:
            target = sample.target_waypoints
            if target.shape[0] != future_steps:
                raise ValueError("All target_waypoints must share the same future length within a batch")
            target_waypoints.append(target)

    target_behavior = torch.tensor(
        [sample.target_behavior if sample.target_behavior is not None else 0 for sample in samples],
        dtype=torch.long,
    )
    rationale_texts = [sample.target_rationale or "" for sample in samples]
    rationale_ids, rationale_token_mask = tokenizer.encode_batch(rationale_texts, rationale_max_length)
    rationale_token_mask = rationale_token_mask & valid_rationale.unsqueeze(-1)
    rationale_ids = rationale_ids * rationale_token_mask.long()

    return {
        "images": images,
        "ego_hist": ego_hist,
        "velocity": velocity,
        "acceleration": acceleration,
        "route_command_ids": route_command_ids,
        "route_command_text": [sample.route_command for sample in samples],
        "language_input": language_input,
        "language_mask": torch.tensor([bool(text.strip()) for text in language_input], dtype=torch.bool),
        "target_waypoints": torch.stack(target_waypoints, dim=0),
        "target_behavior": target_behavior,
        "target_rationale_text": rationale_texts,
        "target_rationale_ids": rationale_ids,
        "target_rationale_token_mask": rationale_token_mask,
        "valid_masks": {
            "waypoints": valid_waypoints,
            "behavior": valid_behavior,
            "rationale": valid_rationale,
        },
    }


def build_multitask_collate_fn(
    tokenizer: Optional[HashTextTokenizer] = None,
    rationale_max_length: int = 16,
) -> Callable[[Sequence[BaseDrivingSample]], Dict[str, Any]]:
    """Build a collate function with fixed tokenizer configuration."""

    return partial(multitask_collate, tokenizer=tokenizer, rationale_max_length=rationale_max_length)
