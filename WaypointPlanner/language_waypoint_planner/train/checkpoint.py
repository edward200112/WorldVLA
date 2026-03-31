"""Checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    scaler: Optional[Any] = None,
) -> None:
    """Save model, optimizer, and scaler state."""

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[Any] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """Load a checkpoint into model, optimizer, and scaler."""

    payload = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    return payload
