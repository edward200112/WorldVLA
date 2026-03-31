"""Training loop implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch

from language_waypoint_planner.configs.schema import ExperimentConfig
from language_waypoint_planner.eval.evaluator import run_evaluation
from language_waypoint_planner.losses import PlannerLossComputer
from language_waypoint_planner.models import PlannerModel

from .checkpoint import load_checkpoint, save_checkpoint
from .dataset_factory import build_dataloaders
from .logger import LoggerCollection, StdoutLogger, TensorBoardLogger
from .seed import set_deterministic_seed


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Recursively move tensors in a batch to a device."""

    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif isinstance(value, dict):
            moved[key] = move_batch_to_device(value, device)
        else:
            moved[key] = value
    return moved


def _build_logger(output_dir: Path, use_tensorboard: bool) -> LoggerCollection:
    loggers = [StdoutLogger()]
    if use_tensorboard:
        loggers.append(TensorBoardLogger(output_dir / "tensorboard"))
    return LoggerCollection(loggers)


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    return torch.autocast(device_type=device.type, enabled=False)


def train_one_epoch(
    model: PlannerModel,
    loss_computer: PlannerLossComputer,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    grad_clip_norm: float,
    use_amp: bool,
    logger: Optional[LoggerCollection] = None,
    global_step: int = 0,
    log_every_n_steps: int = 10,
    max_steps_per_epoch: Optional[int] = None,
) -> int:
    """Train the model for a single epoch."""

    model.train()
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp and device.type == "cuda")
    for step, batch in enumerate(dataloader):
        if max_steps_per_epoch is not None and step >= max_steps_per_epoch:
            break
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device=device, enabled=use_amp):
            outputs = model(
                images=batch["images"],
                ego_hist=batch["ego_hist"],
                velocity=batch["velocity"],
                acceleration=batch["acceleration"],
                route_command_ids=batch["route_command_ids"],
                language_input=batch["language_input"],
            )
            loss_dict = loss_computer(outputs, batch)
            total_loss = loss_dict["total"]

        if scaler.is_enabled():
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        global_step += 1
        if logger is not None and (global_step % log_every_n_steps == 0 or global_step == 1):
            logger.log_metrics(
                split=f"train/epoch_{epoch}",
                step=global_step,
                metrics={key: float(value.detach().cpu()) for key, value in loss_dict.items()},
            )
    return global_step


def run_training(config: ExperimentConfig) -> Dict[str, float]:
    """Run end-to-end training and optional validation."""

    set_deterministic_seed(config.training.seed)
    train_loader, val_loader, dimensions, rationale_tokenizer = build_dataloaders(config.data, config.model)
    device = torch.device(config.training.device)
    model = PlannerModel.from_config(
        config=config.model,
        image_size=dimensions.image_size,
        temporal_window=dimensions.temporal_window,
        num_cameras=dimensions.num_cameras,
        history_steps=dimensions.history_steps,
        future_steps=dimensions.future_steps,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    loss_computer = PlannerLossComputer(
        config=config.losses,
        future_steps=dimensions.future_steps,
        pad_token_id=rationale_tokenizer.pad_token_id,
    ).to(device)

    output_dir = Path(config.logging.output_dir) / config.logging.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = _build_logger(output_dir, config.logging.use_tensorboard)

    global_step = 0
    if config.training.resume and config.training.checkpoint_path:
        payload = load_checkpoint(
            Path(config.training.checkpoint_path),
            model=model,
            optimizer=optimizer,
            map_location=config.training.device,
        )
        global_step = int(payload.get("global_step", 0))

    final_metrics: Dict[str, float] = {}
    for epoch in range(config.training.epochs):
        global_step = train_one_epoch(
            model=model,
            loss_computer=loss_computer,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            grad_clip_norm=config.training.grad_clip_norm,
            use_amp=config.training.use_amp,
            logger=logger,
            global_step=global_step,
            log_every_n_steps=config.logging.log_every_n_steps,
            max_steps_per_epoch=config.training.max_steps_per_epoch,
        )

        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
        save_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
        )

        if val_loader is not None and (epoch + 1) % config.evaluation.eval_every_n_epochs == 0:
            final_metrics = run_evaluation(
                model=model,
                dataloader=val_loader,
                device=device,
                output_dir=output_dir / "eval" / f"epoch_{epoch}",
                save_visualizations=config.evaluation.save_visualizations,
                num_visualizations=config.evaluation.num_visualizations,
            )
            logger.log_metrics(split=f"val/epoch_{epoch}", step=global_step, metrics=final_metrics)

    logger.close()
    return final_metrics
