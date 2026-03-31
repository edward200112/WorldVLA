"""Configuration loading utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .schema import (
    DataConfig,
    DatasetSourceConfig,
    EvalConfig,
    ExperimentConfig,
    LoggingConfig,
    LossConfig,
    ModelConfig,
    TrainConfig,
)


def _read_config_dict(config_path: Path) -> Dict[str, Any]:
    """Load a configuration file as a dictionary."""

    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise TypeError(f"Expected mapping in config, got {type(loaded)!r}")
        return loaded
    except ModuleNotFoundError:
        loaded = json.loads(text)
        if not isinstance(loaded, dict):
            raise TypeError(f"Expected mapping in config, got {type(loaded)!r}")
        return loaded


def _to_tuple(values: Iterable[int]) -> Tuple[int, int]:
    sequence = tuple(int(v) for v in values)
    if len(sequence) != 2:
        raise ValueError(f"Expected length-2 tuple, got {sequence!r}")
    return sequence


def _build_dataset_source(raw: Dict[str, Any]) -> DatasetSourceConfig:
    cameras = raw.get("cameras", ["front", "front_left", "front_right"])
    image_size = _to_tuple(raw.get("image_size", [128, 128]))
    return DatasetSourceConfig(
        name=str(raw["name"]),
        annotation_file=raw.get("annotation_file"),
        data_root=str(raw.get("data_root", "")),
        split=str(raw.get("split", "train")),
        use_synthetic=bool(raw.get("use_synthetic", False)),
        synthetic_length=int(raw.get("synthetic_length", 32)),
        cameras=[str(camera) for camera in cameras],
        image_size=image_size,
        temporal_window=int(raw.get("temporal_window", 4)),
        future_steps=int(raw.get("future_steps", 20)),
        history_steps=int(raw.get("history_steps", 16)),
        auto_behavior_from_waypoints=bool(raw.get("auto_behavior_from_waypoints", True)),
        rationale_max_length=int(raw.get("rationale_max_length", 16)),
    )


def load_config(config_path: str | Path) -> ExperimentConfig:
    """Load an experiment configuration from disk."""

    path = Path(config_path)
    raw = _read_config_dict(path)

    data_raw = raw.get("data", {})
    train_sources = [_build_dataset_source(item) for item in data_raw.get("train_sources", [])]
    val_sources = [_build_dataset_source(item) for item in data_raw.get("val_sources", [])]
    data = DataConfig(
        train_sources=train_sources,
        val_sources=val_sources,
        num_workers=int(data_raw.get("num_workers", 0)),
        batch_size=int(data_raw.get("batch_size", 4)),
        shuffle=bool(data_raw.get("shuffle", True)),
    )

    model_raw = raw.get("model", {})
    model = ModelConfig(
        vision_encoder_type=str(model_raw.get("vision_encoder_type", "lite_vit")),
        vision_hidden_dim=int(model_raw.get("vision_hidden_dim", 128)),
        ego_encoder_type=str(model_raw.get("ego_encoder_type", "transformer")),
        text_backend=str(model_raw.get("text_backend", "hash")),
        text_model_name=model_raw.get("text_model_name"),
        text_vocab_size=int(model_raw.get("text_vocab_size", 2048)),
        text_max_length=int(model_raw.get("text_max_length", 24)),
        rationale_vocab_size=int(model_raw.get("rationale_vocab_size", 2048)),
        rationale_max_length=int(model_raw.get("rationale_max_length", 16)),
        fusion_dim=int(model_raw.get("fusion_dim", 128)),
        fusion_heads=int(model_raw.get("fusion_heads", 4)),
        fusion_layers=int(model_raw.get("fusion_layers", 2)),
        dropout=float(model_raw.get("dropout", 0.1)),
        freeze_text_encoder=bool(model_raw.get("freeze_text_encoder", False)),
        max_fusion_tokens=int(model_raw.get("max_fusion_tokens", 128)),
        lite_vit_patch_size=int(model_raw.get("lite_vit_patch_size", 16)),
        ego_transformer_heads=int(model_raw.get("ego_transformer_heads", 4)),
        ego_transformer_layers=int(model_raw.get("ego_transformer_layers", 2)),
    )

    loss_raw = raw.get("losses", {})
    losses = LossConfig(
        waypoint_weight=float(loss_raw.get("waypoint_weight", 1.0)),
        behavior_weight=float(loss_raw.get("behavior_weight", 1.0)),
        rationale_weight=float(loss_raw.get("rationale_weight", 0.5)),
        smoothness_weight=float(loss_raw.get("smoothness_weight", 0.1)),
        preference_weight=float(loss_raw.get("preference_weight", 0.0)),
        waypoint_final_step_weight=float(loss_raw.get("waypoint_final_step_weight", 3.0)),
    )

    logging_raw = raw.get("logging", {})
    logging = LoggingConfig(
        use_tensorboard=bool(logging_raw.get("use_tensorboard", False)),
        log_every_n_steps=int(logging_raw.get("log_every_n_steps", 10)),
        output_dir=str(logging_raw.get("output_dir", "outputs")),
        run_name=str(logging_raw.get("run_name", "debug_run")),
    )

    evaluation_raw = raw.get("evaluation", {})
    evaluation = EvalConfig(
        save_visualizations=bool(evaluation_raw.get("save_visualizations", True)),
        num_visualizations=int(evaluation_raw.get("num_visualizations", 8)),
        eval_every_n_epochs=int(evaluation_raw.get("eval_every_n_epochs", 1)),
    )

    training_raw = raw.get("training", {})
    training = TrainConfig(
        seed=int(training_raw.get("seed", 7)),
        epochs=int(training_raw.get("epochs", 2)),
        learning_rate=float(training_raw.get("learning_rate", 1e-3)),
        weight_decay=float(training_raw.get("weight_decay", 1e-4)),
        grad_clip_norm=float(training_raw.get("grad_clip_norm", 1.0)),
        use_amp=bool(training_raw.get("use_amp", True)),
        device=str(training_raw.get("device", "cpu")),
        checkpoint_path=training_raw.get("checkpoint_path"),
        resume=bool(training_raw.get("resume", False)),
        max_steps_per_epoch=training_raw.get("max_steps_per_epoch"),
    )

    return ExperimentConfig(
        experiment_name=str(raw.get("experiment_name", "language_waypoint_planner")),
        data=data,
        model=model,
        losses=losses,
        logging=logging,
        evaluation=evaluation,
        training=training,
    )
