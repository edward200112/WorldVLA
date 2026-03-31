"""Dataclass-backed experiment configuration schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class DatasetSourceConfig:
    """Configuration for a single dataset source."""

    name: str
    annotation_file: Optional[str] = None
    data_root: str = ""
    split: str = "train"
    use_synthetic: bool = False
    synthetic_length: int = 32
    cameras: List[str] = field(default_factory=lambda: ["front", "front_left", "front_right"])
    image_size: Tuple[int, int] = (128, 128)
    temporal_window: int = 4
    future_steps: int = 20
    history_steps: int = 16
    auto_behavior_from_waypoints: bool = True
    rationale_max_length: int = 16


@dataclass
class DataConfig:
    """Data-related configuration."""

    train_sources: List[DatasetSourceConfig] = field(default_factory=list)
    val_sources: List[DatasetSourceConfig] = field(default_factory=list)
    num_workers: int = 0
    batch_size: int = 4
    shuffle: bool = True


@dataclass
class ModelConfig:
    """Model-related configuration."""

    vision_encoder_type: str = "lite_vit"
    vision_hidden_dim: int = 128
    ego_encoder_type: str = "transformer"
    text_backend: str = "hash"
    text_model_name: Optional[str] = None
    text_vocab_size: int = 2048
    text_max_length: int = 24
    rationale_vocab_size: int = 2048
    rationale_max_length: int = 16
    fusion_dim: int = 128
    fusion_heads: int = 4
    fusion_layers: int = 2
    dropout: float = 0.1
    freeze_text_encoder: bool = False
    max_fusion_tokens: int = 128
    lite_vit_patch_size: int = 16
    ego_transformer_heads: int = 4
    ego_transformer_layers: int = 2


@dataclass
class LossConfig:
    """Loss hyper-parameters."""

    waypoint_weight: float = 1.0
    behavior_weight: float = 1.0
    rationale_weight: float = 0.5
    smoothness_weight: float = 0.1
    preference_weight: float = 0.0
    waypoint_final_step_weight: float = 3.0


@dataclass
class LoggingConfig:
    """Logging-related configuration."""

    use_tensorboard: bool = False
    log_every_n_steps: int = 10
    output_dir: str = "outputs"
    run_name: str = "debug_run"


@dataclass
class EvalConfig:
    """Evaluation-related configuration."""

    save_visualizations: bool = True
    num_visualizations: int = 8
    eval_every_n_epochs: int = 1


@dataclass
class TrainConfig:
    """Training-related configuration."""

    seed: int = 7
    epochs: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    use_amp: bool = True
    device: str = "cpu"
    checkpoint_path: Optional[str] = None
    resume: bool = False
    max_steps_per_epoch: Optional[int] = None


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    experiment_name: str = "language_waypoint_planner"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    losses: LossConfig = field(default_factory=LossConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
