from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

ACTION_QUANTIZATION_MODES: Tuple[str, str] = (
    "uniform_with_deadband",
    "nonuniform_zero_dense",
)


@dataclass
class TokenConfig:
    """统一离散 token 空间的配置。"""

    image_size: Tuple[int, int] = (64, 64)
    bev_size: Tuple[int, int] = (32, 32)
    image_patch_size: int = 8
    bev_patch_size: int = 8
    bev_codebook_size: int = 128
    action_bins_per_dim: int = 8
    summary_bins_per_dim: int = 8
    future_action_horizon: int = 8
    history_action_horizon: int = 4
    future_bev_frames: int = 3
    max_text_tokens: int = 32
    action_value_range: float = 1.5
    action_zero_deadband: float = 0.1
    action_quantization_mode: str = "nonuniform_zero_dense"
    action_longitudinal_quantization_mode: str | None = None
    action_lateral_quantization_mode: str | None = None
    action_longitudinal_zero_dense_power: float = 1.5
    action_lateral_zero_dense_power: float = 2.5
    action_lateral_near_zero_threshold: float = 0.1
    system_prompt: str = "你是一个 unified-token 自动驾驶规划模型。"
    action_token_prefix: str = "UT_ACT"
    bev_token_prefix: str = "UT_BEV"
    summary_token_prefix: str = "UT_SUM"

    @property
    def action_codebook_size(self) -> int:
        """返回动作离散词表大小。"""
        return self.action_bins_per_dim * self.action_bins_per_dim

    @property
    def summary_codebook_size(self) -> int:
        """返回历史动作摘要离散词表大小。"""
        return self.summary_bins_per_dim * self.summary_bins_per_dim

    @property
    def image_tokens_per_frame(self) -> int:
        """返回一张前视图图像会被切成多少个 token。"""
        return (self.image_size[0] // self.image_patch_size) * (
            self.image_size[1] // self.image_patch_size
        )

    @property
    def bev_tokens_per_frame(self) -> int:
        """返回一帧 BEV 会被切成多少个 token。"""
        return (self.bev_size[0] // self.bev_patch_size) * (
            self.bev_size[1] // self.bev_patch_size
        )


@dataclass
class ModelConfig:
    """Emu3 与 LoRA 的配置。"""

    model_name: str = "hf_models/Emu3-Chat-hf"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "eager"
    load_in_4bit: bool = False
    gradient_checkpointing: bool = True
    use_selective_attention_mask: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass
class TrainConfig:
    """训练循环的最小配置。"""

    seed: int = 42
    batch_size: int = 1
    num_workers: int = 0
    dataset_size: int = 8
    num_epochs: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    action_loss_weight: float = 6.0
    future_bev_loss_weight: float = 1.0
    supervise_action_only: bool = False
    max_grad_norm: float = 1.0
    log_every: int = 1
    output_dir: str = "outputs/unitok_drive_lite"


@dataclass
class ExperimentConfig:
    """整个最小原型的总配置。"""

    root_dir: str
    model: ModelConfig = field(default_factory=ModelConfig)
    tokens: TokenConfig = field(default_factory=TokenConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def build_default_config(root_dir: Path) -> ExperimentConfig:
    """构造项目默认配置。"""
    return ExperimentConfig(root_dir=str(root_dir))
