from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import torch
from torch.utils.data import Dataset

from .config import TokenConfig
from .discretizer import UnifiedDriveDiscretizer


@dataclass
class UnifiedDrivingSample:
    """最小主链路的统一样本格式。"""

    front_image: torch.Tensor
    bev_now: torch.Tensor
    history_actions: torch.Tensor
    future_actions: torch.Tensor
    future_bevs: torch.Tensor
    navigation_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


# 兼容旧名称，避免主链路外的早期调用立刻断掉。
DriveSample = UnifiedDrivingSample


def rollout_future_bevs_from_actions(
    bev_now: torch.Tensor,
    future_actions: torch.Tensor,
    token_config: TokenConfig,
) -> torch.Tensor:
    """根据未来动作粗略滚动出未来 BEV。"""
    if future_actions.ndim != 2 or future_actions.shape[-1] != 2:
        raise ValueError(
            "future_actions 必须是形状为 [T_future, 2] 的张量，"
            f"当前收到 {tuple(future_actions.shape)}"
        )

    frames: List[torch.Tensor] = []
    for frame_index in range(token_config.future_bev_frames):
        action = future_actions[min(frame_index, future_actions.shape[0] - 1)]
        shift_y = int(torch.round(action[0]).item())
        shift_x = int(torch.round(action[1]).item())
        frame = torch.roll(bev_now, shifts=(shift_y, shift_x), dims=(1, 2))
        attenuation = 1.0 - 0.08 * frame_index
        frames.append((frame * attenuation).clamp(0.0, 1.0))
    return torch.stack(frames, dim=0)


class ToyUnifiedDriveDataset(Dataset):
    """使用合成数据打通 unified-token 训练主流程。"""

    def __init__(self, size: int, token_config: TokenConfig, seed: int = 42) -> None:
        """初始化 toy 数据集。"""
        if size <= 0:
            raise ValueError(f"toy 数据集大小必须大于 0，当前收到 size={size}")
        self.size = size
        self.token_config = token_config
        self.seed = seed

    def __len__(self) -> int:
        """返回 toy 数据集长度。"""
        return self.size

    def _build_navigation_text(self, index: int) -> str:
        """构造一条结构化导航文本。"""
        commands = ["左转", "右转", "直行", "通过路口后并线", "进入主路后保持车道"]
        road_types = ["城市道路", "匝道", "十字路口", "环岛入口", "高架出口"]
        speed_targets = ["30km/h", "40km/h", "50km/h", "60km/h"]
        command = commands[index % len(commands)]
        road_type = road_types[index % len(road_types)]
        speed = speed_targets[index % len(speed_targets)]
        return f"[导航] 当前道路={road_type}; 下一操作={command}; 目标速度={speed}; 注意避让前方车辆。"

    def _make_base_bev(self, generator: torch.Generator) -> torch.Tensor:
        """生成一张基础 BEV 栅格。"""
        height, width = self.token_config.bev_size
        bev = torch.rand((1, height, width), generator=generator)
        center_y = height // 2
        bev[:, center_y - 2 : center_y + 2, :] *= 0.3
        bev[:, :, width // 2 - 1 : width // 2 + 1] *= 0.6
        return bev.clamp(0.0, 1.0)

    def _make_front_image(self, generator: torch.Generator) -> torch.Tensor:
        """生成一张与 BEV 同步的 toy 前视图。"""
        height, width = self.token_config.image_size
        image = torch.rand((3, height, width), generator=generator)
        image[0] *= 0.8
        image[1] *= 0.9
        horizontal_gradient = torch.linspace(0.0, 1.0, width).view(1, 1, width)
        image = (image + horizontal_gradient) / 2.0
        return image.clamp(0.0, 1.0)

    def _make_history_actions(self, generator: torch.Generator) -> torch.Tensor:
        """生成历史动作序列。"""
        horizon = self.token_config.history_action_horizon
        return (
            torch.rand((horizon, 2), generator=generator) * 2.0 - 1.0
        ) * self.token_config.action_value_range

    def _make_future_actions(self, generator: torch.Generator) -> torch.Tensor:
        """生成未来动作序列。"""
        horizon = self.token_config.future_action_horizon
        base = (torch.rand((horizon, 2), generator=generator) * 2.0 - 1.0) * 0.6
        turn_bias = torch.linspace(-0.4, 0.4, horizon).unsqueeze(-1)
        base[:, :1] += turn_bias
        return base.clamp(
            -self.token_config.action_value_range,
            self.token_config.action_value_range,
        )

    def __getitem__(self, index: int) -> UnifiedDrivingSample:
        """返回一条合成 driving 样本。"""
        generator = torch.Generator().manual_seed(self.seed + index)
        bev_now = self._make_base_bev(generator)
        future_actions = self._make_future_actions(generator)
        return UnifiedDrivingSample(
            front_image=self._make_front_image(generator),
            bev_now=bev_now,
            history_actions=self._make_history_actions(generator),
            future_actions=future_actions,
            future_bevs=rollout_future_bevs_from_actions(
                bev_now=bev_now,
                future_actions=future_actions,
                token_config=self.token_config,
            ),
            navigation_text=self._build_navigation_text(index),
            metadata={
                "source": "toy",
                "dataset_type": "toy",
                "sample_index": index,
                "seed": self.seed,
            },
        )


def build_dataset(
    dataset_type: str,
    token_config: TokenConfig,
    seed: int = 42,
    dataset_size: int | None = None,
    nuscenes_root: str | None = None,
    nuscenes_version: str = "v1.0-mini",
    nuscenes_split: str = "mini_train",
    max_samples: int | None = None,
    focus_scene_token: str | None = None,
) -> Dataset:
    """按数据源类型创建最小主链路数据集。"""
    normalized_dataset_type = dataset_type.strip().lower()
    if normalized_dataset_type == "toy":
        toy_size = dataset_size if dataset_size is not None else 8
        return ToyUnifiedDriveDataset(
            size=toy_size,
            token_config=token_config,
            seed=seed,
        )

    if normalized_dataset_type == "nuscenes":
        if not nuscenes_root:
            raise ValueError("使用 nuScenes 数据集时必须提供 --nuscenes_root。")
        from .nuscenes_adapter import NuScenesUnifiedDriveDataset

        return NuScenesUnifiedDriveDataset(
            root=nuscenes_root,
            version=nuscenes_version,
            split=nuscenes_split,
            token_config=token_config,
            max_samples=max_samples,
            focus_scene_token=focus_scene_token,
            seed=seed,
        )

    raise ValueError(f"不支持的数据集类型: {dataset_type}，可选值为 toy 或 nuscenes。")


class UnifiedDriveCollator:
    """把样本列表整理成训练 batch。"""

    def __init__(self, discretizer: UnifiedDriveDiscretizer, pad_token_id: int) -> None:
        """初始化最小 collator。"""
        self.discretizer = discretizer
        self.pad_token_id = pad_token_id

    def _pad_sequences(self, values: Sequence[List[int]], pad_value: int) -> torch.Tensor:
        """把不同长度的一维序列 pad 到同一长度。"""
        max_length = max(len(value) for value in values)
        padded = [value + [pad_value] * (max_length - len(value)) for value in values]
        return torch.tensor(padded, dtype=torch.long)

    def __call__(self, samples: Sequence[UnifiedDrivingSample]) -> Dict[str, Any]:
        """把一批样本编码成模型输入。"""
        encodings = [self.discretizer.build_training_sequence(sample) for sample in samples]
        input_ids = self._pad_sequences([encoding.input_ids for encoding in encodings], self.pad_token_id)
        labels = self._pad_sequences([encoding.labels for encoding in encodings], -100)
        token_types = self._pad_sequences([encoding.token_types for encoding in encodings], 0)
        attention_mask = input_ids.ne(self.pad_token_id).long()
        return {
            "input_ids": input_ids,
            "labels": labels,
            "token_types": token_types,
            "attention_mask": attention_mask,
            "pixel_values_list": [encoding.pixel_values for encoding in encodings],
            "image_sizes_list": [encoding.image_sizes for encoding in encodings],
        }
