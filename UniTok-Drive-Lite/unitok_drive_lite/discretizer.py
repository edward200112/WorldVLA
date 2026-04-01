from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Sequence

import torch
import torch.nn.functional as F

from .config import TokenConfig

if TYPE_CHECKING:
    from .data import DriveSample


@dataclass
class UnifiedTokenLayout:
    """保存统一 token 空间里各类 token 的实际 id。"""

    marker_ids: Dict[str, int]
    image_token_ids: List[int]
    bev_token_ids: List[int]
    action_token_ids: List[int]
    summary_token_ids: List[int]


@dataclass
class SequenceEncoding:
    """保存一条样本被编码后的训练序列。"""

    input_ids: List[int]
    labels: List[int]
    role_ids: List[int]
    action_token_ids: List[int]
    future_bev_token_ids: List[List[int]]
    action_query_positions: List[int]
    future_bev_query_positions: List[List[int]]


class UnifiedDriveDiscretizer:
    """把图像、BEV、动作统一映射到 Chameleon 的 reserved token 空间。"""

    def __init__(self, tokenizer, config: TokenConfig) -> None:
        """初始化统一离散器并绑定 tokenizer。"""
        self.tokenizer = tokenizer
        self.config = config
        self._validate_config()
        self.layout = self._build_layout()
        self.action_token_to_index = {
            token_id: index for index, token_id in enumerate(self.layout.action_token_ids)
        }
        self.summary_token_to_index = {
            token_id: index for index, token_id in enumerate(self.layout.summary_token_ids)
        }
        self.bev_token_to_index = {
            token_id: index for index, token_id in enumerate(self.layout.bev_token_ids)
        }

    def _validate_config(self) -> None:
        """校验 patch 尺寸和离散配置是否合法。"""
        image_height, image_width = self.config.image_size
        bev_height, bev_width = self.config.bev_size
        if image_height % self.config.image_patch_size != 0 or image_width % self.config.image_patch_size != 0:
            raise ValueError("image_size 必须能被 image_patch_size 整除。")
        if bev_height % self.config.bev_patch_size != 0 or bev_width % self.config.bev_patch_size != 0:
            raise ValueError("bev_size 必须能被 bev_patch_size 整除。")

    def _build_layout(self) -> UnifiedTokenLayout:
        """为不同模态分配一段不冲突的 reserved token。"""
        marker_names = [
            "nav_start",
            "nav_end",
            "front_start",
            "front_end",
            "bev_now_start",
            "bev_now_end",
            "hist_action_start",
            "hist_action_end",
            "future_action_start",
            "future_action_end",
            "action_query",
            "bev_query",
        ]
        for frame_index in range(self.config.future_bev_frames):
            marker_names.append(f"future_bev_{frame_index + 1}_start")
            marker_names.append(f"future_bev_{frame_index + 1}_end")

        cursor = self.config.reserved_token_start
        marker_tokens = self._reserve_tokens(cursor, len(marker_names))
        cursor += len(marker_tokens)
        image_tokens = self._reserve_tokens(cursor, self.config.image_codebook_size)
        cursor += len(image_tokens)
        bev_tokens = self._reserve_tokens(cursor, self.config.bev_codebook_size)
        cursor += len(bev_tokens)
        action_tokens = self._reserve_tokens(cursor, self.config.action_codebook_size)
        cursor += len(action_tokens)
        summary_tokens = self._reserve_tokens(cursor, self.config.summary_codebook_size)

        return UnifiedTokenLayout(
            marker_ids={
                name: self.tokenizer.convert_tokens_to_ids(token)
                for name, token in zip(marker_names, marker_tokens)
            },
            image_token_ids=[self.tokenizer.convert_tokens_to_ids(token) for token in image_tokens],
            bev_token_ids=[self.tokenizer.convert_tokens_to_ids(token) for token in bev_tokens],
            action_token_ids=[self.tokenizer.convert_tokens_to_ids(token) for token in action_tokens],
            summary_token_ids=[self.tokenizer.convert_tokens_to_ids(token) for token in summary_tokens],
        )

    def _reserve_tokens(self, start_index: int, count: int) -> List[str]:
        """从 Chameleon 现有 reserved token 中切出一段作为统一词表。"""
        tokens = [f"<reserved{token_index:05d}>" for token_index in range(start_index, start_index + count)]
        unknown_id = getattr(self.tokenizer, "unk_token_id", None)
        for token in tokens:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id is None:
                raise ValueError(f"tokenizer 无法识别 reserved token: {token}")
            if unknown_id is not None and token_id == unknown_id:
                raise ValueError(f"reserved token 映射到了 unk_token，请调整 reserved_token_start: {token}")
        return tokens

    def _pool_image_to_scalars(self, image: torch.Tensor) -> torch.Tensor:
        """把前视图图像平均池化成一维 patch 标量序列。"""
        pooled = F.avg_pool2d(
            image.unsqueeze(0),
            kernel_size=self.config.image_patch_size,
            stride=self.config.image_patch_size,
        )
        return pooled.mean(dim=1).flatten().clamp(0.0, 1.0)

    def _pool_bev_to_scalars(self, bev: torch.Tensor) -> torch.Tensor:
        """把 BEV 平均池化成一维 patch 标量序列。"""
        pooled = F.avg_pool2d(
            bev.unsqueeze(0),
            kernel_size=self.config.bev_patch_size,
            stride=self.config.bev_patch_size,
        )
        return pooled.flatten().clamp(0.0, 1.0)

    def _quantize_scalars(self, values: torch.Tensor, token_ids: Sequence[int]) -> List[int]:
        """把 0 到 1 的连续标量量化到离散 token。"""
        indices = torch.round(values * (len(token_ids) - 1)).to(torch.long)
        return [token_ids[index] for index in indices.tolist()]

    def _quantize_vectors(
        self,
        vectors: torch.Tensor,
        bins_per_dim: int,
        token_ids: Sequence[int],
    ) -> List[int]:
        """把二维连续动作量化成单个离散 token。"""
        clipped = vectors.clamp(-self.config.action_value_range, self.config.action_value_range)
        normalized = (clipped + self.config.action_value_range) / (2.0 * self.config.action_value_range)
        normalized = normalized.clamp(0.0, 1.0)
        discrete = torch.round(normalized * (bins_per_dim - 1)).to(torch.long)
        flat_index = discrete[:, 0] * bins_per_dim + discrete[:, 1]
        return [token_ids[index] for index in flat_index.tolist()]

    def encode_front_image(self, image: torch.Tensor) -> List[int]:
        """把前视图图像编码成统一 token。"""
        return self._quantize_scalars(self._pool_image_to_scalars(image), self.layout.image_token_ids)

    def encode_bev(self, bev: torch.Tensor) -> List[int]:
        """把一帧 BEV 编码成统一 token。"""
        return self._quantize_scalars(self._pool_bev_to_scalars(bev), self.layout.bev_token_ids)

    def encode_future_bevs(self, future_bevs: torch.Tensor) -> List[List[int]]:
        """把未来多帧 BEV 编码成统一 token。"""
        return [self.encode_bev(frame) for frame in future_bevs]

    def encode_future_actions(self, future_actions: torch.Tensor) -> List[int]:
        """把未来连续动作编码成 raw action token。"""
        return self._quantize_vectors(
            future_actions,
            bins_per_dim=self.config.action_bins_per_dim,
            token_ids=self.layout.action_token_ids,
        )

    def encode_history_summary(self, history_actions: torch.Tensor) -> int:
        """把历史动作压缩成一个 summary token。"""
        summary_vector = history_actions.mean(dim=0, keepdim=True)
        return self._quantize_vectors(
            summary_vector,
            bins_per_dim=self.config.summary_bins_per_dim,
            token_ids=self.layout.summary_token_ids,
        )[0]

    def _tokenize_navigation_text(self, nav_text: str) -> List[int]:
        """把结构化导航文本编码成文本 token。"""
        return self.tokenizer.encode(
            nav_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.config.max_text_tokens,
        )

    def build_training_sequence(self, sample: DriveSample) -> SequenceEncoding:
        """把一条 driving 样本组织成训练序列与标签。"""
        marker_ids = self.layout.marker_ids
        nav_ids = (
            [marker_ids["nav_start"]]
            + self._tokenize_navigation_text(sample.navigation_text)
            + [marker_ids["nav_end"]]
        )
        front_ids = (
            [marker_ids["front_start"]]
            + self.encode_front_image(sample.front_image)
            + [marker_ids["front_end"]]
        )
        bev_now_ids = (
            [marker_ids["bev_now_start"]]
            + self.encode_bev(sample.bev_now)
            + [marker_ids["bev_now_end"]]
        )
        hist_ids = [
            marker_ids["hist_action_start"],
            self.encode_history_summary(sample.history_actions),
            marker_ids["hist_action_end"],
        ]
        action_ids = self.encode_future_actions(sample.future_actions)
        future_bev_ids = self.encode_future_bevs(sample.future_bevs)

        context_ids = nav_ids + front_ids + bev_now_ids + hist_ids
        query_ids: List[int] = [marker_ids["future_action_start"]]
        labels: List[int] = [-100] * len(context_ids) + [-100]
        role_ids: List[int] = [0] * len(context_ids) + [0]
        action_query_positions: List[int] = []
        future_bev_query_positions: List[List[int]] = []

        for action_token_id in action_ids:
            action_query_positions.append(len(context_ids) + len(query_ids))
            query_ids.append(marker_ids["action_query"])
            labels.append(action_token_id)
            role_ids.append(1)

        query_ids.append(marker_ids["future_action_end"])
        labels.append(-100)
        role_ids.append(0)

        for frame_index, frame_token_ids in enumerate(future_bev_ids):
            query_ids.append(marker_ids[f"future_bev_{frame_index + 1}_start"])
            labels.append(-100)
            role_ids.append(0)

            current_frame_positions: List[int] = []
            for frame_token_id in frame_token_ids:
                current_frame_positions.append(len(context_ids) + len(query_ids))
                query_ids.append(marker_ids["bev_query"])
                labels.append(frame_token_id)
                role_ids.append(0)

            future_bev_query_positions.append(current_frame_positions)
            query_ids.append(marker_ids[f"future_bev_{frame_index + 1}_end"])
            labels.append(-100)
            role_ids.append(0)

        input_ids = context_ids + query_ids
        return SequenceEncoding(
            input_ids=input_ids,
            labels=labels,
            role_ids=role_ids,
            action_token_ids=action_ids,
            future_bev_token_ids=future_bev_ids,
            action_query_positions=action_query_positions,
            future_bev_query_positions=future_bev_query_positions,
        )

    def build_generation_queries(self, sample: DriveSample) -> SequenceEncoding:
        """构造推理时的一次性 query 序列。"""
        return self.build_training_sequence(sample)

    def decode_action_token_ids(self, token_ids: Sequence[int]) -> torch.Tensor:
        """把 raw action token 还原为连续动作增量。"""
        vectors: List[List[float]] = []
        bins_per_dim = self.config.action_bins_per_dim
        for token_id in token_ids:
            flat_index = self.action_token_to_index[token_id]
            x_bin = flat_index // bins_per_dim
            y_bin = flat_index % bins_per_dim
            discrete = torch.tensor([x_bin, y_bin], dtype=torch.float32)
            normalized = discrete / max(bins_per_dim - 1, 1)
            continuous = normalized * (2.0 * self.config.action_value_range) - self.config.action_value_range
            vectors.append(continuous.tolist())
        return torch.tensor(vectors, dtype=torch.float32)

    def decode_action_tokens_to_trajectory(self, token_ids: Sequence[int]) -> torch.Tensor:
        """把动作 token 累积成连续轨迹。"""
        deltas = self.decode_action_token_ids(token_ids)
        return torch.cumsum(deltas, dim=0)

    def decode_bev_token_ids(self, token_ids: Sequence[int]) -> torch.Tensor:
        """把一帧 BEV token 还原成粗粒度连续栅格。"""
        patch_values = []
        for token_id in token_ids:
            level_index = self.bev_token_to_index[token_id]
            scalar = level_index / max(len(self.layout.bev_token_ids) - 1, 1)
            patch_values.append(scalar)
        patch_tensor = torch.tensor(patch_values, dtype=torch.float32).view(
            self.config.bev_size[0] // self.config.bev_patch_size,
            self.config.bev_size[1] // self.config.bev_patch_size,
        )
        expanded = patch_tensor.repeat_interleave(self.config.bev_patch_size, dim=0).repeat_interleave(
            self.config.bev_patch_size, dim=1
        )
        return expanded.unsqueeze(0)
