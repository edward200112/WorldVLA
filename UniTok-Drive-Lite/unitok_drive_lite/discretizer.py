from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import TokenConfig
from .masking import TokenType
from .token_registry import TokenRegistry

if TYPE_CHECKING:
    from .data import UnifiedDrivingSample


@dataclass
class UnifiedTokenLayout:
    """保存统一 token 空间里各类 token 的实际 id。"""

    marker_ids: Dict[str, int]
    action_token_ids: List[int]
    bev_token_ids: List[int]
    summary_token_ids: List[int]


@dataclass
class SequenceEncoding:
    """保存一条样本被编码后的训练序列。"""

    input_ids: List[int]
    labels: List[int]
    token_types: List[int]
    action_token_ids: List[int]
    future_bev_token_ids: List[List[int]]
    action_query_positions: List[int]
    future_bev_query_positions: List[List[int]]
    pixel_values: torch.Tensor
    image_sizes: torch.Tensor

def build_global_special_tokens(config: TokenConfig) -> List[str]:
    """兼容旧调用方式，返回 registry 里统一维护的全部 special tokens。"""
    return TokenRegistry.from_token_config(config).all_special_tokens


def _to_pil_image(image: Any) -> Image.Image:
    """把 torch / numpy / PIL 输入统一转成 RGB PIL.Image。"""
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if torch.is_tensor(image):
        array = image.detach().cpu().numpy()
    elif isinstance(image, np.ndarray):
        array = image
    else:
        raise TypeError(f"不支持的图像类型: {type(image)}")

    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    elif array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    elif array.ndim != 3:
        raise ValueError(f"图像 shape 不合法: {array.shape}")

    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] != 3:
        raise ValueError(f"图像最后一维必须为 1 或 3，当前收到 {array.shape}")

    if array.dtype != np.uint8:
        if array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


class UnifiedDriveDiscretizer:
    """把多模态驾驶观测组织成 Emu3 chat prefix + 固定全局 token 序列。"""

    def __init__(self, tokenizer: Any, processor: Any, config: TokenConfig) -> None:
        """初始化统一离散器、token 布局和 Emu3 processor。"""
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.token_registry = TokenRegistry.from_token_config(config)
        self._validate_config()
        self.layout = self._build_layout()
        self.image_placeholder_id = self._resolve_image_placeholder_id()

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

    def _resolve_token_id(self, token: str) -> int:
        """把 special token 字符串解析成稳定的 token id。"""
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        unk_token_id = getattr(self.tokenizer, "unk_token_id", None)
        if token_id is None or (unk_token_id is not None and token_id == unk_token_id):
            raise ValueError(f"tokenizer 无法识别 special token: {token}")
        return int(token_id)

    def _resolve_image_placeholder_id(self) -> int:
        """解析 Emu3 文本中的图像占位 token id。"""
        candidate_tokens: List[str] = []
        for token_name in ("image_token", "boi_token"):
            value = getattr(self.tokenizer, token_name, None)
            if isinstance(value, str):
                candidate_tokens.append(value)
        for token_name in ("image_token",):
            value = getattr(self.processor, token_name, None)
            if isinstance(value, str):
                candidate_tokens.append(value)
        candidate_tokens.extend(["<image>", "<|image|>", "<|extra_0|>"])

        seen: set[str] = set()
        for token in candidate_tokens:
            if token in seen:
                continue
            seen.add(token)
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            unk_token_id = getattr(self.tokenizer, "unk_token_id", None)
            if token_id is not None and (unk_token_id is None or token_id != unk_token_id):
                return int(token_id)
        raise ValueError("无法解析 Emu3 的图像占位 token id。")

    def _build_layout(self) -> UnifiedTokenLayout:
        """解析固定全局 token 的实际 id。"""
        resolved_registry = self.token_registry.resolve_tokenizer(self.tokenizer)
        marker_ids = {
            "action_query": self._resolve_token_id("<ACT>"),
            "bev_query": self._resolve_token_id("<BEV>"),
            "summary_marker": self._resolve_token_id("<ACT_SUMMARY>"),
        }
        return UnifiedTokenLayout(
            marker_ids=marker_ids,
            action_token_ids=resolved_registry.action_token_ids,
            bev_token_ids=resolved_registry.bev_token_ids,
            summary_token_ids=resolved_registry.summary_token_ids,
        )

    def _truncate_navigation_text(self, nav_text: str) -> str:
        """按配置长度截断导航文本，避免 prompt 过长。"""
        nav_ids = self.tokenizer.encode(nav_text, add_special_tokens=False, truncation=True, max_length=self.config.max_text_tokens)
        return self.tokenizer.decode(nav_ids, skip_special_tokens=False)

    def _build_chat_prefix(self, nav_text: str) -> str:
        """用 Emu3 chat template 构造系统前缀和用户观测部分。"""
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.config.system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "请结合当前观测规划未来 1 秒动作，并预测未来 3 帧 BEV。\n"
                            f"导航信息：{self._truncate_navigation_text(nav_text)}\n"
                            "前视图："
                        ),
                    },
                    {"type": "image"},
                    {"type": "text", "text": "\n当前 BEV："},
                    {"type": "image"},
                ],
            },
        ]
        return self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _pool_bev_to_scalars(self, bev: torch.Tensor) -> torch.Tensor:
        """把 BEV 平均池化成一维 patch 标量序列。"""
        pooled = F.avg_pool2d(
            bev.unsqueeze(0),
            kernel_size=self.config.bev_patch_size,
            stride=self.config.bev_patch_size,
        )
        return pooled.flatten().clamp(0.0, 1.0)

    def _quantize_scalars(self, values: torch.Tensor, token_ids: Sequence[int]) -> List[int]:
        """把 0 到 1 的连续标量量化到固定离散 token。"""
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

    def encode_bev(self, bev: torch.Tensor) -> List[int]:
        """把一帧 BEV 编码成固定全局 BEV token。"""
        return self._quantize_scalars(self._pool_bev_to_scalars(bev), self.layout.bev_token_ids)

    def encode_future_bevs(self, future_bevs: torch.Tensor) -> List[List[int]]:
        """把未来多帧 BEV 编码成离散 token。"""
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

    def _build_prefix_ids_and_types(self, nav_text: str) -> tuple[List[int], List[int]]:
        """把 Emu3 chat prefix 编码成 token id 和 token type。"""
        prefix_text = self._build_chat_prefix(nav_text)
        prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        token_types = [int(TokenType.TEXT)] * len(prefix_ids)

        image_positions = [index for index, token_id in enumerate(prefix_ids) if token_id == self.image_placeholder_id]
        if len(image_positions) < 2:
            raise ValueError("Emu3 chat prefix 中必须包含两个图像占位 token。")
        token_types[image_positions[0]] = int(TokenType.IMAGE)
        token_types[image_positions[1]] = int(TokenType.CURRENT_BEV)
        return prefix_ids, token_types

    def _append_text(
        self,
        input_ids: List[int],
        labels: List[int],
        token_types: List[int],
        text: str,
    ) -> None:
        """向统一序列尾部追加普通文本 token。"""
        text_ids = self.tokenizer.encode(text, add_special_tokens=False)
        input_ids.extend(text_ids)
        labels.extend([-100] * len(text_ids))
        token_types.extend([int(TokenType.TEXT)] * len(text_ids))

    def _build_vision_inputs(self, sample: UnifiedDrivingSample) -> tuple[torch.Tensor, torch.Tensor]:
        """把前视图和当前 BEV 渲染成 Emu3 image_processor 所需张量。"""
        images = [_to_pil_image(sample.front_image), _to_pil_image(sample.bev_now)]
        image_inputs = self.processor.image_processor(images=images, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"]
        image_sizes = image_inputs.get("image_sizes")
        if image_sizes is None:
            image_sizes = torch.tensor(
                [[image.height, image.width] for image in images],
                dtype=torch.long,
            )
        return pixel_values, image_sizes

    def build_training_sequence(self, sample: UnifiedDrivingSample) -> SequenceEncoding:
        """把一条 driving 样本组织成 Emu3 训练序列与视觉输入。"""
        prefix_ids, prefix_token_types = self._build_prefix_ids_and_types(sample.navigation_text)
        input_ids = list(prefix_ids)
        labels = [-100] * len(prefix_ids)
        token_types = list(prefix_token_types)
        action_query_positions: List[int] = []
        future_bev_query_positions: List[List[int]] = []

        summary_token_id = self.encode_history_summary(sample.history_actions)
        action_ids = self.encode_future_actions(sample.future_actions)
        future_bev_ids = self.encode_future_bevs(sample.future_bevs)

        self._append_text(input_ids, labels, token_types, "\n历史动作摘要：")
        input_ids.append(self.layout.marker_ids["summary_marker"])
        labels.append(-100)
        token_types.append(int(TokenType.TEXT))
        input_ids.append(summary_token_id)
        labels.append(-100)
        token_types.append(int(TokenType.HISTORY_ACTION_SUMMARY))

        self._append_text(input_ids, labels, token_types, "\n未来动作 token：")
        for action_token_id in action_ids:
            action_query_positions.append(len(input_ids))
            input_ids.append(self.layout.marker_ids["action_query"])
            labels.append(action_token_id)
            token_types.append(int(TokenType.FUTURE_ACTION))

        self._append_text(input_ids, labels, token_types, "\n未来 BEV token：")
        for frame_index, frame_token_ids in enumerate(future_bev_ids):
            self._append_text(input_ids, labels, token_types, f"\n第 {frame_index + 1} 帧：")
            current_frame_positions: List[int] = []
            for frame_token_id in frame_token_ids:
                current_frame_positions.append(len(input_ids))
                input_ids.append(self.layout.marker_ids["bev_query"])
                labels.append(frame_token_id)
                token_types.append(int(TokenType.FUTURE_BEV))
            future_bev_query_positions.append(current_frame_positions)

        pixel_values, image_sizes = self._build_vision_inputs(sample)
        return SequenceEncoding(
            input_ids=input_ids,
            labels=labels,
            token_types=token_types,
            action_token_ids=action_ids,
            future_bev_token_ids=future_bev_ids,
            action_query_positions=action_query_positions,
            future_bev_query_positions=future_bev_query_positions,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
        )

    def build_generation_queries(self, sample: UnifiedDrivingSample) -> SequenceEncoding:
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
            self.config.bev_patch_size,
            dim=1,
        )
        return expanded.unsqueeze(0)
