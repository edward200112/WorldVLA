from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .config import TokenConfig
from .data import UnifiedDrivingSample
from .eval_utils import bev_occupancy_stats


@dataclass(frozen=True)
class EgoPose2D:
    """最小 2D ego pose 表示。"""

    translation_xy: torch.Tensor
    yaw: float


@dataclass(frozen=True)
class NuScenesSampleDescriptor:
    """保存一个可用 nuScenes 样本的最小索引信息。"""

    scene_token: str
    scene_name: str
    sample_token: str


class NuScenesUnifiedDriveDataset(Dataset):
    """最小 nuScenes 适配器。

    目标是把 nuScenes 的单个 timestamp 包装成当前 unified-token 原型可消费的
    `UnifiedDrivingSample`，而不是实现完整 benchmark 级数据管线。
    """

    def __init__(
        self,
        root: str | Path,
        version: str = "v1.0-mini",
        split: str = "mini_train",
        token_config: TokenConfig | None = None,
        max_samples: int | None = None,
        focus_scene_token: str | None = None,
        seed: int = 42,
    ) -> None:
        if token_config is None:
            raise ValueError("NuScenesUnifiedDriveDataset 需要显式传入 token_config。")

        self.root = Path(root).expanduser().resolve()
        self.version = version
        self.split = split
        self.token_config = token_config
        self.max_samples = max_samples
        self.focus_scene_token = focus_scene_token
        self.seed = seed

        self.forward_extent_m = 25.0
        self.backward_extent_m = 10.0
        self.side_extent_m = 12.5
        self.motion_normalization_m = 4.0

        self._validate_root()
        self.nusc, split_scenes = self._init_nuscenes()
        self.scene_by_token = {scene["token"]: scene for scene in self.nusc.scene}
        self.sample_descriptors = self._collect_sample_descriptors(split_scenes)
        self.pose_cache: dict[str, EgoPose2D] = {}

        if not self.sample_descriptors:
            focus_scene_message = ""
            if self.focus_scene_token:
                focus_scene_message = f" focus_scene_token={self.focus_scene_token}"
            raise RuntimeError(
                "未找到可用的 nuScenes 样本。请检查 "
                f"--nuscenes_root={self.root} --nuscenes_version={self.version} "
                f"--nuscenes_split={self.split}{focus_scene_message} 是否匹配，"
                "并确认数据集中存在 CAM_FRONT 图像。"
            )

    def _validate_root(self) -> None:
        """在初始化 devkit 前先给出清晰的路径错误。"""
        if not self.root.exists():
            raise FileNotFoundError(
                f"未找到 nuScenes 数据根目录: {self.root}。请确认 --nuscenes_root 指向数据集根目录。"
            )

        table_dir = self.root / self.version
        if not table_dir.exists():
            raise FileNotFoundError(
                f"未找到 nuScenes 标注表目录: {table_dir}。"
                "目录结构通常应包含 samples/、sweeps/ 以及 v1.0-mini/ 或 v1.0-trainval/。"
            )

        required_tables = [
            "scene.json",
            "sample.json",
            "sample_data.json",
            "sample_annotation.json",
            "ego_pose.json",
            "log.json",
        ]
        missing_tables = [name for name in required_tables if not (table_dir / name).exists()]
        if missing_tables:
            raise FileNotFoundError(
                "nuScenes 标注表不完整，缺少以下文件: "
                f"{missing_tables}，请检查 {table_dir}"
            )

    def _init_nuscenes(self) -> tuple[Any, List[str]]:
        """懒加载 devkit，并解析 split 对应的 scene 名称。"""
        try:
            from nuscenes.nuscenes import NuScenes
            from nuscenes.utils.splits import create_splits_scenes
        except ImportError as error:
            raise ImportError(
                "未安装 nuScenes devkit。请先执行 `pip install nuscenes-devkit`，"
                "然后再使用 --dataset_type nuscenes。"
            ) from error

        available_splits = create_splits_scenes()
        if self.split not in available_splits:
            raise ValueError(
                f"不支持的 nuScenes split: {self.split}。"
                f"可选值包括: {sorted(available_splits.keys())}"
            )

        nusc = NuScenes(version=self.version, dataroot=str(self.root), verbose=False)
        return nusc, list(available_splits[self.split])

    def _collect_sample_descriptors(self, split_scenes: List[str]) -> List[NuScenesSampleDescriptor]:
        """按 split 枚举可用 sample token。"""
        allowed_scene_names = set(split_scenes)
        descriptors: List[NuScenesSampleDescriptor] = []
        for scene_record in self.nusc.scene:
            if scene_record["name"] not in allowed_scene_names:
                continue
            if self.focus_scene_token and scene_record["token"] != self.focus_scene_token:
                continue

            sample_token = scene_record["first_sample_token"]
            while sample_token:
                sample_record = self.nusc.get("sample", sample_token)
                cam_front_token = sample_record["data"].get("CAM_FRONT")
                if cam_front_token:
                    sample_data = self.nusc.get("sample_data", cam_front_token)
                    image_path = self.root / sample_data["filename"]
                    if image_path.exists():
                        descriptors.append(
                            NuScenesSampleDescriptor(
                                scene_token=scene_record["token"],
                                scene_name=scene_record["name"],
                                sample_token=sample_token,
                            )
                        )
                        if self.max_samples is not None and len(descriptors) >= self.max_samples:
                            return descriptors
                sample_token = sample_record["next"]

        return descriptors

    def __len__(self) -> int:
        """返回可用样本数。"""
        return len(self.sample_descriptors)

    def _get_sample_record(self, sample_token: str) -> Dict[str, Any]:
        return self.nusc.get("sample", sample_token)

    def _get_scene_record(self, descriptor: NuScenesSampleDescriptor) -> Dict[str, Any]:
        return self.scene_by_token[descriptor.scene_token]

    def _get_cam_front_sample_data(self, sample_record: Dict[str, Any]) -> Dict[str, Any]:
        cam_front_token = sample_record["data"].get("CAM_FRONT")
        if not cam_front_token:
            raise RuntimeError(
                f"sample_token={sample_record['token']} 不包含 CAM_FRONT，"
                "当前最小适配器要求使用 CAM_FRONT 作为前视图。"
            )
        return self.nusc.get("sample_data", cam_front_token)

    def _quaternion_to_yaw(self, rotation: List[float]) -> float:
        """把 nuScenes 的 [w, x, y, z] 四元数转成 yaw。"""
        w, x, y, z = rotation
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _get_pose_for_sample(self, sample_token: str) -> EgoPose2D:
        """读取与 sample 对应的当前 ego pose。"""
        cached = self.pose_cache.get(sample_token)
        if cached is not None:
            return cached

        sample_record = self._get_sample_record(sample_token)
        sample_data = self._get_cam_front_sample_data(sample_record)
        ego_pose = self.nusc.get("ego_pose", sample_data["ego_pose_token"])
        pose = EgoPose2D(
            translation_xy=torch.tensor(ego_pose["translation"][:2], dtype=torch.float32),
            yaw=self._quaternion_to_yaw(ego_pose["rotation"]),
        )
        self.pose_cache[sample_token] = pose
        return pose

    def _transform_world_delta_to_ego_xy(self, delta_xy: torch.Tensor, yaw: float) -> torch.Tensor:
        """把世界坐标系位移投影到目标 ego 坐标系。"""
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        forward = cos_yaw * delta_xy[0] + sin_yaw * delta_xy[1]
        lateral = -sin_yaw * delta_xy[0] + cos_yaw * delta_xy[1]
        return torch.tensor([forward, lateral], dtype=torch.float32)

    def _transform_global_xy_to_anchor_ego(self, global_xy: torch.Tensor, anchor_pose: EgoPose2D) -> torch.Tensor:
        """把全局坐标点转换到 anchor sample 的 ego 坐标系。

        所有 nuScenes future BEV target 都统一在 anchor sample 的坐标系下栅格化。
        这样当前帧与未来帧可以直接对齐，避免 future supervision 退化成简单的 `torch.roll(...)`。
        """
        return self._transform_world_delta_to_ego_xy(
            global_xy - anchor_pose.translation_xy,
            anchor_pose.yaw,
        )

    def _scale_motion_to_action(self, ego_delta_xy: torch.Tensor) -> torch.Tensor:
        """把米制位移缩放到当前 unified-token 原型的动作范围。"""
        scaled = (
            ego_delta_xy / self.motion_normalization_m * self.token_config.action_value_range
        )
        return scaled.clamp(
            -self.token_config.action_value_range,
            self.token_config.action_value_range,
        )

    def _build_history_actions(self, sample_token: str) -> torch.Tensor:
        """提取历史动作序列，不足部分补零。"""
        zero_action = torch.zeros(2, dtype=torch.float32)
        reversed_actions: List[torch.Tensor] = []
        current_token = sample_token

        for _ in range(self.token_config.history_action_horizon):
            current_sample = self._get_sample_record(current_token)
            prev_token = current_sample["prev"]
            if not prev_token:
                reversed_actions.append(zero_action.clone())
                continue

            prev_pose = self._get_pose_for_sample(prev_token)
            current_pose = self._get_pose_for_sample(current_token)
            world_delta = current_pose.translation_xy - prev_pose.translation_xy
            ego_delta = self._transform_world_delta_to_ego_xy(world_delta, prev_pose.yaw)
            reversed_actions.append(self._scale_motion_to_action(ego_delta))
            current_token = prev_token

        return torch.stack(list(reversed(reversed_actions)), dim=0)

    def _build_future_actions(self, sample_token: str) -> torch.Tensor:
        """提取未来动作序列，不足部分补零。"""
        zero_action = torch.zeros(2, dtype=torch.float32)
        actions: List[torch.Tensor] = []
        current_token = sample_token

        for _ in range(self.token_config.future_action_horizon):
            current_sample = self._get_sample_record(current_token)
            next_token = current_sample["next"]
            if not next_token:
                actions.append(zero_action.clone())
                continue

            current_pose = self._get_pose_for_sample(current_token)
            next_pose = self._get_pose_for_sample(next_token)
            world_delta = next_pose.translation_xy - current_pose.translation_xy
            ego_delta = self._transform_world_delta_to_ego_xy(world_delta, current_pose.yaw)
            actions.append(self._scale_motion_to_action(ego_delta))
            current_token = next_token

        return torch.stack(actions, dim=0)

    def _load_front_image(self, sample_record: Dict[str, Any]) -> torch.Tensor:
        """加载并缩放 CAM_FRONT 图像。"""
        sample_data = self._get_cam_front_sample_data(sample_record)
        image_path = self.root / sample_data["filename"]
        if not image_path.exists():
            raise FileNotFoundError(
                f"未找到 CAM_FRONT 图像文件: {image_path}，sample_token={sample_record['token']}"
            )

        target_height, target_width = self.token_config.image_size
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = image.resize((target_width, target_height), resample=Image.BICUBIC)
            image_array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(image_array).permute(2, 0, 1).contiguous()

    def _meters_to_grid(self, forward: float, lateral: float) -> tuple[int, int] | None:
        """把 ego 平面坐标映射到最小 BEV 栅格。"""
        if not (-self.backward_extent_m <= forward <= self.forward_extent_m):
            return None
        if not (-self.side_extent_m <= lateral <= self.side_extent_m):
            return None

        height, width = self.token_config.bev_size
        forward_extent = self.forward_extent_m + self.backward_extent_m
        row = int(round((self.forward_extent_m - forward) / forward_extent * (height - 1)))
        col = int(round((lateral + self.side_extent_m) / (2.0 * self.side_extent_m) * (width - 1)))
        row = max(0, min(height - 1, row))
        col = max(0, min(width - 1, col))
        return row, col

    def _paint_rect(
        self,
        bev: torch.Tensor,
        row: int,
        col: int,
        half_row: int,
        half_col: int,
        value: float,
    ) -> None:
        """在最小 BEV 上画一个轴对齐矩形。"""
        height, width = bev.shape[-2:]
        row_start = max(0, row - half_row)
        row_end = min(height, row + half_row + 1)
        col_start = max(0, col - half_col)
        col_end = min(width, col + half_col + 1)
        bev[:, row_start:row_end, col_start:col_end] = torch.maximum(
            bev[:, row_start:row_end, col_start:col_end],
            torch.full((1, row_end - row_start, col_end - col_start), value, dtype=bev.dtype),
        )

    def _build_bev_for_sample(
        self,
        target_sample_record: Dict[str, Any],
        anchor_pose: EgoPose2D,
        target_pose: EgoPose2D,
        *,
        anchor_sample_token: str,
        target_sample_token: str,
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        """把任意 sample 的 annotation 栅格化到 anchor sample 的 ego 坐标系下。"""
        height, width = self.token_config.bev_size
        bev = torch.zeros((1, height, width), dtype=torch.float32)

        target_ego_xy = self._transform_global_xy_to_anchor_ego(
            target_pose.translation_xy,
            anchor_pose,
        )
        ego_pixel = self._meters_to_grid(
            forward=float(target_ego_xy[0].item()),
            lateral=float(target_ego_xy[1].item()),
        )
        if ego_pixel is not None:
            self._paint_rect(bev, ego_pixel[0], ego_pixel[1], half_row=1, half_col=1, value=1.0)

        meters_per_row = (self.forward_extent_m + self.backward_extent_m) / max(height, 1)
        meters_per_col = (2.0 * self.side_extent_m) / max(width, 1)
        annotation_count_total = 0
        annotation_count_in_bounds = 0
        for annotation_token in target_sample_record["anns"]:
            annotation_count_total += 1
            annotation = self.nusc.get("sample_annotation", annotation_token)
            world_xy = torch.tensor(annotation["translation"][:2], dtype=torch.float32)
            local_xy = self._transform_global_xy_to_anchor_ego(world_xy, anchor_pose)
            pixel = self._meters_to_grid(
                forward=float(local_xy[0].item()),
                lateral=float(local_xy[1].item()),
            )
            if pixel is None:
                continue
            annotation_count_in_bounds += 1

            box_width_m, box_length_m = annotation["size"][:2]
            half_row = max(1, int(round((box_length_m / 2.0) / max(meters_per_row, 1e-6))))
            half_col = max(1, int(round((box_width_m / 2.0) / max(meters_per_col, 1e-6))))
            self._paint_rect(bev, pixel[0], pixel[1], half_row=half_row, half_col=half_col, value=0.7)

        bev = bev.clamp(0.0, 1.0)
        return bev, {
            "anchor_sample_token": anchor_sample_token,
            "target_sample_token": target_sample_token,
            "annotation_count_total": annotation_count_total,
            "annotation_count_in_bounds": annotation_count_in_bounds,
            "target_ego_anchor_xy": target_ego_xy.tolist(),
            "occupancy_stats": bev_occupancy_stats(bev),
        }

    def _build_bev_now(
        self,
        sample_record: Dict[str, Any],
        ego_pose: EgoPose2D,
        sample_token: str,
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        """构造当前 anchor sample 的 BEV。"""
        return self._build_bev_for_sample(
            target_sample_record=sample_record,
            anchor_pose=ego_pose,
            target_pose=ego_pose,
            anchor_sample_token=sample_token,
            target_sample_token=sample_token,
        )

    def _collect_future_sample_tokens(
        self,
        anchor_sample_token: str,
        horizon: int,
    ) -> tuple[List[str], List[int], List[bool]]:
        """收集 future horizon 对应的 sample token，不足时重复最后一个可用 future sample。"""
        future_tokens: List[str] = []
        future_timestamps: List[int] = []
        padding_flags: List[bool] = []
        current_token = anchor_sample_token
        last_valid_future_token: str | None = None

        for _ in range(horizon):
            current_sample = self._get_sample_record(current_token)
            next_token = current_sample["next"]
            if next_token:
                current_token = next_token
                last_valid_future_token = next_token
                padding_flags.append(False)
            else:
                if last_valid_future_token is None:
                    last_valid_future_token = anchor_sample_token
                padding_flags.append(True)

            future_tokens.append(last_valid_future_token)
            future_timestamps.append(int(self._get_sample_record(last_valid_future_token)["timestamp"]))

        return future_tokens, future_timestamps, padding_flags

    def _build_future_bevs_from_future_samples(
        self,
        *,
        anchor_sample_token: str,
        anchor_pose: EgoPose2D,
        bev_now: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        """从真实 future sample annotation 构造未来 BEV supervision。"""
        future_sample_tokens, future_timestamps, padding_flags = self._collect_future_sample_tokens(
            anchor_sample_token,
            self.token_config.future_bev_frames,
        )

        frames: List[torch.Tensor] = []
        frame_stats: List[Dict[str, Any]] = []
        for future_sample_token in future_sample_tokens:
            future_sample_record = self._get_sample_record(future_sample_token)
            future_pose = self._get_pose_for_sample(future_sample_token)
            frame, stats = self._build_bev_for_sample(
                target_sample_record=future_sample_record,
                anchor_pose=anchor_pose,
                target_pose=future_pose,
                anchor_sample_token=anchor_sample_token,
                target_sample_token=future_sample_token,
            )
            frames.append(frame)
            frame_stats.append(stats)

        future_bevs = torch.stack(frames, dim=0)
        future_info = {
            "future_bev_source": "nuscenes_annotations",
            "anchor_sample_token": anchor_sample_token,
            "future_sample_tokens": future_sample_tokens,
            "future_timestamps": future_timestamps,
            "coordinate_frame": "anchor_ego",
            "future_bev_padding_flags": padding_flags,
            "future_bev_padding_mode": "repeat_last_valid_frame",
            "future_bev_frame_stats": frame_stats,
        }
        self._validate_future_bevs(bev_now=bev_now, future_bevs=future_bevs, future_info=future_info)
        return future_bevs, future_info

    def _validate_future_bevs(
        self,
        *,
        bev_now: torch.Tensor,
        future_bevs: torch.Tensor,
        future_info: Dict[str, Any],
    ) -> None:
        """对 future BEV supervision 做轻量调试校验。"""
        expected_shape = (
            self.token_config.future_bev_frames,
            1,
            self.token_config.bev_size[0],
            self.token_config.bev_size[1],
        )
        if tuple(future_bevs.shape) != expected_shape:
            raise ValueError(
                f"future_bevs shape 不正确: got={tuple(future_bevs.shape)} expected={expected_shape}"
            )
        if not torch.isfinite(future_bevs).all():
            raise ValueError("future_bevs 中包含非有限值。")

        frame_stats: List[Dict[str, Any]] = future_info["future_bev_frame_stats"]
        for frame_index, stats in enumerate(frame_stats):
            occupancy_sum = float(stats["occupancy_stats"]["sum"])
            if stats["annotation_count_in_bounds"] > 0 and occupancy_sum <= 0.0:
                raise ValueError(
                    "future BEV occupancy 为空，但当前帧存在可栅格化 annotation: "
                    f"frame_index={frame_index} target_sample_token={stats['target_sample_token']}"
                )

        padding_flags: List[bool] = future_info["future_bev_padding_flags"]
        has_real_future_frame = any(not flag for flag in padding_flags)
        if has_real_future_frame:
            all_identical_to_current = all(torch.allclose(frame, bev_now) for frame in future_bevs)
            if all_identical_to_current:
                raise ValueError(
                    "检测到所有 future BEV 都与当前 bev_now 完全一致，"
                    "这通常意味着 future supervision 仍在退化成占位图。"
                )

    def _build_navigation_text(
        self,
        scene_record: Dict[str, Any],
        sample_record: Dict[str, Any],
    ) -> str:
        """构造一个最小占位导航文本。"""
        log_record = self.nusc.get("log", scene_record["log_token"])
        location = log_record.get("location", "unknown")
        description = (scene_record.get("description") or "").replace("\n", " ").strip()
        if description:
            description = description[:80]

        segments = [
            f"[导航] 数据源=nuScenes",
            f"split={self.split}",
            f"scene={scene_record['name']}",
            f"location={location}",
            f"timestamp={sample_record['timestamp']}",
        ]
        if description:
            segments.append(f"场景描述={description}")
        segments.append("请结合当前前视图和简化 BEV，保持平稳行驶并关注前方交通。")
        return "; ".join(segments) + "。"

    def __getitem__(self, index: int) -> UnifiedDrivingSample:
        """返回一条与最小主链路兼容的 nuScenes 样本。"""
        if index < 0 or index >= len(self.sample_descriptors):
            raise IndexError(f"sample_index 超出范围: {index}，当前数据集大小为 {len(self)}")

        descriptor = self.sample_descriptors[index]
        scene_record = self._get_scene_record(descriptor)
        sample_record = self._get_sample_record(descriptor.sample_token)
        ego_pose = self._get_pose_for_sample(descriptor.sample_token)

        front_image = self._load_front_image(sample_record)
        bev_now, current_bev_stats = self._build_bev_now(
            sample_record=sample_record,
            ego_pose=ego_pose,
            sample_token=descriptor.sample_token,
        )
        history_actions = self._build_history_actions(descriptor.sample_token)
        future_actions = self._build_future_actions(descriptor.sample_token)
        future_bevs, future_bev_info = self._build_future_bevs_from_future_samples(
            anchor_sample_token=descriptor.sample_token,
            anchor_pose=ego_pose,
            bev_now=bev_now,
        )
        navigation_text = self._build_navigation_text(scene_record, sample_record)

        return UnifiedDrivingSample(
            front_image=front_image,
            bev_now=bev_now,
            history_actions=history_actions,
            future_actions=future_actions,
            future_bevs=future_bevs,
            navigation_text=navigation_text,
            metadata={
                "source": "nuscenes",
                "dataset_type": "nuscenes",
                "adapter": "NuScenesUnifiedDriveDataset",
                "version": self.version,
                "split": self.split,
                "coordinate_frame": "anchor_ego",
                "scene_token": descriptor.scene_token,
                "scene_name": descriptor.scene_name,
                "sample_token": descriptor.sample_token,
                "timestamp": sample_record["timestamp"],
                "anchor_sample_token": descriptor.sample_token,
                "anchor_timestamp": sample_record["timestamp"],
                "motion_normalization_m": self.motion_normalization_m,
                "focus_scene_token": self.focus_scene_token,
                "current_bev_source": "nuscenes_annotations",
                "current_bev_stats": current_bev_stats,
                "future_bev_source": future_bev_info["future_bev_source"],
                "future_sample_tokens": future_bev_info["future_sample_tokens"],
                "future_timestamps": future_bev_info["future_timestamps"],
                "future_bev_padding_flags": future_bev_info["future_bev_padding_flags"],
                "future_bev_padding_mode": future_bev_info["future_bev_padding_mode"],
                "future_bev_frame_stats": future_bev_info["future_bev_frame_stats"],
            },
        )
