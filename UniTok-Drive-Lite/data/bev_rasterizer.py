"""Deprecated: 顶层 BEV 渲染实验组件。

当前仓库的唯一权威运行主链路是 `scripts/` + `unitok_drive_lite/`。
本文件继续保留给数据预处理实验和 demo 使用，不再作为官方主链路的一部分维护。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw


Color = tuple[int, int, int]


@dataclass
class BEVRasterizerConfig:
    """保存 BEV 栅格图渲染配置。"""

    image_size: int = 224
    pixels_per_meter: float = 8.0
    background_color: Color = (0, 0, 0)
    drivable_color: Color = (40, 40, 40)
    ego_history_color: Color = (255, 255, 255)
    vehicle_color: Color = (0, 120, 255)
    pedestrian_color: Color = (255, 60, 60)
    lane_color: Color = (0, 255, 0)
    ego_current_color: Color = (255, 255, 255)
    ego_history_width: int = 3
    lane_width: int = 2
    pedestrian_radius_px: int = 4
    vehicle_half_length_m: float = 2.2
    vehicle_half_width_m: float = 1.0
    ego_current_radius_px: int = 5


def _to_numpy_points(points: object) -> np.ndarray:
    """把任意点集输入统一转成 shape 为 [N, 2] 的 numpy 数组。"""
    if points is None:
        return np.zeros((0, 2), dtype=np.float32)

    array = np.asarray(points, dtype=np.float32)
    if array.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    if array.ndim == 1:
        if array.shape[0] < 2:
            raise ValueError(f"单个点至少需要两个坐标分量，当前收到 shape={array.shape}")
        array = array[None, :]

    if array.ndim != 2:
        raise ValueError(f"点集应为二维数组，当前收到 shape={array.shape}")
    if array.shape[1] < 2:
        raise ValueError(f"点集最后一维至少需要 2，当前收到 shape={array.shape}")

    return array[:, :2]


def _to_polyline_list(polylines: object) -> list[np.ndarray]:
    """把多条折线输入统一转成由 [Ni, 2] 数组组成的列表。"""
    if polylines is None:
        return []

    if isinstance(polylines, np.ndarray):
        if polylines.ndim == 3:
            return [_to_numpy_points(polyline) for polyline in polylines]
        if polylines.ndim == 2:
            return [_to_numpy_points(polylines)]

    if isinstance(polylines, (list, tuple)):
        if len(polylines) == 0:
            return []
        return [_to_numpy_points(polyline) for polyline in polylines]

    raise ValueError("lane_polylines 必须是列表、元组或 numpy 数组。")


def _round_point(point: Sequence[float]) -> tuple[int, int]:
    """把浮点像素坐标转成整数像素坐标。"""
    return int(round(float(point[0]))), int(round(float(point[1])))


class BEVRasterizer:
    """把 ego-centric 场景元素渲染成 RGB BEV 图像。"""

    def __init__(self, config: BEVRasterizerConfig | None = None) -> None:
        """初始化渲染器。"""
        self.config = config or BEVRasterizerConfig()

    def _world_to_image(self, points: np.ndarray) -> np.ndarray:
        """把 ego 坐标系下的米制坐标转换到图像像素坐标。

        坐标约定：
        - 输入坐标使用 ego 局部坐标系
        - x 轴朝前
        - y 轴朝左
        - ego 位于图像中心
        - 前方朝上

        张量 shape：
        - points: [N, 2]
        - image_points: [N, 2]
        """
        if points.shape[0] == 0:
            return points.copy()

        center = (self.config.image_size - 1) / 2.0
        image_points = np.zeros_like(points, dtype=np.float32)

        # 前方朝上，因此 x 越大，图像 y 越小。
        image_points[:, 1] = center - points[:, 0] * self.config.pixels_per_meter

        # 左侧朝左，因此 y 越大，图像 x 越小。
        image_points[:, 0] = center - points[:, 1] * self.config.pixels_per_meter
        return image_points

    def _draw_polyline(
        self,
        draw: ImageDraw.ImageDraw,
        polyline: np.ndarray,
        color: Color,
        width: int,
    ) -> None:
        """在图像上绘制一条折线。"""
        if polyline.shape[0] == 0:
            return

        pixel_points = self._world_to_image(polyline)
        if pixel_points.shape[0] == 1:
            center_x, center_y = _round_point(pixel_points[0])
            radius = max(width, 1)
            draw.ellipse(
                (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
                fill=color,
            )
            return

        draw.line([_round_point(point) for point in pixel_points], fill=color, width=width)

    def _draw_filled_polygon(
        self,
        draw: ImageDraw.ImageDraw,
        polygon: np.ndarray,
        color: Color,
    ) -> None:
        """在图像上绘制填充 polygon。"""
        if polygon.shape[0] < 3:
            return
        pixel_points = self._world_to_image(polygon)
        draw.polygon([_round_point(point) for point in pixel_points], fill=color)

    def _draw_pedestrians(
        self,
        draw: ImageDraw.ImageDraw,
        pedestrian_centers: np.ndarray,
    ) -> None:
        """绘制行人中心点。"""
        if pedestrian_centers.shape[0] == 0:
            return

        pixel_points = self._world_to_image(pedestrian_centers)
        radius = self.config.pedestrian_radius_px
        for point in pixel_points:
            center_x, center_y = _round_point(point)
            draw.ellipse(
                (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
                fill=self.config.pedestrian_color,
            )

    def _draw_vehicles(
        self,
        draw: ImageDraw.ImageDraw,
        vehicle_centers: np.ndarray,
    ) -> None:
        """绘制车辆中心点对应的固定大小矩形。"""
        if vehicle_centers.shape[0] == 0:
            return

        pixel_points = self._world_to_image(vehicle_centers)
        half_length_px = self.config.vehicle_half_length_m * self.config.pixels_per_meter
        half_width_px = self.config.vehicle_half_width_m * self.config.pixels_per_meter

        for point in pixel_points:
            center_x, center_y = float(point[0]), float(point[1])
            draw.rectangle(
                (
                    int(round(center_x - half_width_px)),
                    int(round(center_y - half_length_px)),
                    int(round(center_x + half_width_px)),
                    int(round(center_y + half_length_px)),
                ),
                fill=self.config.vehicle_color,
            )

    def _draw_ego_history(
        self,
        draw: ImageDraw.ImageDraw,
        ego_history: np.ndarray,
    ) -> None:
        """绘制 ego 历史轨迹和当前 ego 位置。"""
        if ego_history.shape[0] > 0:
            self._draw_polyline(
                draw=draw,
                polyline=ego_history,
                color=self.config.ego_history_color,
                width=self.config.ego_history_width,
            )

        center = (self.config.image_size - 1) / 2.0
        radius = self.config.ego_current_radius_px
        draw.ellipse(
            (
                int(round(center - radius)),
                int(round(center - radius)),
                int(round(center + radius)),
                int(round(center + radius)),
            ),
            fill=self.config.ego_current_color,
        )

    def _extract_sample_fields(self, sample: Mapping[str, object]) -> dict[str, object]:
        """从输入样本中提取各类场景元素，并兼容常见别名。"""
        return {
            "ego_history": sample.get("ego_history", sample.get("ego_history_traj")),
            "vehicle_centers": sample.get("vehicle_centers", sample.get("vehicles")),
            "pedestrian_centers": sample.get("pedestrian_centers", sample.get("pedestrians")),
            "lane_polylines": sample.get("lane_polylines", sample.get("lanes")),
            "drivable_polygon": sample.get("drivable_polygon", sample.get("drivable_area")),
        }

    def render(self, sample: Mapping[str, object]) -> Image.Image:
        """渲染一张 RGB BEV 图像。

        约定输入 sample 至少支持以下字段：
        - ego_history: [N, 2] 或 [N, 3]
        - vehicle_centers: [M, 2] 或 [M, 3]
        - pedestrian_centers: [P, 2] 或 [P, 3]
        - lane_polylines: list[[Li, 2]] 或 [L, 2] 或 [K, L, 2]
        - drivable_polygon: [Q, 2] 或 [Q, 3]，可选
        """
        fields = self._extract_sample_fields(sample)
        ego_history = _to_numpy_points(fields["ego_history"])
        vehicle_centers = _to_numpy_points(fields["vehicle_centers"])
        pedestrian_centers = _to_numpy_points(fields["pedestrian_centers"])
        lane_polylines = _to_polyline_list(fields["lane_polylines"])
        drivable_polygon = _to_numpy_points(fields["drivable_polygon"])

        canvas = np.full(
            (self.config.image_size, self.config.image_size, 3),
            self.config.background_color,
            dtype=np.uint8,
        )
        image = Image.fromarray(canvas)
        draw = ImageDraw.Draw(image)

        # 绘制顺序从底到顶，避免语义层被后面的图元遮挡。
        if drivable_polygon.shape[0] >= 3:
            self._draw_filled_polygon(draw, drivable_polygon, self.config.drivable_color)

        for polyline in lane_polylines:
            self._draw_polyline(
                draw=draw,
                polyline=polyline,
                color=self.config.lane_color,
                width=self.config.lane_width,
            )

        self._draw_vehicles(draw, vehicle_centers)
        self._draw_pedestrians(draw, pedestrian_centers)
        self._draw_ego_history(draw, ego_history)
        return image


def render_bev(sample: dict) -> Image.Image:
    """使用默认配置把场景字典渲染成一张 224x224 的 PIL RGB 图像。"""
    rasterizer = BEVRasterizer()
    return rasterizer.render(sample)


if __name__ == "__main__":
    """执行最小 demo，生成一张示例 BEV 图像。"""
    demo_sample = {
        # ego 局部坐标系，x 朝前，y 朝左，单位为米。
        "ego_history": np.array(
            [
                [-8.0, 0.2],
                [-6.0, 0.15],
                [-4.0, 0.1],
                [-2.0, 0.05],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "vehicle_centers": np.array(
            [
                [12.0, -2.5],
                [18.0, 2.0],
                [6.0, 4.0],
            ],
            dtype=np.float32,
        ),
        "pedestrian_centers": np.array(
            [
                [10.0, 5.5],
                [7.0, -5.0],
            ],
            dtype=np.float32,
        ),
        "lane_polylines": [
            np.array([[-15.0, 3.5], [-8.0, 3.3], [0.0, 3.1], [10.0, 3.0], [20.0, 2.8]], dtype=np.float32),
            np.array([[-15.0, -3.5], [-8.0, -3.4], [0.0, -3.3], [10.0, -3.2], [20.0, -3.0]], dtype=np.float32),
            np.array([[-15.0, 0.0], [-8.0, 0.0], [0.0, 0.0], [10.0, 0.0], [20.0, 0.0]], dtype=np.float32),
        ],
        "drivable_polygon": np.array(
            [
                [-20.0, -6.0],
                [24.0, -6.0],
                [24.0, 6.0],
                [-20.0, 6.0],
            ],
            dtype=np.float32,
        ),
    }

    output_image = render_bev(demo_sample)
    output_path = Path(__file__).with_name("bev_demo.png")
    output_image.save(output_path)
    print(f"BEV demo image saved to: {output_path}")
