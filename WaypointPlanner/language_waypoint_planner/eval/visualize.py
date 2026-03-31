"""Qualitative visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw


def _to_pil_image(image_tensor: torch.Tensor) -> Image.Image:
    array = image_tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    array = (array * 255.0).astype(np.uint8)
    return Image.fromarray(array)


def _project_waypoint(point: torch.Tensor, origin: tuple[float, float], scale: float) -> tuple[float, float]:
    forward = float(point[0])
    lateral = float(point[1])
    x = origin[0] + lateral * scale
    y = origin[1] - forward * scale
    return x, y


def save_prediction_visualizations(
    batch: dict,
    pred_waypoints: torch.Tensor,
    pred_rationales: Sequence[str],
    output_dir: Path,
    max_items: int = 8,
) -> List[Path]:
    """Save trajectory overlays and metadata to disk."""

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []
    count = min(max_items, pred_waypoints.shape[0])
    for index in range(count):
        front_image = batch["images"][index, -1, 0]
        image = _to_pil_image(front_image).resize((640, 360))
        draw = ImageDraw.Draw(image)
        origin = (image.width * 0.5, image.height * 0.88)
        scale = 18.0

        gt_waypoints = batch["target_waypoints"][index].detach().cpu()
        pred_points = pred_waypoints[index].detach().cpu()

        if batch["valid_masks"]["waypoints"][index]:
            gt_pixels = [_project_waypoint(point, origin=origin, scale=scale) for point in gt_waypoints]
            draw.line(gt_pixels, fill=(32, 220, 32), width=4)
            for pixel in gt_pixels:
                draw.ellipse((pixel[0] - 2, pixel[1] - 2, pixel[0] + 2, pixel[1] + 2), fill=(32, 220, 32))

        pred_pixels = [_project_waypoint(point, origin=origin, scale=scale) for point in pred_points]
        draw.line(pred_pixels, fill=(220, 32, 32), width=4)
        for pixel in pred_pixels:
            draw.ellipse((pixel[0] - 2, pixel[1] - 2, pixel[0] + 2, pixel[1] + 2), fill=(220, 32, 32))

        draw.rectangle((8, 8, image.width - 8, 82), fill=(0, 0, 0))
        draw.text((16, 16), f"route: {batch['route_command_text'][index]}", fill=(255, 255, 255))
        draw.text((16, 36), f"rationale: {pred_rationales[index][:72]}", fill=(255, 255, 255))
        draw.text((16, 56), "red=pred, green=gt", fill=(255, 255, 255))
        output_path = output_dir / f"sample_{index:03d}.png"
        image.save(output_path)
        saved_paths.append(output_path)
    return saved_paths
