"""Vision encoder variants for multi-camera image sequences."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
from torchvision import models


class LightweightViTBackbone(nn.Module):
    """Small ViT-style image backbone for efficient offline experiments."""

    def __init__(
        self,
        image_size: Tuple[int, int],
        patch_size: int,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError(f"Image size {image_size} must be divisible by patch_size={patch_size}")
        self.patch_embed = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of RGB images into a single feature vector per image."""

        patches = self.patch_embed(images).flatten(2).transpose(1, 2)
        tokens = patches + self.position_embedding[:, : patches.shape[1]]
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        return self.norm(pooled)


class VisionEncoder(nn.Module):
    """Encode `[B, T, N, 3, H, W]` image sequences into fusion-ready tokens."""

    def __init__(
        self,
        encoder_type: str,
        image_size: Tuple[int, int],
        output_dim: int,
        temporal_window: int,
        num_cameras: int,
        patch_size: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder_type = encoder_type
        self.temporal_window = temporal_window
        self.num_cameras = num_cameras
        self.output_dim = output_dim

        if encoder_type == "resnet50":
            backbone = models.resnet50(weights=None)
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            backbone_dim = 2048
        elif encoder_type == "lite_vit":
            self.backbone = LightweightViTBackbone(
                image_size=image_size,
                patch_size=patch_size,
                hidden_dim=output_dim,
                num_heads=4,
                num_layers=2,
                dropout=dropout,
            )
            backbone_dim = output_dim
        else:
            raise ValueError(f"Unsupported vision encoder type: {encoder_type}")

        self.projection = nn.Linear(backbone_dim, output_dim) if backbone_dim != output_dim else nn.Identity()
        self.time_embedding = nn.Embedding(temporal_window, output_dim)
        self.camera_embedding = nn.Embedding(num_cameras, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Return vision tokens and a validity mask."""

        if images.ndim != 6:
            raise ValueError(f"Expected images with shape [B, T, N, 3, H, W], got {tuple(images.shape)}")
        batch_size, time_steps, num_cameras, _, _, _ = images.shape
        if time_steps != self.temporal_window:
            raise ValueError(f"Expected temporal_window={self.temporal_window}, got {time_steps}")
        if num_cameras != self.num_cameras:
            raise ValueError(f"Expected num_cameras={self.num_cameras}, got {num_cameras}")

        flat_images = images.reshape(batch_size * time_steps * num_cameras, *images.shape[3:])
        if self.encoder_type == "resnet50":
            pooled = self.backbone(flat_images).flatten(1)
        else:
            pooled = self.backbone(flat_images)
        projected = self.projection(pooled)
        tokens = projected.reshape(batch_size, time_steps, num_cameras, self.output_dim)

        time_ids = torch.arange(time_steps, device=images.device).view(1, time_steps, 1)
        camera_ids = torch.arange(num_cameras, device=images.device).view(1, 1, num_cameras)
        tokens = tokens + self.time_embedding(time_ids) + self.camera_embedding(camera_ids)
        tokens = self.dropout(tokens.reshape(batch_size, time_steps * num_cameras, self.output_dim))
        valid_mask = torch.ones(batch_size, time_steps * num_cameras, dtype=torch.bool, device=images.device)
        return tokens, valid_mask, {"image_features": projected}
