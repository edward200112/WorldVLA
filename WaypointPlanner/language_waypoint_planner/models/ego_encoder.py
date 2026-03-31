"""Encoders for ego-state history."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn


class EgoEncoder(nn.Module):
    """Encode ego history, velocity, and acceleration into one or more tokens."""

    def __init__(
        self,
        encoder_type: str,
        history_steps: int,
        output_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder_type = encoder_type
        self.history_steps = history_steps
        feature_dim = 4

        if encoder_type == "mlp":
            self.encoder = nn.Sequential(
                nn.Linear(history_steps * feature_dim, output_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(output_dim * 2, output_dim),
            )
        elif encoder_type == "transformer":
            self.input_projection = nn.Linear(feature_dim, output_dim)
            self.position_embedding = nn.Parameter(torch.zeros(1, history_steps, output_dim))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=output_dim,
                nhead=num_heads,
                dim_feedforward=output_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
                enable_nested_tensor=False,
            )
            self.norm = nn.LayerNorm(output_dim)
        else:
            raise ValueError(f"Unsupported ego encoder type: {encoder_type}")

    def forward(
        self,
        ego_hist: torch.Tensor,
        velocity: torch.Tensor,
        acceleration: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Return ego tokens and validity mask."""

        if ego_hist.ndim != 3 or ego_hist.shape[-1] != 2:
            raise ValueError(f"Expected ego_hist with shape [B, L, 2], got {tuple(ego_hist.shape)}")
        if velocity.ndim != 3 or velocity.shape[-1] != 1:
            raise ValueError(f"Expected velocity with shape [B, L, 1], got {tuple(velocity.shape)}")
        if acceleration.ndim != 3 or acceleration.shape[-1] != 1:
            raise ValueError(
                f"Expected acceleration with shape [B, L, 1], got {tuple(acceleration.shape)}"
            )
        batch_size, history_steps, _ = ego_hist.shape
        if history_steps != self.history_steps:
            raise ValueError(f"Expected history_steps={self.history_steps}, got {history_steps}")
        features = torch.cat([ego_hist, velocity, acceleration], dim=-1)

        if self.encoder_type == "mlp":
            token = self.encoder(features.reshape(batch_size, history_steps * features.shape[-1])).unsqueeze(1)
            valid_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=ego_hist.device)
            return token, valid_mask, {"ego_features": features}

        tokens = self.input_projection(features) + self.position_embedding[:, :history_steps]
        encoded = self.encoder(tokens)
        encoded = self.norm(encoded)
        valid_mask = torch.ones(batch_size, history_steps, dtype=torch.bool, device=ego_hist.device)
        return encoded, valid_mask, {"ego_features": features}
