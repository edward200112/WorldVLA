"""Fusion transformer for multimodal token aggregation."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn


class FusionTransformer(nn.Module):
    """Fuse visual, ego, route, and language tokens into a shared representation."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        max_tokens: int = 128,
    ) -> None:
        super().__init__()
        self.max_tokens = max_tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, max_tokens + 1, hidden_dim))
        self.modality_embedding = nn.Embedding(5, hidden_dim)
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
        self.norm = nn.LayerNorm(hidden_dim)

    def _make_modality_ids(
        self,
        batch_size: int,
        lengths: Tuple[int, int, int, int],
        device: torch.device,
    ) -> torch.Tensor:
        visual_len, ego_len, route_len, language_len = lengths
        ids = torch.cat(
            [
                torch.full((visual_len,), 0, dtype=torch.long, device=device),
                torch.full((ego_len,), 1, dtype=torch.long, device=device),
                torch.full((route_len,), 2, dtype=torch.long, device=device),
                torch.full((language_len,), 3, dtype=torch.long, device=device),
            ],
            dim=0,
        )
        return ids.unsqueeze(0).expand(batch_size, -1)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        visual_valid: torch.Tensor,
        ego_tokens: torch.Tensor,
        ego_valid: torch.Tensor,
        route_tokens: torch.Tensor,
        route_valid: torch.Tensor,
        language_tokens: Optional[torch.Tensor] = None,
        language_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Fuse modality tokens and return the global token plus contextual sequence."""

        if language_tokens is None:
            language_tokens = visual_tokens.new_zeros(visual_tokens.shape[0], 0, visual_tokens.shape[-1])
            language_valid = visual_valid.new_zeros(visual_valid.shape[0], 0)
        if language_valid is None:
            raise ValueError("language_valid must be provided when language_tokens is not None")

        tokens = torch.cat([visual_tokens, ego_tokens, route_tokens, language_tokens], dim=1)
        valid_mask = torch.cat([visual_valid, ego_valid, route_valid, language_valid], dim=1)
        total_tokens = tokens.shape[1]
        if total_tokens > self.max_tokens:
            raise ValueError(
                f"FusionTransformer received {total_tokens} tokens, exceeding max_tokens={self.max_tokens}"
            )

        batch_size = tokens.shape[0]
        modality_ids = self._make_modality_ids(
            batch_size=batch_size,
            lengths=(visual_tokens.shape[1], ego_tokens.shape[1], route_tokens.shape[1], language_tokens.shape[1]),
            device=tokens.device,
        )
        modality_tokens = self.modality_embedding(modality_ids)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        cls_modality = self.modality_embedding(
            torch.full((batch_size, 1), 4, dtype=torch.long, device=tokens.device)
        )
        fused = torch.cat([cls_token + cls_modality, tokens + modality_tokens], dim=1)
        fused = fused + self.position_embedding[:, : fused.shape[1]]
        fused_valid = torch.cat(
            [torch.ones(batch_size, 1, dtype=torch.bool, device=tokens.device), valid_mask],
            dim=1,
        )
        encoded = self.encoder(fused, src_key_padding_mask=~fused_valid)
        encoded = self.norm(encoded)
        return {
            "global_token": encoded[:, 0],
            "sequence_tokens": encoded[:, 1:],
            "valid_mask": fused_valid[:, 1:],
        }
