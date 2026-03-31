"""Planner model composing all encoders, fusion, and task heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import torch
from torch import nn

from language_waypoint_planner.configs.schema import ModelConfig
from language_waypoint_planner.data.tokenizer import HashTextTokenizer

from .ego_encoder import EgoEncoder
from .fusion_transformer import FusionTransformer
from .heads import BehaviorHead, RationaleHead, WaypointHead
from .route_encoder import RouteCommandEncoder
from .text_encoder import TextEncoder
from .vision_encoder import VisionEncoder


@dataclass
class PlannerOutput:
    """Structured model output."""

    pred_waypoints: torch.Tensor
    pred_behavior_logits: torch.Tensor
    pred_rationale_logits_or_tokens: torch.Tensor
    aux_outputs: Dict[str, Any]


class PlannerModel(nn.Module):
    """Language-enhanced waypoint planner."""

    def __init__(
        self,
        config: ModelConfig,
        image_size: tuple[int, int],
        temporal_window: int,
        num_cameras: int,
        history_steps: int,
        future_steps: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.future_steps = future_steps
        self.vision_encoder = VisionEncoder(
            encoder_type=config.vision_encoder_type,
            image_size=image_size,
            output_dim=config.fusion_dim,
            temporal_window=temporal_window,
            num_cameras=num_cameras,
            patch_size=config.lite_vit_patch_size,
            dropout=config.dropout,
        )
        self.ego_encoder = EgoEncoder(
            encoder_type=config.ego_encoder_type,
            history_steps=history_steps,
            output_dim=config.fusion_dim,
            num_heads=config.ego_transformer_heads,
            num_layers=config.ego_transformer_layers,
            dropout=config.dropout,
        )
        self.route_encoder = RouteCommandEncoder(output_dim=config.fusion_dim)
        self.text_encoder = TextEncoder(
            backend=config.text_backend,
            output_dim=config.fusion_dim,
            max_length=config.text_max_length,
            vocab_size=config.text_vocab_size,
            freeze=config.freeze_text_encoder,
            model_name=config.text_model_name,
            dropout=config.dropout,
        )
        self.fusion_transformer = FusionTransformer(
            hidden_dim=config.fusion_dim,
            num_heads=config.fusion_heads,
            num_layers=config.fusion_layers,
            dropout=config.dropout,
            max_tokens=config.max_fusion_tokens,
        )
        self.waypoint_head = WaypointHead(hidden_dim=config.fusion_dim, future_steps=future_steps)
        self.behavior_head = BehaviorHead(hidden_dim=config.fusion_dim)
        self.rationale_head = RationaleHead(
            hidden_dim=config.fusion_dim,
            vocab_size=config.rationale_vocab_size,
            max_length=config.rationale_max_length,
        )
        self.rationale_tokenizer = HashTextTokenizer(vocab_size=config.rationale_vocab_size)

    @classmethod
    def from_config(
        cls,
        config: ModelConfig,
        image_size: tuple[int, int],
        temporal_window: int,
        num_cameras: int,
        history_steps: int,
        future_steps: int,
    ) -> "PlannerModel":
        """Construct a planner model from a dataclass config."""

        return cls(
            config=config,
            image_size=image_size,
            temporal_window=temporal_window,
            num_cameras=num_cameras,
            history_steps=history_steps,
            future_steps=future_steps,
        )

    def forward(
        self,
        images: torch.Tensor,
        ego_hist: torch.Tensor,
        velocity: torch.Tensor,
        acceleration: torch.Tensor,
        route_command_ids: torch.Tensor,
        language_input: Sequence[str],
    ) -> PlannerOutput:
        """Forward pass over all available modalities."""

        device = images.device
        vision_tokens, vision_valid, vision_aux = self.vision_encoder(images)
        ego_tokens, ego_valid, ego_aux = self.ego_encoder(ego_hist, velocity, acceleration)
        route_tokens, route_valid, route_aux = self.route_encoder(route_command_ids)
        language_tokens, language_valid, text_aux = self.text_encoder(language_input, device=device)
        fused = self.fusion_transformer(
            visual_tokens=vision_tokens,
            visual_valid=vision_valid,
            ego_tokens=ego_tokens,
            ego_valid=ego_valid,
            route_tokens=route_tokens,
            route_valid=route_valid,
            language_tokens=language_tokens,
            language_valid=language_valid,
        )

        global_token = fused["global_token"]
        pred_waypoints = self.waypoint_head(global_token)
        pred_behavior_logits = self.behavior_head(global_token)
        pred_rationale_logits = self.rationale_head(global_token)
        return PlannerOutput(
            pred_waypoints=pred_waypoints,
            pred_behavior_logits=pred_behavior_logits,
            pred_rationale_logits_or_tokens=pred_rationale_logits,
            aux_outputs={
                "vision": vision_aux,
                "ego": ego_aux,
                "route": route_aux,
                "text": text_aux,
                "fusion": fused,
            },
        )

    def decode_rationale_logits(self, logits: torch.Tensor) -> list[str]:
        """Decode fixed-length rationale logits into coarse text tokens."""

        token_ids = logits.argmax(dim=-1)
        return [self.rationale_tokenizer.decode(tokens.tolist()) for tokens in token_ids]
