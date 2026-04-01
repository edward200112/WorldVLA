from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn


class ActionTokenizer(nn.Module):
    """使用简单 MLP + 可学习 codebook 的 VQ-VAE 风格轨迹离散器。"""

    def __init__(
        self,
        trajectory_horizon: int,
        action_dim: int = 3,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        num_latent_tokens: int = 4,
        codebook_size: int = 512,
        commitment_cost: float = 0.25,
    ) -> None:
        """初始化 action tokenizer。

        参数说明：
        - trajectory_horizon: 未来轨迹长度 T
        - action_dim: 每个时间步的动作维度，默认是 (x, y, yaw) 共 3 维
        - hidden_dim: MLP 隐层维度
        - latent_dim: 每个离散 token 对应的连续隐变量维度
        - num_latent_tokens: 每条轨迹会被压缩成多少个离散 token
        - codebook_size: 码本大小
        - commitment_cost: VQ-VAE 中 commitment loss 的系数
        """
        super().__init__()
        if trajectory_horizon <= 0:
            raise ValueError("trajectory_horizon 必须大于 0。")
        if action_dim <= 0:
            raise ValueError("action_dim 必须大于 0。")
        if num_latent_tokens <= 0:
            raise ValueError("num_latent_tokens 必须大于 0。")
        if codebook_size <= 1:
            raise ValueError("codebook_size 必须大于 1。")

        self.trajectory_horizon = trajectory_horizon
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_latent_tokens = num_latent_tokens
        self.codebook_size = codebook_size
        self.commitment_cost = commitment_cost
        self.flat_action_dim = trajectory_horizon * action_dim

        # 编码器输入 shape: [B, T, 3] -> [B, T * 3]
        # 编码器输出 shape: [B, num_latent_tokens * latent_dim]
        self.encoder = nn.Sequential(
            nn.Linear(self.flat_action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_latent_tokens * latent_dim),
        )

        # codebook shape: [codebook_size, latent_dim]
        self.codebook = nn.Embedding(codebook_size, latent_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

        # 解码器输入 shape: [B, num_latent_tokens, latent_dim] -> [B, num_latent_tokens * latent_dim]
        # 解码器输出 shape: [B, T * 3] -> [B, T, 3]
        self.decoder = nn.Sequential(
            nn.Linear(num_latent_tokens * latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.flat_action_dim),
        )

    def _validate_actions(self, actions: torch.Tensor) -> None:
        """校验输入轨迹张量的 shape 是否正确。"""
        if actions.ndim != 3:
            raise ValueError(f"actions 的 shape 应为 [B, T, 3]，当前收到 {tuple(actions.shape)}")
        if actions.shape[1] != self.trajectory_horizon:
            raise ValueError(
                f"actions 的时间长度 T 必须等于 {self.trajectory_horizon}，当前收到 {actions.shape[1]}"
            )
        if actions.shape[2] != self.action_dim:
            raise ValueError(
                f"actions 的最后一维必须等于 {self.action_dim}，当前收到 {actions.shape[2]}"
            )

    def _encode(self, actions: torch.Tensor) -> torch.Tensor:
        """把连续轨迹编码成连续隐变量。

        张量 shape 变化：
        - actions: [B, T, 3]
        - flat_actions: [B, T * 3]
        - encoder_output: [B, num_latent_tokens * latent_dim]
        - latents: [B, num_latent_tokens, latent_dim]
        """
        self._validate_actions(actions)

        batch_size = actions.shape[0]
        flat_actions = actions.reshape(batch_size, self.flat_action_dim)
        encoder_output = self.encoder(flat_actions)
        latents = encoder_output.reshape(batch_size, self.num_latent_tokens, self.latent_dim)
        return latents

    def _nearest_code_indices(self, latents: torch.Tensor) -> torch.Tensor:
        """为每个连续隐变量找到最近邻 codebook 索引。

        张量 shape 变化：
        - latents: [B, N, D]
        - flat_latents: [B * N, D]
        - codebook_weight: [K, D]
        - distances: [B * N, K]
        - indices: [B, N]
        其中：
        - N = num_latent_tokens
        - D = latent_dim
        - K = codebook_size
        """
        batch_size, num_tokens, latent_dim = latents.shape
        flat_latents = latents.reshape(batch_size * num_tokens, latent_dim)
        codebook_weight = self.codebook.weight

        # 计算平方欧氏距离，distances 越小表示越接近。
        latents_norm = flat_latents.pow(2).sum(dim=1, keepdim=True)
        codebook_norm = codebook_weight.pow(2).sum(dim=1).unsqueeze(0)
        distances = latents_norm + codebook_norm - 2.0 * flat_latents @ codebook_weight.t()

        indices = torch.argmin(distances, dim=1)
        return indices.reshape(batch_size, num_tokens)

    def _quantize(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """对连续隐变量做最近邻量化，并返回 straight-through 版本的量化结果。

        张量 shape 变化：
        - latents: [B, N, D]
        - indices: [B, N]
        - quantized: [B, N, D]
        - quantized_st: [B, N, D]
        """
        indices = self._nearest_code_indices(latents)
        quantized = self.codebook(indices)

        codebook_loss = F.mse_loss(quantized, latents.detach())
        commitment_loss = F.mse_loss(latents, quantized.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # 使用 straight-through estimator，让梯度能从 decoder 回传到 encoder。
        quantized_st = latents + (quantized - latents).detach()
        return quantized_st, indices, vq_loss

    def _decode_latents(self, quantized_latents: torch.Tensor) -> torch.Tensor:
        """把量化后的隐变量重建回连续轨迹。

        张量 shape 变化：
        - quantized_latents: [B, N, D]
        - flat_latents: [B, N * D]
        - decoder_output: [B, T * 3]
        - recon_actions: [B, T, 3]
        """
        batch_size = quantized_latents.shape[0]
        flat_latents = quantized_latents.reshape(batch_size, self.num_latent_tokens * self.latent_dim)
        decoder_output = self.decoder(flat_latents)
        recon_actions = decoder_output.reshape(batch_size, self.trajectory_horizon, self.action_dim)
        return recon_actions

    @torch.no_grad()
    def encode_to_indices(self, actions: torch.Tensor) -> torch.Tensor:
        """把连续轨迹编码成离散 token 索引。

        输入：
        - actions: [B, T, 3]

        输出：
        - indices: [B, num_latent_tokens]
        """
        latents = self._encode(actions)
        indices = self._nearest_code_indices(latents)
        return indices

    @torch.no_grad()
    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """把离散 token 索引解码回连续轨迹。

        输入：
        - indices: [B, num_latent_tokens]

        输出：
        - recon_actions: [B, T, 3]
        """
        if indices.ndim != 2:
            raise ValueError(f"indices 的 shape 应为 [B, num_latent_tokens]，当前收到 {tuple(indices.shape)}")
        if indices.shape[1] != self.num_latent_tokens:
            raise ValueError(
                f"indices 的第二维必须等于 {self.num_latent_tokens}，当前收到 {indices.shape[1]}"
            )

        quantized_latents = self.codebook(indices)
        recon_actions = self._decode_latents(quantized_latents)
        return recon_actions

    def forward(self, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """执行一次完整的 VQ-VAE 风格前向计算。

        输入：
        - actions: [B, T, 3]

        输出字段：
        - recon: [B, T, 3]，重建轨迹
        - indices: [B, num_latent_tokens]，离散 token 索引
        - vq_loss: []，向量量化损失
        - recon_loss: []，轨迹重建损失
        - loss: []，总训练损失
        """
        latents = self._encode(actions)
        quantized_latents, indices, vq_loss = self._quantize(latents)
        recon = self._decode_latents(quantized_latents)
        recon_loss = F.mse_loss(recon, actions)
        loss = recon_loss + vq_loss

        return {
            "recon": recon,
            "indices": indices,
            "vq_loss": vq_loss,
            "recon_loss": recon_loss,
            "loss": loss,
        }


if __name__ == "__main__":
    """做一个最小 smoke check，确认模块可独立前向与反向。"""
    batch_size = 2
    horizon = 10

    model = ActionTokenizer(
        trajectory_horizon=horizon,
        hidden_dim=128,
        latent_dim=32,
        num_latent_tokens=4,
        codebook_size=256,
    )
    actions = torch.randn(batch_size, horizon, 3)

    outputs = model(actions)
    outputs["loss"].backward()

    print("recon shape:", tuple(outputs["recon"].shape))
    print("indices shape:", tuple(outputs["indices"].shape))
    print("loss:", float(outputs["loss"].item()))
