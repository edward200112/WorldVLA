from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from torch.optim import AdamW

from .config import ExperimentConfig
from .data import DriveSample
from .model import UnifiedDriveModel


def seed_everything(seed: int) -> None:
    """固定所有常见随机种子。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """把 batch 里的 tensor 全部移动到目标设备。"""
    moved_batch: Dict[str, Any] = {}
    for name, value in batch.items():
        if torch.is_tensor(value):
            moved_batch[name] = value.to(device)
        elif isinstance(value, list):
            moved_batch[name] = [item.to(device) if torch.is_tensor(item) else item for item in value]
        else:
            moved_batch[name] = value
    return moved_batch


def build_optimizer(model: UnifiedDriveModel, config: ExperimentConfig) -> AdamW:
    """只为可训练参数创建 AdamW 优化器。"""
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return AdamW(
        trainable_parameters,
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )


def train_one_epoch(
    model: UnifiedDriveModel,
    dataloader,
    optimizer: AdamW,
    device: torch.device,
    max_grad_norm: float,
    log_every: int,
) -> float:
    """执行一个清晰、直接的训练 epoch。"""
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(dataloader, start=1):
        batch = move_batch_to_device(batch, device)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_types=batch["token_types"],
            pixel_values_list=batch["pixel_values_list"],
            image_sizes_list=batch["image_sizes_list"],
            labels=batch["labels"],
        )
        loss = outputs["loss"]
        if loss is None:
            raise RuntimeError("训练阶段 loss 不应为空。")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        if step % log_every == 0:
            print(f"[train] step={step} loss={loss.item():.4f}")
    return total_loss / max(len(dataloader), 1)


def save_experiment_artifacts(
    model: UnifiedDriveModel,
    config: ExperimentConfig,
    checkpoint_dir: Path,
) -> None:
    """保存最小检查点与配置文件。"""
    model.save_checkpoint(checkpoint_dir)
    with (checkpoint_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(asdict(config), file, ensure_ascii=False, indent=2)


def _restrict_argmax(logits: torch.Tensor, allowed_token_ids: Sequence[int]) -> int:
    """只在允许的 token 子集内做贪心解码。"""
    allowed_logits = logits[torch.tensor(list(allowed_token_ids), device=logits.device)]
    best_index = int(torch.argmax(allowed_logits).item())
    return int(list(allowed_token_ids)[best_index])


@torch.no_grad()
def _decode_positions_from_logits(
    logits: torch.Tensor,
    query_positions: Sequence[int],
    allowed_token_ids: Sequence[int],
) -> List[int]:
    """从单次前向得到的 logits 中抽取指定位置的预测 token。"""
    predicted_ids: List[int] = []
    for position in query_positions:
        token_logits = logits[0, position]
        predicted_ids.append(_restrict_argmax(token_logits, allowed_token_ids))
    return predicted_ids


@torch.no_grad()
def greedy_rollout(
    model: UnifiedDriveModel,
    sample: DriveSample,
    device: torch.device,
) -> Dict[str, Any]:
    """按固定结构贪心生成 future action 和未来 3 帧 BEV。"""
    model.eval()
    query_encoding = model.discretizer.build_generation_queries(sample)
    input_tensor = torch.tensor([query_encoding.input_ids], dtype=torch.long, device=device)
    token_type_tensor = torch.tensor([query_encoding.token_types], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_tensor)
    outputs = model(
        input_ids=input_tensor,
        attention_mask=attention_mask,
        token_types=token_type_tensor,
        pixel_values_list=[query_encoding.pixel_values.to(device)],
        image_sizes_list=[query_encoding.image_sizes.to(device)],
        labels=None,
    )
    logits = outputs["logits"]
    predicted_action_tokens = _decode_positions_from_logits(
        logits=logits,
        query_positions=query_encoding.action_query_positions,
        allowed_token_ids=model.discretizer.layout.action_token_ids,
    )
    predicted_bev_tokens: List[List[int]] = []
    for frame_positions in query_encoding.future_bev_query_positions:
        predicted_bev_tokens.append(
            _decode_positions_from_logits(
                logits=logits,
                query_positions=frame_positions,
                allowed_token_ids=model.discretizer.layout.bev_token_ids,
            )
        )

    trajectory = model.discretizer.decode_action_tokens_to_trajectory(predicted_action_tokens)
    decoded_bevs = [model.discretizer.decode_bev_token_ids(tokens) for tokens in predicted_bev_tokens]
    return {
        "predicted_action_tokens": predicted_action_tokens,
        "predicted_trajectory": trajectory,
        "predicted_future_bevs": decoded_bevs,
    }
