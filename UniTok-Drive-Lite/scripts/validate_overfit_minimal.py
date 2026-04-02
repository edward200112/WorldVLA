"""最小 overfit 验证脚本。

用途：
- 在 toy 或 nuScenes 的极小子集上快速过拟合
- 观察 loss 是否下降
- 对比一个样本的 GT future action token 与预测 token
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unitok_drive_lite import (
    UnifiedDriveCollator,
    UnifiedDriveModel,
    build_default_config,
)
from unitok_drive_lite.script_utils import add_dataset_selection_args, build_dataset_from_args
from unitok_drive_lite.train_utils import (
    build_optimizer,
    greedy_rollout,
    save_experiment_artifacts,
    seed_everything,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    """解析最小 overfit 验证脚本参数。"""
    parser = argparse.ArgumentParser(description="在极小数据子集上做最小 overfit 验证。")
    add_dataset_selection_args(parser, include_focus_scene_token=True)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_train_samples", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="outputs/unitok_drive_lite_overfit")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    return parser.parse_args()


def _build_train_subset(dataset: Any, max_train_samples: int) -> Subset:
    """限制 overfit 用的训练子集大小。"""
    if max_train_samples <= 0:
        raise ValueError(f"--max_train_samples 必须大于 0，当前收到 {max_train_samples}")
    subset_size = min(len(dataset), max_train_samples)
    if subset_size <= 0:
        raise RuntimeError("构建 overfit 子集时没有可用样本。")
    return Subset(dataset, list(range(subset_size)))


def _trajectory_stats(predicted: torch.Tensor, target: torch.Tensor) -> Dict[str, Any]:
    """返回一组简洁的轨迹对比指标。"""
    if predicted.shape != target.shape:
        raise ValueError(
            f"预测轨迹与目标轨迹 shape 不一致: predicted={tuple(predicted.shape)} target={tuple(target.shape)}"
        )
    difference = predicted - target
    final_delta = predicted[-1] - target[-1]
    return {
        "predicted_shape": list(predicted.shape),
        "target_shape": list(target.shape),
        "mean_abs_error": float(difference.abs().mean().item()),
        "max_abs_error": float(difference.abs().max().item()),
        "final_step_l2": float(torch.linalg.vector_norm(final_delta).item()),
        "predicted_final_point": predicted[-1].tolist(),
        "target_final_point": target[-1].tolist(),
    }


def main() -> None:
    """运行最小 overfit 验证。"""
    args = parse_args()
    config = build_default_config(PROJECT_ROOT)
    config.train.num_epochs = args.num_epochs
    config.train.output_dir = args.output_dir
    config.model.load_in_4bit = args.load_in_4bit
    config.model.gradient_checkpointing = not args.no_gradient_checkpointing

    seed_everything(config.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        full_dataset = build_dataset_from_args(
            args=args,
            token_config=config.tokens,
            seed=config.train.seed,
        )
    except (ImportError, FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"[data] {error}") from error

    train_dataset = _build_train_subset(full_dataset, args.max_train_samples)
    config.train.dataset_size = len(train_dataset)
    print(
        f"[data] dataset_type={args.dataset_type} raw_dataset_size={len(full_dataset)} "
        f"train_subset_size={len(train_dataset)}"
    )

    model = UnifiedDriveModel(config).to(device)
    print(f"[model] backbone={config.model.model_name}")
    print(
        f"[model] load_in_4bit={config.model.load_in_4bit} "
        f"gradient_checkpointing={config.model.gradient_checkpointing}"
    )

    collator = UnifiedDriveCollator(
        discretizer=model.discretizer,
        pad_token_id=model.tokenizer.pad_token_id,
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        collate_fn=collator,
    )
    optimizer = build_optimizer(model, config)

    epoch_losses: list[float] = []
    for epoch in range(config.train.num_epochs):
        average_loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            max_grad_norm=config.train.max_grad_norm,
            log_every=config.train.log_every,
        )
        epoch_losses.append(float(average_loss))
        print(f"[overfit] epoch={epoch + 1} average_loss={average_loss:.4f}")

    checkpoint_dir = PROJECT_ROOT / config.train.output_dir / "checkpoint_last"
    save_experiment_artifacts(model, config, checkpoint_dir)
    print(f"[save] checkpoint_dir={checkpoint_dir}")

    reference_sample = train_dataset[0]
    gt_action_tokens = model.discretizer.encode_future_actions(reference_sample.future_actions)
    gt_quantized_trajectory = model.discretizer.decode_action_tokens_to_trajectory(gt_action_tokens)
    raw_future_trajectory = torch.cumsum(reference_sample.future_actions, dim=0)

    rollout_output = greedy_rollout(model, reference_sample, device)
    predicted_action_tokens = rollout_output["predicted_action_tokens"]
    predicted_trajectory = rollout_output["predicted_trajectory"].detach().cpu()
    token_match_count = sum(
        int(predicted == target)
        for predicted, target in zip(predicted_action_tokens, gt_action_tokens)
    )

    quantized_stats = _trajectory_stats(predicted_trajectory, gt_quantized_trajectory)
    raw_stats = _trajectory_stats(predicted_trajectory, raw_future_trajectory)
    summary = {
        "dataset_type": args.dataset_type,
        "raw_dataset_size": len(full_dataset),
        "train_subset_size": len(train_dataset),
        "epoch_losses": epoch_losses,
        "reference_metadata": reference_sample.metadata,
        "gt_action_tokens": gt_action_tokens,
        "predicted_action_tokens": predicted_action_tokens,
        "token_match_count": token_match_count,
        "token_match_ratio": token_match_count / max(len(gt_action_tokens), 1),
        "quantized_trajectory_stats": quantized_stats,
        "raw_trajectory_stats": raw_stats,
    }

    summary_path = checkpoint_dir / "overfit_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("[report] gt_action_tokens=", gt_action_tokens)
    print("[report] predicted_action_tokens=", predicted_action_tokens)
    print(
        f"[report] token_match_count={token_match_count} "
        f"token_match_ratio={summary['token_match_ratio']:.3f}"
    )
    print("[report] quantized_trajectory_stats=", quantized_stats)
    print("[report] raw_trajectory_stats=", raw_stats)
    print(f"[report] summary_json={summary_path}")


if __name__ == "__main__":
    main()
