"""最小 overfit 验证脚本。"""

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
from unitok_drive_lite.eval_utils import (
    future_bev_difference_summary,
    token_match_summary,
    trajectory_stats,
    tensor_to_list,
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
    parser.add_argument("--action_loss_weight", type=float, default=6.0)
    parser.add_argument("--future_bev_loss_weight", type=float, default=1.0)
    parser.add_argument("--supervise_action_only", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--save_debug_artifacts", action="store_true")
    return parser.parse_args()


def _build_train_subset(dataset: Any, max_train_samples: int) -> Subset:
    """限制 overfit 用的训练子集大小。"""
    if max_train_samples <= 0:
        raise ValueError(f"--max_train_samples 必须大于 0，当前收到 {max_train_samples}")
    subset_size = min(len(dataset), max_train_samples)
    if subset_size <= 0:
        raise RuntimeError("构建 overfit 子集时没有可用样本。")
    return Subset(dataset, list(range(subset_size)))


def _stack_predicted_future_bevs(predicted_future_bevs: list[torch.Tensor]) -> torch.Tensor:
    """把 rollout 返回的 future BEV 列表堆叠成 [F, 1, H, W]。"""
    return torch.stack([frame.detach().cpu().to(torch.float32) for frame in predicted_future_bevs], dim=0)


def _save_debug_artifacts(
    checkpoint_dir: Path,
    reference_sample: Any,
    gt_action_tokens: list[int],
    predicted_action_tokens: list[int],
    gt_quantized_trajectory: torch.Tensor,
    predicted_quantized_trajectory: torch.Tensor,
    gt_raw_trajectory: torch.Tensor,
    predicted_raw_trajectory: torch.Tensor,
    gt_future_bevs: torch.Tensor,
    predicted_future_bevs: torch.Tensor,
) -> Path:
    """把关键张量保存成单个 `.pt` 调试包。"""
    debug_dir = checkpoint_dir / "debug_artifacts"
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_path = debug_dir / "reference_sample_debug.pt"
    torch.save(
        {
            "metadata": reference_sample.metadata,
            "navigation_text": reference_sample.navigation_text,
            "front_image": reference_sample.front_image.detach().cpu(),
            "bev_now": reference_sample.bev_now.detach().cpu(),
            "future_actions": reference_sample.future_actions.detach().cpu(),
            "gt_action_tokens": gt_action_tokens,
            "predicted_action_tokens": predicted_action_tokens,
            "gt_quantized_trajectory": gt_quantized_trajectory.detach().cpu(),
            "predicted_quantized_trajectory": predicted_quantized_trajectory.detach().cpu(),
            "gt_raw_trajectory": gt_raw_trajectory.detach().cpu(),
            "predicted_raw_trajectory": predicted_raw_trajectory.detach().cpu(),
            "gt_future_bevs": gt_future_bevs.detach().cpu(),
            "predicted_future_bevs": predicted_future_bevs.detach().cpu(),
        },
        debug_path,
    )
    return debug_path


def main() -> None:
    """运行最小 overfit 验证。"""
    args = parse_args()
    config = build_default_config(PROJECT_ROOT)
    config.train.num_epochs = args.num_epochs
    config.train.output_dir = args.output_dir
    config.train.action_loss_weight = args.action_loss_weight
    config.train.future_bev_loss_weight = args.future_bev_loss_weight
    config.train.supervise_action_only = args.supervise_action_only
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
    print(
        f"[train] action_loss_weight={config.train.action_loss_weight} "
        f"future_bev_loss_weight={config.train.future_bev_loss_weight} "
        f"supervise_action_only={config.train.supervise_action_only}"
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
    gt_action_tokens = [int(token_id) for token_id in model.discretizer.encode_future_actions(reference_sample.future_actions)]
    gt_quantized_trajectory = model.discretizer.decode_action_tokens_to_trajectory(gt_action_tokens)
    gt_raw_trajectory = torch.cumsum(reference_sample.future_actions.detach().cpu().to(torch.float32), dim=0)
    gt_future_bevs = reference_sample.future_bevs.detach().cpu().to(torch.float32)

    rollout_output = greedy_rollout(model, reference_sample, device)
    predicted_action_tokens = [int(token_id) for token_id in rollout_output["predicted_action_tokens"]]
    predicted_quantized_trajectory = rollout_output["predicted_trajectory"].detach().cpu().to(torch.float32)
    predicted_raw_trajectory = predicted_quantized_trajectory.clone()
    predicted_future_bevs = _stack_predicted_future_bevs(rollout_output["predicted_future_bevs"])

    match = token_match_summary(gt_action_tokens, predicted_action_tokens)
    quantized_stats = trajectory_stats(predicted_quantized_trajectory, gt_quantized_trajectory)
    raw_stats = trajectory_stats(predicted_raw_trajectory, gt_raw_trajectory)
    future_bev_stats = future_bev_difference_summary(predicted_future_bevs, gt_future_bevs)

    summary = {
        "dataset_type": args.dataset_type,
        "raw_dataset_size": len(full_dataset),
        "train_subset_size": len(train_dataset),
        "epoch_losses": epoch_losses,
        "action_loss_weight": config.train.action_loss_weight,
        "future_bev_loss_weight": config.train.future_bev_loss_weight,
        "supervise_action_only": config.train.supervise_action_only,
        "reference_metadata": reference_sample.metadata,
        "gt_action_tokens": gt_action_tokens,
        "predicted_action_tokens": predicted_action_tokens,
        "token_match_count": match["token_match_count"],
        "token_match_ratio": match["token_match_ratio"],
        "exact_sequence_match": match["exact_sequence_match"],
        "per_position_token_correctness": match["per_position_token_correctness"],
        "gt_quantized_trajectory": tensor_to_list(gt_quantized_trajectory),
        "predicted_quantized_trajectory": tensor_to_list(predicted_quantized_trajectory),
        "gt_raw_trajectory": tensor_to_list(gt_raw_trajectory),
        "predicted_raw_trajectory": tensor_to_list(predicted_raw_trajectory),
        "predicted_raw_trajectory_source": "decoded_action_tokens",
        "quantized_trajectory_stats": quantized_stats,
        "raw_trajectory_stats": raw_stats,
        "future_bev_difference_summary": future_bev_stats,
    }

    debug_artifact_path: str | None = None
    if args.save_debug_artifacts:
        debug_path = _save_debug_artifacts(
            checkpoint_dir=checkpoint_dir,
            reference_sample=reference_sample,
            gt_action_tokens=gt_action_tokens,
            predicted_action_tokens=predicted_action_tokens,
            gt_quantized_trajectory=gt_quantized_trajectory,
            predicted_quantized_trajectory=predicted_quantized_trajectory,
            gt_raw_trajectory=gt_raw_trajectory,
            predicted_raw_trajectory=predicted_raw_trajectory,
            gt_future_bevs=gt_future_bevs,
            predicted_future_bevs=predicted_future_bevs,
        )
        debug_artifact_path = str(debug_path)
        summary["debug_artifact_path"] = debug_artifact_path

    summary_path = checkpoint_dir / "overfit_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("[report] gt_action_tokens=", gt_action_tokens)
    print("[report] predicted_action_tokens=", predicted_action_tokens)
    print(
        f"[report] token_match_count={match['token_match_count']} "
        f"token_match_ratio={match['token_match_ratio']:.3f} "
        f"exact_sequence_match={match['exact_sequence_match']}"
    )
    print("[report] per_position_token_correctness=", match["per_position_token_correctness"])
    print("[report] quantized_trajectory_stats=", quantized_stats)
    print("[report] raw_trajectory_stats=", raw_stats)
    print("[report] future_bev_difference_summary=", future_bev_stats)
    if debug_artifact_path is not None:
        print(f"[report] debug_artifact_path={debug_artifact_path}")
    print(f"[report] summary_json={summary_path}")


if __name__ == "__main__":
    main()
