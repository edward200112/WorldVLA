"""官方最小训练入口。

当前仓库默认只维护 `scripts/` + `unitok_drive_lite/` 这条主链路。
如果后续继续迁移 Emu3 或调整 unified-token 训练逻辑，应优先修改这里及其包内依赖。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unitok_drive_lite import (
    ToyUnifiedDriveDataset,
    UnifiedDriveCollator,
    UnifiedDriveModel,
    build_default_config,
)
from unitok_drive_lite.train_utils import (
    build_optimizer,
    greedy_rollout,
    save_experiment_artifacts,
    seed_everything,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    """解析最小训练脚本参数。"""
    parser = argparse.ArgumentParser(description="训练最小版 UniTok-Drive-Lite Emu3 主干。")
    parser.add_argument("--dataset_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="outputs/unitok_drive_lite")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    return parser.parse_args()


def main() -> None:
    """运行最小训练闭环。"""
    args = parse_args()
    project_root = PROJECT_ROOT
    config = build_default_config(project_root)
    config.train.dataset_size = args.dataset_size
    config.train.num_epochs = args.num_epochs
    config.train.output_dir = args.output_dir
    config.model.load_in_4bit = args.load_in_4bit
    config.model.gradient_checkpointing = not args.no_gradient_checkpointing

    seed_everything(config.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UnifiedDriveModel(config).to(device)
    print(f"[model] backbone={config.model.model_name}")
    print(
        f"[model] load_in_4bit={config.model.load_in_4bit} "
        f"gradient_checkpointing={config.model.gradient_checkpointing}"
    )
    total_parameters, trainable_parameters = model.count_trainable_parameters()
    print(f"[model] total_parameters={total_parameters:,}")
    print(f"[model] trainable_parameters={trainable_parameters:,}")

    dataset = ToyUnifiedDriveDataset(
        size=config.train.dataset_size,
        token_config=config.tokens,
        seed=config.train.seed,
    )
    collator = UnifiedDriveCollator(
        discretizer=model.discretizer,
        pad_token_id=model.tokenizer.pad_token_id,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        collate_fn=collator,
    )
    optimizer = build_optimizer(model, config)

    for epoch in range(config.train.num_epochs):
        average_loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            max_grad_norm=config.train.max_grad_norm,
            log_every=config.train.log_every,
        )
        print(f"[train] epoch={epoch + 1} average_loss={average_loss:.4f}")

    checkpoint_dir = project_root / config.train.output_dir / "checkpoint_last"
    save_experiment_artifacts(model, config, checkpoint_dir)
    print(f"[save] checkpoint_dir={checkpoint_dir}")

    demo_output = greedy_rollout(model, dataset[0], device)
    print("[demo] predicted_action_tokens=", demo_output["predicted_action_tokens"])
    print("[demo] predicted_trajectory_shape=", tuple(demo_output["predicted_trajectory"].shape))
    print("[demo] predicted_future_bev_frames=", len(demo_output["predicted_future_bevs"]))


if __name__ == "__main__":
    main()
