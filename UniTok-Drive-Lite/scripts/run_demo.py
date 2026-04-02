"""官方最小推理入口。

当前仓库默认只维护 `scripts/` + `unitok_drive_lite/` 这条主链路。
如果后续继续迁移 Emu3 或调整 unified-token 推理逻辑，应优先修改这里及其包内依赖。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unitok_drive_lite import ToyUnifiedDriveDataset, UnifiedDriveModel, build_default_config
from unitok_drive_lite.train_utils import greedy_rollout, seed_everything


def parse_args() -> argparse.Namespace:
    """解析 demo 推理脚本参数。"""
    parser = argparse.ArgumentParser(description="运行最小版 UniTok-Drive-Lite Emu3 推理演示。")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="outputs/unitok_drive_lite/checkpoint_last",
    )
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--load_in_4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    """加载 LoRA 检查点并运行一次贪心推理。"""
    args = parse_args()
    project_root = PROJECT_ROOT
    checkpoint_dir = project_root / args.checkpoint_dir
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"未找到检查点目录: {checkpoint_dir}")

    config = build_default_config(project_root)
    config.model.load_in_4bit = args.load_in_4bit
    seed_everything(config.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UnifiedDriveModel(config).to(device)
    model.load_checkpoint(checkpoint_dir)
    print(f"[model] backbone={config.model.model_name}")
    print(f"[model] load_in_4bit={config.model.load_in_4bit}")

    dataset = ToyUnifiedDriveDataset(
        size=max(args.sample_index + 1, 1),
        token_config=config.tokens,
        seed=config.train.seed,
    )
    sample = dataset[args.sample_index]
    output = greedy_rollout(model, sample, device)

    print("[input] navigation_text=", sample.navigation_text)
    print("[output] predicted_action_tokens=", output["predicted_action_tokens"])
    print("[output] predicted_trajectory=")
    print(output["predicted_trajectory"])
    print("[output] decoded_future_bev_shapes=")
    for frame in output["predicted_future_bevs"]:
        print(tuple(frame.shape))


if __name__ == "__main__":
    main()
