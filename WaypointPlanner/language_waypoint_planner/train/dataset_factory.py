"""Dataset and dataloader construction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from torch.utils.data import ConcatDataset, DataLoader, Dataset

from language_waypoint_planner.configs.schema import DataConfig, DatasetSourceConfig, ModelConfig
from language_waypoint_planner.data import HashTextTokenizer, build_dataset, build_multitask_collate_fn


@dataclass
class DatasetDimensions:
    """Shared input/output dimensions inferred from dataset configs."""

    image_size: Tuple[int, int]
    temporal_window: int
    num_cameras: int
    history_steps: int
    future_steps: int
    cameras: List[str]


def infer_dataset_dimensions(sources: Sequence[DatasetSourceConfig]) -> DatasetDimensions:
    """Ensure all dataset sources are shape-compatible for a single model."""

    if not sources:
        raise ValueError("At least one dataset source is required")
    reference = sources[0]
    for source in sources[1:]:
        comparable = (
            source.image_size == reference.image_size,
            source.temporal_window == reference.temporal_window,
            source.history_steps == reference.history_steps,
            source.future_steps == reference.future_steps,
            source.cameras == reference.cameras,
        )
        if not all(comparable):
            raise ValueError(
                "All dataset sources must share image_size, temporal_window, history_steps, future_steps, and cameras"
            )
    return DatasetDimensions(
        image_size=reference.image_size,
        temporal_window=reference.temporal_window,
        num_cameras=len(reference.cameras),
        history_steps=reference.history_steps,
        future_steps=reference.future_steps,
        cameras=list(reference.cameras),
    )


def _build_dataset_collection(sources: Sequence[DatasetSourceConfig]) -> Dataset:
    datasets = [build_dataset(source) for source in sources]
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def build_dataloaders(
    data_config: DataConfig,
    model_config: ModelConfig,
) -> Tuple[DataLoader, DataLoader | None, DatasetDimensions, HashTextTokenizer]:
    """Build train/validation dataloaders and the rationale tokenizer used by collate."""

    dimensions = infer_dataset_dimensions(data_config.train_sources or data_config.val_sources)
    rationale_tokenizer = HashTextTokenizer(vocab_size=model_config.rationale_vocab_size)
    collate_fn = build_multitask_collate_fn(
        tokenizer=rationale_tokenizer,
        rationale_max_length=model_config.rationale_max_length,
    )
    train_dataset = _build_dataset_collection(data_config.train_sources)
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config.batch_size,
        shuffle=data_config.shuffle,
        num_workers=data_config.num_workers,
        collate_fn=collate_fn,
    )

    val_loader = None
    if data_config.val_sources:
        val_dataset = _build_dataset_collection(data_config.val_sources)
        val_loader = DataLoader(
            val_dataset,
            batch_size=data_config.batch_size,
            shuffle=False,
            num_workers=data_config.num_workers,
            collate_fn=collate_fn,
        )
    return train_loader, val_loader, dimensions, rationale_tokenizer
