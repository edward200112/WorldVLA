"""Logging hooks for training and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List


class BaseLogger:
    """Base logger interface."""

    def log_metrics(self, split: str, step: int, metrics: Dict[str, float]) -> None:
        raise NotImplementedError

    def close(self) -> None:
        """Release logger resources."""


class StdoutLogger(BaseLogger):
    """Minimal stdout logger."""

    def log_metrics(self, split: str, step: int, metrics: Dict[str, float]) -> None:
        formatted = ", ".join(f"{key}={value:.4f}" for key, value in sorted(metrics.items()))
        print(f"[{split}] step={step}: {formatted}")


class TensorBoardLogger(BaseLogger):
    """Optional TensorBoard logger."""

    def __init__(self, log_dir: Path) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "TensorBoard logging requested, but tensorboard is not installed."
            ) from exc
        self.writer = SummaryWriter(log_dir=str(log_dir))

    def log_metrics(self, split: str, step: int, metrics: Dict[str, float]) -> None:
        for key, value in metrics.items():
            self.writer.add_scalar(f"{split}/{key}", value, global_step=step)

    def close(self) -> None:
        self.writer.close()


class LoggerCollection(BaseLogger):
    """Fan-out logger collection."""

    def __init__(self, loggers: Iterable[BaseLogger]) -> None:
        self.loggers: List[BaseLogger] = list(loggers)

    def log_metrics(self, split: str, step: int, metrics: Dict[str, float]) -> None:
        for logger in self.loggers:
            logger.log_metrics(split, step, metrics)

    def close(self) -> None:
        for logger in self.loggers:
            logger.close()
