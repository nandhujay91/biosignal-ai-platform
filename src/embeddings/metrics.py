from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TrainingMetrics:
    """
    Stores training statistics for one epoch.
    """

    epoch: int
    train_loss: float
    validation_loss: float | None = None
    learning_rate: float | None = None


class MetricTracker:
    """
    Tracks training and validation losses.
    """

    def __init__(self) -> None:

        self.reset()

    def reset(self) -> None:

        self.train_loss = 0.0
        self.validation_loss = 0.0

        self.train_batches = 0
        self.validation_batches = 0

    def update_train(
        self,
        loss: torch.Tensor,
    ) -> None:

        self.train_loss += loss.item()
        self.train_batches += 1

    def update_validation(
        self,
        loss: torch.Tensor,
    ) -> None:

        self.validation_loss += loss.item()
        self.validation_batches += 1

    @property
    def average_train_loss(self) -> float:

        if self.train_batches == 0:
            return 0.0

        return self.train_loss / self.train_batches

    @property
    def average_validation_loss(self) -> float:

        if self.validation_batches == 0:
            return 0.0

        return self.validation_loss / self.validation_batches