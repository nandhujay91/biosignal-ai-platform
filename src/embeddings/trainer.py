from __future__ import annotations

from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .checkpoint import CheckpointManager
from .metrics import MetricTracker
from .scheduler import LearningRateScheduler
from .ts2vec import TS2Vec


class TS2VecTrainer:
    """
    Production trainer for TS2Vec.

    Responsibilities
    ----------------
    - Training
    - Validation
    - Metric tracking
    - Learning-rate scheduling
    - Checkpoint management
    """

    def __init__(
        self,
        model: TS2Vec,
        train_loader: DataLoader,
        validation_loader: DataLoader | None = None,
        learning_rate: float = 1e-3,
        epochs: int = 100,
        device: str | torch.device = "cpu",
        checkpoint_dir: str | Path = "artifacts/checkpoints",
    ) -> None:

        self.device = torch.device(device)

        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.validation_loader = validation_loader

        self.epochs = epochs

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
        )

        self.scheduler = LearningRateScheduler(
            optimizer=self.optimizer,
            epochs=epochs,
        )

        self.metrics = MetricTracker()

        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            checkpoint_name="ts2vec_best.pt",
        )

        self.best_loss = float("inf")
        self.start_epoch = 0

    def train_epoch(
        self,
    ) -> float:
        """
        Train one epoch.
        """

        self.model.train()

        self.metrics.reset()

        for batch in self.train_loader:

            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch

            x = x.to(self.device)

            self.optimizer.zero_grad()

            loss = self.model(x)

            loss.backward()

            self.optimizer.step()

            self.metrics.update_train(loss)

        self.scheduler.step()

        return self.metrics.average_train_loss

    @torch.no_grad()
    def validate(
        self,
    ) -> float:
        """
        Validate one epoch.
        """

        if self.validation_loader is None:
            return 0.0

        self.model.eval()

        # Reset only validation statistics
        self.metrics.validation_loss = 0.0
        self.metrics.validation_batches = 0

        for batch in self.validation_loader:

            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch

            x = x.to(self.device)

            loss = self.model(x)

            self.metrics.update_validation(loss)

        return self.metrics.average_validation_loss

    def save_best(
        self,
        epoch: int,
        validation_loss: float,
    ) -> None:
        """
        Save best checkpoint.
        """

        if validation_loss < self.best_loss:

            self.best_loss = validation_loss

            self.checkpoint_manager.save(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler.scheduler,
                epoch=epoch,
                best_loss=self.best_loss,
            )

    def load_checkpoint(
        self,
    ) -> None:
        """
        Resume training if checkpoint exists.
        """

        try:

            epoch, best_loss = self.checkpoint_manager.load(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler.scheduler,
            )

            self.start_epoch = epoch + 1
            self.best_loss = best_loss

            print(f"Resumed from epoch {epoch}")

        except FileNotFoundError:

            print("No checkpoint found. Starting fresh.")

    def train(
        self,
    ) -> None:
        """
        Execute full training.
        """

        self.load_checkpoint()

        for epoch in range(
            self.start_epoch,
            self.epochs,
        ):

            train_loss = self.train_epoch()

            validation_loss = self.validate()

            if self.validation_loader is not None:

                self.save_best(
                    epoch=epoch,
                    validation_loss=validation_loss,
                )

            print(
                f"Epoch {epoch + 1}/{self.epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Validation Loss: {validation_loss:.6f} | "
                f"LR: {self.scheduler.get_lr():.8f}"
            )
