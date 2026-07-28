from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .classifier import EmbeddingClassifier


class ClassifierTrainer:
    """
    Trainer for embedding classifier with
    class imbalance handling.
    """

    def __init__(
        self,
        model: EmbeddingClassifier,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        learning_rate: float = 1e-3,
        epochs: int = 20,
        checkpoint_dir: str = "artifacts/classifier",
        device: str = "cpu",
        class_weights: list[float] | None = None,
    ) -> None:

        self.device = torch.device(device)

        self.model = model.to(
            self.device
        )

        self.train_loader = train_loader
        self.validation_loader = validation_loader

        self.epochs = epochs


        if class_weights is not None:

            weights = torch.tensor(
                class_weights,
                dtype=torch.float32,
            ).to(self.device)

            self.criterion = nn.CrossEntropyLoss(
                weight=weights
            )

        else:

            self.criterion = nn.CrossEntropyLoss()


        self.optimizer = Adam(
            self.model.parameters(),
            lr=learning_rate,
        )


        self.best_loss = float("inf")


        self.checkpoint_dir = Path(
            checkpoint_dir
        )

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        self.checkpoint_path = (
            self.checkpoint_dir
            / "classifier_best.pt"
        )


    def train_epoch(self) -> float:

        self.model.train()

        total_loss = 0.0


        for embeddings, labels in self.train_loader:

            embeddings = embeddings.to(
                self.device
            )

            labels = labels.to(
                self.device
            )


            self.optimizer.zero_grad()


            outputs = self.model(
                embeddings
            )


            loss = self.criterion(
                outputs,
                labels,
            )


            loss.backward()

            self.optimizer.step()


            total_loss += loss.item()


        return (
            total_loss /
            len(self.train_loader)
        )


    @torch.no_grad()
    def validate(self) -> float:

        self.model.eval()

        total_loss = 0.0


        for embeddings, labels in self.validation_loader:

            embeddings = embeddings.to(
                self.device
            )

            labels = labels.to(
                self.device
            )


            outputs = self.model(
                embeddings
            )


            loss = self.criterion(
                outputs,
                labels,
            )


            total_loss += loss.item()


        return (
            total_loss /
            len(self.validation_loader)
        )


    def save_checkpoint(self) -> None:

        torch.save(
            {
                "model_state_dict":
                    self.model.state_dict(),

                "best_loss":
                    self.best_loss,
            },
            self.checkpoint_path,
        )


    def train(self) -> None:

        for epoch in range(
            self.epochs
        ):

            train_loss = (
                self.train_epoch()
            )

            validation_loss = (
                self.validate()
            )


            print(
                f"Epoch {epoch+1}/{self.epochs}"
            )

            print(
                f"Train Loss      : {train_loss:.4f}"
            )

            print(
                f"Validation Loss : {validation_loss:.4f}"
            )


            if validation_loss < self.best_loss:

                self.best_loss = validation_loss

                self.save_checkpoint()

                print(
                    "Best model saved."
                )