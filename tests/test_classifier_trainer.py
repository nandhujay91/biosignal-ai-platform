from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from src.classifier.classifier import EmbeddingClassifier
from src.classifier.trainer import ClassifierTrainer


def main() -> None:

    torch.manual_seed(42)

    embeddings = torch.randn(
        100,
        128,
    )

    labels = torch.randint(
        low=0,
        high=3,
        size=(100,),
    )

    dataset = TensorDataset(
        embeddings,
        labels,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
    )

    validation_loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
    )

    model = EmbeddingClassifier(
        input_dim=128,
        hidden_dim=64,
        num_classes=3,
    )

    trainer = ClassifierTrainer(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        learning_rate=1e-3,
        epochs=3,
        device="cpu",
    )

    trainer.train()

    print("\nClassifier trainer test passed successfully.")


if __name__ == "__main__":
    main()