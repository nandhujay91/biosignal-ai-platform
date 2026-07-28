from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.embeddings import TS2Vec
from src.embeddings import TS2VecTrainer


def main() -> None:

    x = torch.randn(
        32,
        320,
        8,
    )

    dataset = TensorDataset(x)

    train_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
    )

    validation_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
    )

    model = TS2Vec(
        input_dim=8,
        hidden_dim=128,
        projection_dim=64,
        depth=8,
    )

    trainer = TS2VecTrainer(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        learning_rate=1e-3,
        epochs=2,
        device="cpu",
        checkpoint_dir="artifacts/checkpoints",
    )

    trainer.train()


if __name__ == "__main__":
    main()