from __future__ import annotations

import torch

from src.embeddings import (
    CheckpointManager,
    LearningRateScheduler,
)


def main() -> None:

    model = torch.nn.Linear(
        10,
        2,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
    )

    scheduler = LearningRateScheduler(
        optimizer=optimizer,
        epochs=10,
    ).scheduler

    manager = CheckpointManager(
        checkpoint_dir="artifacts/checkpoints",
        checkpoint_name="linear_test.pt",
    )

    path = manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=5,
        best_loss=0.123,
    )

    print(f"Saved: {path}")

    epoch, best_loss = manager.load(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    print(f"Loaded Epoch: {epoch}")
    print(f"Loaded Best Loss: {best_loss}")


if __name__ == "__main__":
    main()