from __future__ import annotations

from pathlib import Path

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class CheckpointManager:
    """
    Save and load model checkpoints.

    Stores:
    - Model weights
    - Optimizer state
    - Scheduler state
    - Current epoch
    - Best validation loss
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        checkpoint_name: str = "ts2vec_best.pt",
    ) -> None:

        self.checkpoint_dir = Path(checkpoint_dir)

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.checkpoint_path = self.checkpoint_dir / checkpoint_name

    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        epoch: int,
        best_loss: float,
    ) -> Path:

        torch.save(
            {
                "epoch": epoch,
                "best_loss": best_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": (
                    scheduler.state_dict() if scheduler is not None else None
                ),
            },
            self.checkpoint_path,
        )

        return self.checkpoint_path

    def load(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
    ) -> tuple[int, float]:
        """
        Load a checkpoint.

        Training:
            Loads model, optimizer, scheduler.

        Inference:
            Loads only model weights.
        """

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return (
            checkpoint["epoch"],
            checkpoint["best_loss"],
        )
