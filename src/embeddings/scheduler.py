from __future__ import annotations

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR


class LearningRateScheduler:
    """
    Wrapper around PyTorch learning rate schedulers.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        epochs: int,
        eta_min: float = 1e-6,
    ) -> None:

        self.scheduler = CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=eta_min,
        )

    def step(self) -> None:
        self.scheduler.step()

    def get_lr(self) -> float:
        return self.scheduler.get_last_lr()[0]