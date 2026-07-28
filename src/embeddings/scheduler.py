from __future__ import annotations

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
        """
        Update learning rate.
        """

        self.scheduler.step()

    def get_lr(self) -> float:
        """
        Return current learning rate as float.
        """

        lr = self.scheduler.get_last_lr()[0]

        return float(lr)
