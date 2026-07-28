from __future__ import annotations

import torch


class TemporalMasking:
    """
    Apply temporal masking to multivariate time-series.

    Input:
        (batch, length, channels)

    Output:
        (batch, length, channels)
    """

    @staticmethod
    def random_mask(
        x: torch.Tensor,
        mask_ratio: float = 0.2,
    ) -> torch.Tensor:

        if not 0 <= mask_ratio < 1:

            raise ValueError(
                "mask_ratio must be between 0 and 1."
            )

        masked = x.clone()

        batch, length, _ = x.shape

        num_mask = int(length * mask_ratio)

        if num_mask == 0:
            return masked

        for b in range(batch):

            indices = torch.randperm(
                length,
                device=x.device,
            )[:num_mask]

            masked[b, indices, :] = 0.0

        return masked