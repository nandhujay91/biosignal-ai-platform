from __future__ import annotations

import random

import torch


class RandomTemporalCrop:
    """
    Random temporal cropping used by TS2Vec.

    Input:
        (batch, length, channels)

    Output:
        (batch, crop_length, channels)
    """

    @staticmethod
    def crop(
        x: torch.Tensor,
        min_crop_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Randomly crop each batch to the same temporal length.

        Args:
            x:
                Tensor of shape (batch, length, channels)

            min_crop_ratio:
                Minimum percentage of sequence retained.

        Returns:
            Cropped tensor.
        """

        if not 0 < min_crop_ratio <= 1:
            raise ValueError("min_crop_ratio must be between 0 and 1.")

        # Only length is required
        _, length, _ = x.shape

        min_crop_length = max(
            1,
            int(length * min_crop_ratio),
        )

        crop_length = random.randint(
            min_crop_length,
            length,
        )

        start = random.randint(
            0,
            length - crop_length,
        )

        end = start + crop_length

        return x[:, start:end, :]
