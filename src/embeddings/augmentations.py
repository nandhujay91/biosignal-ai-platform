from __future__ import annotations

import torch

from .cropping import RandomTemporalCrop
from .masking import TemporalMasking


class TS2VecAugmentations:
    """
    Generate two augmented views for TS2Vec training.

    Input:
        (batch, length, channels)

    Returns:
        view1, view2
    """

    @staticmethod
    def generate_views(
        x: torch.Tensor,
        crop_ratio: float = 0.5,
        mask_ratio: float = 0.2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate two augmented views from the same temporal crop.

        Both views share the same crop length but receive
        independent temporal masking.
        """

        # Generate one random crop
        crop = RandomTemporalCrop.crop(
            x,
            min_crop_ratio=crop_ratio,
        )

        # First augmented view
        view1 = TemporalMasking.random_mask(
            crop,
            mask_ratio=mask_ratio,
        )

        # Second augmented view
        view2 = TemporalMasking.random_mask(
            crop,
            mask_ratio=mask_ratio,
        )

        return view1, view2