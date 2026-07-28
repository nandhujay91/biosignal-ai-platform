from __future__ import annotations

import torch
from torch import nn

from .augmentations import TS2VecAugmentations
from .encoder import TS2VecEncoder
from .losses import HierarchicalContrastiveLoss
from .projector import ProjectionHead


class TS2Vec(nn.Module):
    """
    TS2Vec self-supervised representation learning model.

    During training:
        Input
            ↓
        Augmentation
            ↓
        Encoder
            ↓
        Projection Head
            ↓
        Contrastive Loss

    During inference:
        Input
            ↓
        Encoder
            ↓
        Temporal Mean Pooling
            ↓
        Embedding
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        projection_dim: int = 128,
        depth: int = 8,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim

        self.augmentations = TS2VecAugmentations()

        self.encoder = TS2VecEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            depth=depth,
        )

        self.projector = ProjectionHead(
            hidden_dim=hidden_dim,
            projection_dim=projection_dim,
        )

        self.loss_fn = HierarchicalContrastiveLoss()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Training forward pass.

        Returns
        -------
        torch.Tensor
            Contrastive loss.
        """

        view1, view2 = self.augmentations.generate_views(x)

        z1 = self.encoder(view1)
        z2 = self.encoder(view2)

        p1 = self.projector(z1)
        p2 = self.projector(z2)

        loss = self.loss_fn(p1, p2)

        return loss

    @torch.no_grad()
    def encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate embeddings for inference.

        Parameters
        ----------
        x : torch.Tensor
            Shape:
                (batch, sequence_length, channels)

        Returns
        -------
        torch.Tensor
            Shape:
                (batch, hidden_dim)
        """

        self.eval()

        x = x.to(next(self.parameters()).device)

        features = self.encoder(x)

        embeddings = features.mean(dim=1)

        return embeddings
