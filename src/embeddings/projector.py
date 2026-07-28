from __future__ import annotations

import torch
from torch import nn


class ProjectionHead(nn.Module):
    """
    Projection head used after the TS2Vec encoder.

    Input:
        (batch, length, hidden_dim)

    Output:
        (batch, length, projection_dim)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        projection_dim: int = 128,
    ) -> None:
        super().__init__()

        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.projector(x)
