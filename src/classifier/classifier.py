from __future__ import annotations

import torch
from torch import nn


class EmbeddingClassifier(nn.Module):
    """
    Multi-layer perceptron classifier for TS2Vec embeddings.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 64,
        num_classes: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(x)
