from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """
    PyTorch dataset for TS2Vec embeddings.

    Loads:
        embeddings.npy
        labels.npy

    Returns:
        embedding vector (128,)
        label (0/1/2)
    """

    def __init__(
        self,
        embeddings_path: str | Path,
        labels_path: str | Path,
    ) -> None:

        self.embeddings = np.load(
            embeddings_path
        )

        self.labels = np.load(
            labels_path
        )


        if len(self.embeddings) != len(self.labels):
            raise ValueError(
                "Embeddings and labels size mismatch"
            )


    def __len__(self) -> int:

        return len(self.labels)


    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        embedding = torch.tensor(
            self.embeddings[index],
            dtype=torch.float32,
        )

        label = torch.tensor(
            self.labels[index],
            dtype=torch.long,
        )

        return embedding, label