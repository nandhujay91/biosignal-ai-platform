from __future__ import annotations

import torch
import numpy as np
from torch.utils.data import Dataset


class FusionDataset(Dataset):
    """
    Dataset combining:

    - TS2Vec embeddings
    - Engineered biosignal features
    - Labels
    """

    def __init__(
        self,
        embeddings_path: str,
        features_path: str,
        labels_path: str,
    ) -> None:


        embeddings = np.load(
            embeddings_path
        )


        features = np.load(
            features_path
        )


        labels = np.load(
            labels_path
        )


        assert embeddings.shape[0] == features.shape[0]

        assert embeddings.shape[0] == labels.shape[0]


        # Combine
        self.x = np.concatenate(
            [
                embeddings,
                features,
            ],
            axis=1,
        )


        self.y = labels


    def __len__(self):

        return len(self.y)


    def __getitem__(
        self,
        index,
    ):

        x = torch.tensor(
            self.x[index],
            dtype=torch.float32,
        )


        y = torch.tensor(
            self.y[index],
            dtype=torch.long,
        )


        return x, y