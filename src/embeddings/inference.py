from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoint import CheckpointManager
from .ts2vec import TS2Vec


class TS2VecInference:
    """
    Production inference pipeline for TS2Vec.

    Responsibilities
    ----------------
    - Load trained checkpoint
    - Generate embeddings
    - Encode batches
    - Encode complete datasets
    - Save embeddings
    """

    def __init__(
        self,
        model: TS2Vec,
        checkpoint_dir: str | Path = "artifacts/checkpoints",
        checkpoint_name: str = "ts2vec_best.pt",
        device: str | torch.device = "cpu",
    ) -> None:

        self.device = torch.device(device)

        self.model = model.to(self.device)

        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=checkpoint_name,
        )

    def load_model(self) -> None:
        """
        Load trained checkpoint.
        """

        self.checkpoint_manager.load(
            model=self.model,
            optimizer=None,
            scheduler=None,
        )

        self.model.eval()

        print("TS2Vec model loaded successfully.")

    @torch.no_grad()
    def encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate embeddings for one batch.
        """

        x = x.to(self.device)

        embeddings = self.model.encode(x)

        return embeddings.cpu()

    @torch.no_grad()
    def encode_dataset(
        self,
        dataloader: DataLoader,
    ) -> np.ndarray:
        """
        Generate embeddings for an entire dataset.
        """

        embeddings = []

        self.model.eval()

        for batch in dataloader:

            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch

            batch_embeddings = self.encode(x)

            embeddings.append(batch_embeddings)

        embeddings = torch.cat(
            embeddings,
            dim=0,
        )

        return embeddings.numpy()

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        output_path: str | Path,
    ) -> None:
        """
        Save embeddings as a NumPy file.
        """

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            output_path,
            embeddings,
        )

        print(f"Embeddings saved to: {output_path}")