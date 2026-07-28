from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.embeddings.ts2vec import TS2Vec


class EmbeddingGenerator:
    """
    Generate TS2Vec embeddings from biosignal windows.
    """

    def __init__(
        self,
        checkpoint_path: str | Path = "artifacts/checkpoints/ts2vec_best.pt",
        output_path: str | Path = "artifacts/embeddings/embeddings.npy",
        device: str = "cpu",
    ) -> None:

        self.device = torch.device(device)

        self.checkpoint_path = Path(checkpoint_path)

        self.output_path = Path(output_path)

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def load_model(self) -> TS2Vec:

        model = TS2Vec(
            input_dim=8,
            hidden_dim=128,
            projection_dim=64,
            depth=8,
        )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        model.to(self.device)

        model.eval()

        return model

    @torch.no_grad()
    def generate(
        self,
        windows_path: str | Path,
    ) -> np.ndarray:

        windows = np.load(windows_path)

        model = self.load_model()

        x = torch.tensor(
            windows,
            dtype=torch.float32,
        ).to(self.device)

        embeddings = model.encoder(x)

        embeddings = embeddings.mean(dim=1)

        embeddings = embeddings.cpu().numpy()

        np.save(
            self.output_path,
            embeddings,
        )

        return embeddings
