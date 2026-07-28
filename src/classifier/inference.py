from __future__ import annotations

from pathlib import Path

import torch

from .classifier import EmbeddingClassifier


class ClassifierInference:
    """
    Production inference pipeline for classifier.

    Responsibilities
    ----------------
    - Load trained classifier checkpoint
    - Predict class from embeddings
    """


    def __init__(
        self,
        model: EmbeddingClassifier,
        checkpoint_path: str | Path = "artifacts/classifier/classifier_best.pt",
        device: str = "cpu",
    ) -> None:

        self.device = torch.device(device)

        self.model = model.to(
            self.device
        )

        self.checkpoint_path = Path(
            checkpoint_path
        )


    def load_model(self) -> None:
        """
        Load classifier weights.
        """

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}"
            )

        state_dict = torch.load(
            self.checkpoint_path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            state_dict
        )

        self.model.eval()

        print(
            "Classifier model loaded successfully."
        )


    @torch.no_grad()
    def predict(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict class labels.

        Input:
            (batch, embedding_dim)

        Output:
            (batch,)
        """

        embeddings = embeddings.to(
            self.device
        )

        logits = self.model(
            embeddings
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        return predictions.cpu()