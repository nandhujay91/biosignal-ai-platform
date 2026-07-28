from __future__ import annotations

import os

import torch

from src.classifier.classifier import EmbeddingClassifier


class RetrainingPipeline:
    """
    Automatic model retraining pipeline.

    Flow:

    Drift detected
          |
          ↓
    Retrain classifier
          |
          ↓
    Save new model version
    """

    def __init__(
        self,
        output_path="artifacts/model/v2/classifier.pt",
    ):

        self.output_path = output_path

    def retrain(
        self,
        train_features,
        train_labels,
    ):

        print("Starting retraining...")

        model = EmbeddingClassifier(
            input_dim=131,
            num_classes=3,
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
        )

        criterion = torch.nn.CrossEntropyLoss()

        model.train()

        X = torch.tensor(
            train_features,
            dtype=torch.float32,
        )

        y = torch.tensor(
            train_labels,
            dtype=torch.long,
        )

        for epoch in range(5):

            optimizer.zero_grad()

            outputs = model(X)

            loss = criterion(
                outputs,
                y,
            )

            loss.backward()

            optimizer.step()

            print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

        os.makedirs(
            os.path.dirname(self.output_path),
            exist_ok=True,
        )

        torch.save(
            {"model_state_dict": model.state_dict()},
            self.output_path,
        )

        print("New model saved:", self.output_path)

        return self.output_path
