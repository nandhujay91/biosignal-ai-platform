from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.classifier.classifier import EmbeddingClassifier
from src.classifier.fusion_dataset import FusionDataset
from src.classifier.metrics import MetricsCalculator
from src.logger import logger


def stratified_split(
    labels: np.ndarray,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[int], list[int]]:

    rng = np.random.default_rng(seed)

    train_indices: list[int] = []
    val_indices: list[int] = []

    for cls in np.unique(labels):

        indices = np.where(labels == cls)[0]

        rng.shuffle(indices)

        split = int(len(indices) * train_ratio)

        train_indices.extend(indices[:split].tolist())

        val_indices.extend(indices[split:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return train_indices, val_indices


class EvaluationPipeline:
    """
    Evaluation pipeline for fusion classifier.
    """

    def __init__(self) -> None:

        self.report_path = Path("artifacts/reports/evaluation_metrics.json")

        self.report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def run(self) -> dict:

        logger.info("Starting evaluation pipeline.")

        dataset = FusionDataset(
            embeddings_path="artifacts/embeddings/embeddings.npy",
            features_path="artifacts/datasets/v1/Ephy/features.npy",
            labels_path="artifacts/labels/labels.npy",
        )

        labels = np.load("artifacts/labels/labels.npy")

        _, validation_indices = stratified_split(labels)

        validation_dataset = Subset(
            dataset,
            validation_indices,
        )

        loader = DataLoader(
            validation_dataset,
            batch_size=32,
            shuffle=False,
        )

        model = EmbeddingClassifier(
            input_dim=131,
            num_classes=3,
        )

        checkpoint = torch.load(
            "artifacts/classifier/classifier_best.pt",
            map_location="cpu",
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        model.eval()

        predictions: list[int] = []
        true_labels: list[int] = []

        with torch.no_grad():

            for x, y in loader:

                output = model(x)

                pred = torch.argmax(
                    output,
                    dim=1,
                )

                predictions.extend(pred.tolist())

                true_labels.extend(y.tolist())

        predictions_tensor = torch.tensor(predictions)

        true_labels_tensor = torch.tensor(true_labels)

        metrics = MetricsCalculator.calculate(
            predictions_tensor,
            true_labels_tensor,
        )

        report = {
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            "confusion_matrix": metrics.confusion_matrix,
            "class_report": metrics.class_report,
        }

        with open(
            self.report_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                report,
                file,
                indent=4,
            )

        logger.info(f"Evaluation report saved: {self.report_path}")

        print(
            json.dumps(
                report,
                indent=4,
            )
        )

        return report


if __name__ == "__main__":

    pipeline = EvaluationPipeline()

    pipeline.run()
