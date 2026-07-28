from __future__ import annotations

import numpy as np
from torch.utils.data import DataLoader, Subset

from src.classifier.classifier import EmbeddingClassifier
from src.classifier.fusion_dataset import FusionDataset
from src.classifier.trainer import ClassifierTrainer
from src.logger import logger


def stratified_split(
    labels: np.ndarray,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[int], list[int]]:

    rng = np.random.default_rng(seed)

    train_indices: list[int] = []
    validation_indices: list[int] = []

    for cls in np.unique(labels):

        class_indices = np.where(labels == cls)[0]

        rng.shuffle(class_indices)

        split = int(len(class_indices) * train_ratio)

        train_indices.extend(class_indices[:split].tolist())

        validation_indices.extend(class_indices[split:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)

    return train_indices, validation_indices


class TrainingPipeline:
    """
    Complete fusion classifier training pipeline.
    """

    def run(self) -> None:

        logger.info("Starting classifier training pipeline.")

        dataset = FusionDataset(
            embeddings_path="artifacts/embeddings/embeddings.npy",
            features_path="artifacts/datasets/v1/Ephy/features.npy",
            labels_path="artifacts/labels/labels.npy",
        )

        labels = np.load("artifacts/labels/labels.npy")

        train_indices, validation_indices = stratified_split(labels)

        train_dataset = Subset(
            dataset,
            train_indices,
        )

        validation_dataset = Subset(
            dataset,
            validation_indices,
        )

        print(
            "Total Dataset Size:",
            len(dataset),
        )

        print(
            "Train Size:",
            len(train_dataset),
        )

        print(
            "Validation Size:",
            len(validation_dataset),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
        )

        validation_loader = DataLoader(
            validation_dataset,
            batch_size=32,
            shuffle=False,
        )

        model = EmbeddingClassifier(
            input_dim=131,
            num_classes=3,
        )

        class_weights = [
            3.43,  # Normal
            1.16,  # Alert
            0.54,  # Critical
        ]

        trainer = ClassifierTrainer(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            learning_rate=1e-3,
            epochs=10,
            class_weights=class_weights,
        )

        trainer.train()

        logger.info("Classifier training completed successfully.")


if __name__ == "__main__":

    pipeline = TrainingPipeline()

    pipeline.run()
