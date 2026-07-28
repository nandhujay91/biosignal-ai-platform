from __future__ import annotations

import numpy as np
from torch.utils.data import DataLoader, Subset

from src.classifier.classifier import EmbeddingClassifier
from src.classifier.fusion_dataset import FusionDataset
from src.classifier.trainer import ClassifierTrainer


def stratified_split(
    labels: np.ndarray,
    train_ratio: float = 0.8,
    seed: int = 42,
):

    rng = np.random.default_rng(seed)

    train_indices = []
    val_indices = []

    for cls in np.unique(labels):

        class_indices = np.where(labels == cls)[0]

        rng.shuffle(class_indices)

        split = int(len(class_indices) * train_ratio)

        train_indices.extend(class_indices[:split])

        val_indices.extend(class_indices[split:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return train_indices, val_indices


def main():

    dataset = FusionDataset(
        embeddings_path="artifacts/embeddings/embeddings.npy",
        features_path="artifacts/datasets/v1/Ephy/features.npy",
        labels_path="artifacts/labels/labels.npy",
    )

    labels = np.load("artifacts/labels/labels.npy")

    print(
        "Total Dataset Size:",
        len(dataset),
    )

    train_idx, val_idx = stratified_split(labels)

    train_dataset = Subset(
        dataset,
        train_idx,
    )

    val_dataset = Subset(
        dataset,
        val_idx,
    )

    print(
        "Train Size:",
        len(train_dataset),
    )

    print(
        "Validation Size:",
        len(val_dataset),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
    )

    validation_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
    )

    model = EmbeddingClassifier(
        input_dim=131,
        num_classes=3,
    )

    class_weights = [
        3.43,
        1.16,
        0.54,
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

    print("Fusion classifier training completed successfully.")


if __name__ == "__main__":
    main()
