from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.classifier.classifier import EmbeddingClassifier
from src.classifier.fusion_dataset import FusionDataset
from src.classifier.metrics import MetricsCalculator


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

        split_point = int(len(class_indices) * train_ratio)

        train_indices.extend(class_indices[:split_point])

        val_indices.extend(class_indices[split_point:])

    rng.shuffle(train_indices)

    rng.shuffle(val_indices)

    return (
        train_indices,
        val_indices,
    )


def main():

    # Load fusion dataset
    dataset = FusionDataset(
        embeddings_path="artifacts/embeddings/embeddings.npy",
        features_path="artifacts/datasets/v1/Ephy/features.npy",
        labels_path="artifacts/labels/labels.npy",
    )

    print(
        "Dataset Size:",
        len(dataset),
    )

    labels = np.load("artifacts/labels/labels.npy")

    # Same stratified split as training
    _, validation_indices = stratified_split(
        labels,
        train_ratio=0.8,
        seed=42,
    )

    validation_dataset = Subset(
        dataset,
        validation_indices,
    )

    print(
        "Validation Size:",
        len(validation_dataset),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=32,
        shuffle=False,
    )

    # Fusion model
    model = EmbeddingClassifier(
        input_dim=131,
        num_classes=3,
    )

    checkpoint = torch.load(
        "artifacts/classifier/classifier_best.pt",
        map_location="cpu",
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print(
        "Loaded Best Loss:",
        checkpoint["best_loss"],
    )

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for features, labels in validation_loader:

            outputs = model(features)

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            all_predictions.extend(predictions.tolist())

            all_labels.extend(labels.tolist())

    predictions_tensor = torch.tensor(all_predictions)

    labels_tensor = torch.tensor(all_labels)

    metrics = MetricsCalculator.calculate(
        predictions_tensor,
        labels_tensor,
    )

    print("\nEvaluation Results")
    print("------------------")

    print(f"Accuracy  : {metrics.accuracy:.4f}")

    print(f"Precision : {metrics.precision:.4f}")

    print(f"Recall    : {metrics.recall:.4f}")

    print(f"F1 Score  : {metrics.f1_score:.4f}")

    print("\nConfusion Matrix")
    print("----------------")

    print(metrics.confusion_matrix)

    print("\nClass Wise Metrics")
    print("------------------")

    for class_name, values in metrics.class_report.items():

        print(f"\n{class_name}")

        print(f"Precision : {values['precision']:.4f}")

        print(f"Recall    : {values['recall']:.4f}")

        print(f"F1 Score  : {values['f1']:.4f}")

        print(f"Samples   : {values['support']}")

    print("\nFusion classifier evaluation completed successfully.")


if __name__ == "__main__":
    main()
