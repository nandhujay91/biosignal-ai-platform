from __future__ import annotations

import os

import numpy as np

from src.retraining.retrain_pipeline import RetrainingPipeline

FEATURE_COUNT = 131
NUM_CLASSES = 3


def create_retraining_data():
    """
    Simulated new production data after drift.

    Features:
        128 TS2Vec embeddings
        3 signal features

    Shape:
        samples x 131
    """

    np.random.seed(42)

    X = np.random.normal(loc=0.5, scale=1.2, size=(300, FEATURE_COUNT))

    y = np.random.randint(0, NUM_CLASSES, size=300)

    return X, y


def main():

    print("Preparing retraining data...")

    train_features, train_labels = create_retraining_data()

    print("Feature shape:", train_features.shape)

    print("Label shape:", train_labels.shape)

    print("\nStarting retraining pipeline...")

    pipeline = RetrainingPipeline(output_path="artifacts/model/v2/classifier.pt")

    model_path = pipeline.retrain(
        train_features,
        train_labels,
    )

    print("\nRetraining completed.")

    print("New model:", model_path)

    print("Model exists:", os.path.exists(model_path))


if __name__ == "__main__":

    main()
