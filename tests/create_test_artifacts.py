from __future__ import annotations

import json
from pathlib import Path

import torch

from src.classifier.classifier import EmbeddingClassifier

ARTIFACT_DIR = Path("artifacts/classifier")


def create_test_artifacts() -> None:
    """
    Create synthetic classifier artifacts
    for CI testing.
    """

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = EmbeddingClassifier(
        input_dim=131,
        num_classes=3,
    )

    checkpoint = {"model_state_dict": model.state_dict()}

    torch.save(
        checkpoint,
        ARTIFACT_DIR / "classifier.pt",
    )

    torch.save(
        checkpoint,
        ARTIFACT_DIR / "classifier_best.pt",
    )

    class_mapping = {
        "0": "Normal",
        "1": "Alert",
        "2": "Critical",
    }

    with open(
        ARTIFACT_DIR / "class_mapping.json",
        "w",
    ) as file:
        json.dump(
            class_mapping,
            file,
            indent=4,
        )

    (ARTIFACT_DIR / "version.txt").write_text("v1.0.0")


if __name__ == "__main__":
    create_test_artifacts()
