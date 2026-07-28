from __future__ import annotations

import torch

from src.classifier.metrics import MetricsCalculator


def main() -> None:

    labels = torch.tensor(
        [
            0, 1, 2,
            0, 1, 2,
            0, 1, 2,
            0,
        ]
    )

    predictions = torch.tensor(
        [
            0, 1, 2,
            0, 2, 2,
            1, 1, 2,
            0,
        ]
    )

    metrics = MetricsCalculator.calculate(
        predictions=predictions,
        labels=labels,
    )

    print(f"Accuracy : {metrics.accuracy:.4f}")
    print(f"Precision: {metrics.precision:.4f}")
    print(f"Recall   : {metrics.recall:.4f}")
    print(f"F1 Score : {metrics.f1_score:.4f}")

    print(
        "\nClassifier metrics test passed successfully."
    )


if __name__ == "__main__":
    main()
