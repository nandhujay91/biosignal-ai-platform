from __future__ import annotations

import torch

from src.classifier.classifier import EmbeddingClassifier


def main() -> None:

    batch_size = 8
    embedding_dim = 128
    num_classes = 3

    model = EmbeddingClassifier(
        input_dim=embedding_dim,
        hidden_dim=64,
        num_classes=num_classes,
    )

    x = torch.randn(
        batch_size,
        embedding_dim,
    )

    outputs = model(x)

    print(f"Input Shape      : {x.shape}")
    print(f"Output Shape     : {outputs.shape}")

    assert outputs.shape == (
        batch_size,
        num_classes,
    )

    print("Classifier test passed successfully.")


if __name__ == "__main__":
    main()