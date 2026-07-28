from __future__ import annotations

import numpy as np

from src.embeddings.generate_embeddings import EmbeddingGenerator


def main():

    windows_path = (
        "artifacts/datasets/v1/Ephy/windows.npy"
    )


    generator = EmbeddingGenerator(
        checkpoint_path=(
            "artifacts/checkpoints/ts2vec_best.pt"
        ),
        output_path=(
            "artifacts/embeddings/embeddings.npy"
        ),
    )


    embeddings = generator.generate(
        windows_path=windows_path,
    )


    print(
        "Generated Embeddings Shape:",
        embeddings.shape,
    )


    # Validation checks

    assert len(embeddings.shape) == 2

    assert embeddings.shape[0] == 1132

    assert embeddings.shape[1] == 128


    # Verify saved file

    saved = np.load(
        "artifacts/embeddings/embeddings.npy"
    )


    print(
        "Saved Embeddings Shape:",
        saved.shape,
    )


    assert saved.shape == embeddings.shape


    print(
        "Embedding generation test passed successfully."
    )


if __name__ == "__main__":
    main()
