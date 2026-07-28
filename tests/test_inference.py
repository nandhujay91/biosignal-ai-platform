from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.embeddings import TS2Vec
from src.embeddings import TS2VecInference


def main() -> None:

    x = torch.randn(
        16,
        320,
        8,
    )

    dataset = TensorDataset(x)

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
    )

    model = TS2Vec(
        input_dim=8,
        hidden_dim=128,
        projection_dim=64,
        depth=8,
    )

    inference = TS2VecInference(
        model=model,
        checkpoint_dir="artifacts/checkpoints",
        device="cpu",
    )

    try:
        inference.load_model()
    except FileNotFoundError:
        print("No trained checkpoint found. Using randomly initialized model.")

    embeddings = inference.encode_dataset(
        dataloader=dataloader,
    )

    print("Embedding Shape :", embeddings.shape)

    inference.save_embeddings(
        embeddings=embeddings,
        output_path="artifacts/embeddings/embeddings.npy",
    )


if __name__ == "__main__":
    main()