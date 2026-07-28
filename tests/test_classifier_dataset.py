from __future__ import annotations

from torch.utils.data import DataLoader

from src.classifier.dataset import EmbeddingDataset


def main():

    dataset = EmbeddingDataset(
        embeddings_path="artifacts/embeddings/embeddings.npy",
        labels_path="artifacts/labels/labels.npy",
    )


    print("Dataset size:", len(dataset))


    embedding, label = dataset[0]


    print(
        "Embedding shape:",
        embedding.shape,
    )

    print(
        "Label:",
        label,
    )


    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
    )


    x, y = next(iter(loader))


    print(
        "Batch embeddings:",
        x.shape,
    )

    print(
        "Batch labels:",
        y.shape,
    )


    assert len(dataset) == 1132
    assert x.shape[1] == 128


    print(
        "Classifier dataset test passed successfully."
    )


if __name__ == "__main__":
    main()