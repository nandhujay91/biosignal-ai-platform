from src.classifier.fusion_dataset import FusionDataset


def main():

    dataset = FusionDataset(
        embeddings_path="artifacts/embeddings/embeddings.npy",
        features_path="artifacts/datasets/v1/Ephy/features.npy",
        labels_path="artifacts/labels/labels.npy",
    )

    print(
        "Dataset Size:",
        len(dataset),
    )

    x, y = dataset[0]

    print(
        "Input Shape:",
        x.shape,
    )

    print(
        "Label:",
        y,
    )

    assert x.shape[0] == 131

    print("Fusion dataset test passed.")


if __name__ == "__main__":
    main()
