from src.dataset import DatasetLoader


def test_dataset_loader():

    datasets, metadata = DatasetLoader.load()

    print()

    print("=" * 60)
    print("Loaded Datasets")
    print("=" * 60)

    for name, dataset in datasets.items():

        print()

        print(f"{name}")

        print(f"Shape : {dataset.shape}")

        print(metadata[name])


if __name__ == "__main__":
    test_dataset_loader()
