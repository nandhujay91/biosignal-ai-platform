from src.dataset import DatasetBuilder, DatasetSaver
from src.pipeline import SignalPreprocessor
from src.windowing import WindowGenerator


def test_dataset_saver():

    processed = SignalPreprocessor.run("data/test")

    windows = WindowGenerator.generate(processed)

    datasets, metadata = DatasetBuilder.build(
        signals=processed,
        windows=windows,
    )

    saved = DatasetSaver.save(
        datasets=datasets,
        metadata=metadata,
    )

    print()

    print("=" * 60)
    print("Saved Datasets")
    print("=" * 60)

    for name, path in saved.items():

        print(f"{name} -> {path}")


if __name__ == "__main__":
    test_dataset_saver()
