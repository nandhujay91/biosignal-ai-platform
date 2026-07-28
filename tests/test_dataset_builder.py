from src.dataset import DatasetBuilder
from src.pipeline import SignalPreprocessor
from src.windowing import WindowGenerator



def test_dataset_builder():

    print(
        "\nStarting dataset builder test..."
    )


    # Step 1: Preprocess signals

    processed = SignalPreprocessor.run(
        "data/test"
    )


    # Step 2: Generate windows

    windows = WindowGenerator.generate(
        processed
    )


    # Step 3: Build datasets

    datasets, metadata = DatasetBuilder.build(
        signals=processed,
        windows=windows,
    )


    assert datasets is not None
    assert metadata is not None


    print()

    print("=" * 60)
    print("Datasets")
    print("=" * 60)



    for name, dataset in datasets.items():

        print()

        print(name)

        print(
            f"Shape : {dataset.shape}"
        )

        print(
            f"Dtype : {dataset.dtype}"
        )


        assert dataset.dtype == "float32"



    print()

    print("=" * 60)
    print("Metadata")
    print("=" * 60)



    for name, meta in metadata.items():

        print()

        print(name)

        print(meta)



    print()

    print(
        "Dataset builder test completed successfully."
    )



if __name__ == "__main__":

    test_dataset_builder()
