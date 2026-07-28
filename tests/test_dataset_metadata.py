from src.entities import DatasetMetadata


def test_dataset_metadata():

    metadata = DatasetMetadata(
        signal_name="Ephy",
        sampling_rate=256,
        channels=8,
        window_duration_seconds=5,
        window_size=1280,
        step_size=640,
        overlap=0.5,
        normalization="z_score",
        filter_type="butter_bandpass",
        dtype="float32",
    )

    print()

    print("=" * 60)
    print("Dataset Metadata")
    print("=" * 60)

    print(metadata)

    print()

    print("Dictionary Representation")

    print(metadata.to_dict())


if __name__ == "__main__":
    test_dataset_metadata()