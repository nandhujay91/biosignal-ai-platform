from src.pipeline import SignalPreprocessor


def test_signal_preprocessor():

    processed = SignalPreprocessor.run(
        "data/test"
    )

    print()

    print("=" * 60)
    print("Processed Signals")
    print("=" * 60)

    for name, signal in processed.items():

        print(f"\n{name}")

        print(f"Shape     : {signal.data.shape}")
        print(f"Dtype     : {signal.data.dtype}")
        print(f"Channels  : {signal.channels}")
        print(f"Sampling  : {signal.sampling_rate}")


if __name__ == "__main__":
    test_signal_preprocessor()
