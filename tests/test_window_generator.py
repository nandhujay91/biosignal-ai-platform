from src.pipeline import SignalPreprocessor
from src.windowing import WindowGenerator


def test_window_generator():

    processed = SignalPreprocessor.run(
        "data/raw"
    )

    windows = WindowGenerator.generate(
        processed
    )

    print()

    print("=" * 60)
    print("Generated Windows")
    print("=" * 60)

    for name, data in windows.items():

        print(f"\n{name}")

        print(f"Shape : {data.shape}")


if __name__ == "__main__":
    test_window_generator()