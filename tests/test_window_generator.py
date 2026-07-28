from pathlib import Path

from src.pipeline import SignalPreprocessor
from src.windowing import WindowGenerator



def test_window_generator():

    test_data_path = Path(
        "data/test"
    )


    assert test_data_path.exists(), (
        "Test data folder missing"
    )


    processed = SignalPreprocessor.run(
        str(test_data_path)
    )


    windows = WindowGenerator.generate(
        processed
    )


    assert windows is not None


    print()

    print("=" * 60)
    print("Generated Windows")
    print("=" * 60)



    for name, data in windows.items():

        print()

        print(name)

        print(
            f"Shape : {data.shape}"
        )


        assert data is not None



    print()

    print(
        "Window generator test completed successfully."
    )



if __name__ == "__main__":

    test_window_generator()