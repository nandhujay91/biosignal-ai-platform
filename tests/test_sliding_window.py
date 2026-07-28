from src.pipeline import SignalPreprocessor
from src.windowing import SlidingWindow


def test_sliding_window():

    signals = SignalPreprocessor.run(
        "data/test"
    )

    ecg = signals["Ephy"]

    windows = SlidingWindow.generate(
        signal=ecg,
        window_size=1280,
        step_size=640,
    )

    print()

    print("Number of windows :", windows.shape[0])
    print("Window size       :", windows.shape[1])
    print("Channels          :", windows.shape[2])
    print("Shape             :", windows.shape)


if __name__ == "__main__":
    test_sliding_window()
