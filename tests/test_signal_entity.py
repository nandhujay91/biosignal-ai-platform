import numpy as np

from src.entities import Signal


def test_signal_entity():

    signal = Signal(
        name="IMU",
        data=np.zeros((10, 9), dtype=np.int16),
        dtype=np.int16,
        channels=9,
        sampling_rate=64,
    )

    print(signal)


if __name__ == "__main__":
    test_signal_entity()
