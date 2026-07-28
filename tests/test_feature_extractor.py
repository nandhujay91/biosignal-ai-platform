from __future__ import annotations

import numpy as np

from src.entities import Signal
from src.features import FeatureExtractor


def main() -> None:

    # Create dummy ECG signal
    sampling_rate = 256

    time = np.linspace(
        0,
        10,
        sampling_rate * 10,
    )

    # Simulated ECG with peaks
    ecg = np.sin(2 * np.pi * 1.2 * time)

    ecg = ecg.reshape(
        -1,
        1,
    )

    # Create dummy OXYM signal
    oxym = (
        np.ones(
            (
                sampling_rate * 10,
                1,
            )
        )
        * 20
    )

    ecg_signal = Signal(
        name="Ephy",
        data=ecg,
        dtype=ecg.dtype,
        channels=1,
        sampling_rate=sampling_rate,
    )

    oxym_signal = Signal(
        name="Oxym",
        data=oxym,
        dtype=oxym.dtype,
        channels=1,
        sampling_rate=128,
    )

    extractor = FeatureExtractor(
        ecg_sampling_rate=sampling_rate,
    )

    features = extractor.extract(
        ecg_signal=ecg_signal,
        oxym_signal=oxym_signal,
    )

    print("Extracted Features:")
    print(features)

    assert "heart_rate" in features
    assert "spo2" in features
    assert "quality_score" in features

    assert features["heart_rate"] >= 0
    assert features["spo2"] >= 0
    assert features["quality_score"] >= 0

    print("\nFeature extractor test passed successfully.")


if __name__ == "__main__":
    main()
