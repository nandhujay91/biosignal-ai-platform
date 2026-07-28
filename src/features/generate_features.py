from __future__ import annotations

from pathlib import Path

import numpy as np

from src.entities import Signal
from src.features import FeatureExtractor


def generate_features():

    ephy_windows = np.load(
        "artifacts/datasets/v1/Ephy/windows.npy"
    )

    oxym_windows = np.load(
        "artifacts/datasets/v1/Oxym/windows.npy"
    )


    extractor = FeatureExtractor(
        ecg_sampling_rate=256,
    )


    all_features = []


    for ephy, oxym in zip(
        ephy_windows,
        oxym_windows,
    ):

        ephy_signal = Signal(
            name="Ephy",
            data=ephy,
            dtype=ephy.dtype,
            channels=ephy.shape[1],
            sampling_rate=256,
        )


        oxym_signal = Signal(
            name="Oxym",
            data=oxym,
            dtype=oxym.dtype,
            channels=oxym.shape[1],
            sampling_rate=128,
        )


        features = extractor.extract(
            ecg_signal=ephy_signal,
            oxym_signal=oxym_signal,
        )


        all_features.append(
            [
                features["heart_rate"],
                features["spo2"],
                features["quality_score"],
            ]
        )


    features_array = np.array(
        all_features,
        dtype=np.float32,
    )


    output = Path(
        "artifacts/datasets/v1/Ephy/features.npy"
    )


    np.save(
        output,
        features_array,
    )


    print(
        "Saved:",
        output,
    )

    print(
        "Shape:",
        features_array.shape,
    )


if __name__ == "__main__":
    generate_features()