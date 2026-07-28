from __future__ import annotations

from pathlib import Path

import numpy as np

from src.entities import Signal
from src.features import FeatureExtractor
from src.logger import logger


class FeaturePipeline:
    """
    Pipeline for biosignal feature generation.
    """

    def __init__(
        self,
        dataset_path: str = "artifacts/datasets/v1",
        output_path: str = "artifacts/datasets/v1/Ephy/features.npy",
    ) -> None:

        self.dataset_path = Path(dataset_path)

        self.output_path = Path(output_path)

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def run(self):

        ephy_windows = np.load(self.dataset_path / "Ephy" / "windows.npy")

        oxym_windows = np.load(self.dataset_path / "Oxym" / "windows.npy")

        extractor = FeatureExtractor(
            ecg_sampling_rate=256,
        )

        features = []

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

            result = extractor.extract(
                ecg_signal=ephy_signal,
                oxym_signal=oxym_signal,
            )

            features.append(
                [
                    result["heart_rate"],
                    result["spo2"],
                    result["quality_score"],
                ]
            )

        features = np.array(
            features,
            dtype=np.float32,
        )

        np.save(
            self.output_path,
            features,
        )

        logger.info(f"Features saved: {features.shape}")

        return features


if __name__ == "__main__":

    pipeline = FeaturePipeline()

    pipeline.run()
