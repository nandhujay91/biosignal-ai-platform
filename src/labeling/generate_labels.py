from __future__ import annotations

from pathlib import Path

import numpy as np

from src.entities import Signal
from src.features import FeatureExtractor

from .baseline_rules import BaselineLabeler


class LabelGenerator:
    """
    Production label generation pipeline.

    Input:
        Ephy windows
        Oxym windows

    Output:
        labels.npy

    Labels:
        0 -> Normal
        1 -> Alert
        2 -> Critical
    """

    def __init__(
        self,
        output_path: str | Path = "artifacts/labels/labels.npy",
    ) -> None:

        self.output_path = Path(output_path)

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.feature_extractor = FeatureExtractor(
            ecg_sampling_rate=256,
        )

    def generate(
        self,
        ephy_windows: np.ndarray,
        oxym_windows: np.ndarray,
    ) -> np.ndarray:

        labels = []

        for index in range(len(ephy_windows)):

            ephy_signal = Signal(
                name="Ephy",
                data=ephy_windows[index],
                dtype=ephy_windows[index].dtype,
                channels=ephy_windows[index].shape[1],
                sampling_rate=256,
            )

            oxym_signal = Signal(
                name="Oxym",
                data=oxym_windows[index],
                dtype=oxym_windows[index].dtype,
                channels=oxym_windows[index].shape[1],
                sampling_rate=128,
            )

            features = self.feature_extractor.extract(
                ecg_signal=ephy_signal,
                oxym_signal=oxym_signal,
            )

            label = BaselineLabeler.assign_label(
                heart_rate=features["heart_rate"],
                spo2=features["spo2"],
                signal_quality=(features["quality_score"] / 100),
            )

            labels.append(label)

        labels = np.array(
            labels,
            dtype=np.int64,
        )

        np.save(
            self.output_path,
            labels,
        )

        return labels
