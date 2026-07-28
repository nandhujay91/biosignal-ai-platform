from __future__ import annotations

from src.entities import Signal
from src.quality import QualityManager

from .heart_rate import HeartRateExtractor
from .spo2 import SpO2Extractor


class FeatureExtractor:
    """
    Production biosignal feature extraction pipeline.

    Extracts:
    - Heart rate
    - SpO2
    - Signal quality
    """

    def __init__(
        self,
        ecg_sampling_rate: int = 256,
    ) -> None:

        self.heart_rate_extractor = HeartRateExtractor(
            sampling_rate=ecg_sampling_rate,
        )

        self.spo2_extractor = SpO2Extractor()


    def extract(
        self,
        ecg_signal: Signal,
        oxym_signal: Signal,
    ) -> dict[str, float]:

        # Heart rate
        heart_rate = (
            self.heart_rate_extractor.calculate_bpm(
                ecg_signal.data
            )
        )


        # SpO2
        spo2 = (
            self.spo2_extractor.calculate_spo2(
                oxym_signal.data
            )
        )


        # ECG quality
        quality_report = (
            QualityManager.assess(
                ecg_signal
            )
        )


        return {
            "heart_rate": heart_rate,
            "spo2": spo2,
            "quality_score": quality_report.score,
        }