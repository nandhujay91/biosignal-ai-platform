from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


class HeartRateExtractor:
    """
    Extract heart rate from ECG signal.
    """

    def __init__(
        self,
        sampling_rate: int = 256,
    ) -> None:

        self.sampling_rate = sampling_rate

    def calculate_bpm(
        self,
        ecg_signal: np.ndarray,
    ) -> float:
        """
        Calculate BPM from ECG window.
        """

        if ecg_signal.ndim > 1:
            ecg_signal = ecg_signal[:, 0]

        peaks, _ = find_peaks(
            ecg_signal,
            distance=self.sampling_rate * 0.4,
        )

        if len(peaks) < 2:
            return 0.0

        rr_intervals = np.diff(peaks) / self.sampling_rate

        mean_rr = np.mean(rr_intervals)

        bpm = 60 / mean_rr

        return float(bpm)
