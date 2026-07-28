from __future__ import annotations

import numpy as np


class SpO2Extractor:
    """
    Estimate SpO2 from pulse oximeter signal.

    Note:
    Real clinical SpO2 requires device calibration.
    This implementation provides a production pipeline interface.
    """

    def calculate_spo2(
        self,
        oxym_signal: np.ndarray,
    ) -> float:
        """
        Calculate SpO2 estimation.

        Input:
            Oxym signal window

        Output:
            SpO2 percentage
        """

        if oxym_signal.ndim > 1:
            oxym_signal = oxym_signal[:, 0]

        # Remove invalid values
        oxym_signal = oxym_signal[np.isfinite(oxym_signal)]

        if len(oxym_signal) == 0:
            return 0.0

        signal_mean = np.mean(np.abs(oxym_signal))

        # Placeholder calibration curve
        # Production device calibration replaces this
        spo2 = 100 - (signal_mean * 0.05)

        spo2 = np.clip(
            spo2,
            70,
            100,
        )

        return float(spo2)
