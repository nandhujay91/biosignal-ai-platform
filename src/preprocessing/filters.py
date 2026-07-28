import sys

import numpy as np
from scipy.signal import butter, filtfilt

from src.entities import Signal
from src.exceptions import CustomException
from src.logger import logger


class SignalFilters:
    """
    Apply digital filters to biosignals.
    """

    @staticmethod
    def butter_bandpass_filter(
        signal: Signal,
        lowcut: float,
        highcut: float,
        order: int = 4,
    ) -> Signal:

        try:

            nyquist = 0.5 * signal.sampling_rate

            low = lowcut / nyquist
            high = highcut / nyquist

            b, a = butter(
                order,
                [low, high],
                btype="band",
            )

            filtered_data = np.zeros_like(signal.data)

            for channel in range(signal.channels):

                filtered_data[:, channel] = filtfilt(
                    b,
                    a,
                    signal.data[:, channel],
                )

            logger.info(
                f"{signal.name}: "
                f"Band-pass filter applied "
                f"({lowcut}-{highcut} Hz)"
            )

            return Signal(
                name=signal.name,
                data=filtered_data.astype(signal.dtype),
                dtype=signal.dtype,
                channels=signal.channels,
                sampling_rate=signal.sampling_rate,
            )

        except Exception as error:
            raise CustomException(error, sys)