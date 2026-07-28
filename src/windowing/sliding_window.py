import sys

import numpy as np

from src.entities import Signal
from src.exceptions import CustomException


class SlidingWindow:
    """
    Generate overlapping windows from a signal.
    """

    @staticmethod
    def generate(
        signal: Signal,
        window_size: int,
        step_size: int,
    ) -> np.ndarray:

        try:

            data = signal.data

            windows = []

            for start in range(
                0,
                len(data) - window_size + 1,
                step_size,
            ):

                end = start + window_size

                windows.append(
                    data[start:end]
                )

            return np.asarray(
                windows,
                dtype=np.float32,
            )

        except Exception as error:

            raise CustomException(error, sys)