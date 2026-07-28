import sys
from typing import Any, cast

import numpy as np

from src.config.signal_config import SIGNAL_INFO
from src.entities import Signal
from src.exceptions import CustomException
from src.logger import logger

from .sliding_window import SlidingWindow


class WindowGenerator:
    """
    Generate sliding windows for all biosignals using the
    configuration defined in SIGNAL_INFO.
    """

    @classmethod
    def generate(
        cls,
        signals: dict[str, Signal],
    ) -> dict[str, np.ndarray]:

        try:

            windows: dict[str, np.ndarray] = {}

            for signal_name, signal in signals.items():

                config: dict[str, Any] = cast(
                    dict[str, Any],
                    SIGNAL_INFO[signal_name],
                )

                window_duration = cast(
                    int,
                    config["window_duration"],
                )

                overlap = cast(
                    float,
                    config["overlap"],
                )

                window_size = signal.sampling_rate * window_duration

                step_size = int(window_size * (1 - overlap))

                signal_windows = SlidingWindow.generate(
                    signal=signal,
                    window_size=window_size,
                    step_size=step_size,
                )

                windows[signal_name] = signal_windows

                logger.info(
                    f"{signal_name}: "
                    f"{signal_windows.shape[0]} windows generated "
                    f"(window={window_duration}s, "
                    f"overlap={overlap})."
                )

            logger.info("Window generation completed successfully.")

            return windows

        except Exception as error:

            raise CustomException(error, sys)
