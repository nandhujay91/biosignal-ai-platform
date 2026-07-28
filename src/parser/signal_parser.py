import sys

import numpy as np

from src.config.signal_config import SIGNAL_INFO
from src.entities import Signal
from src.exceptions import CustomException
from src.logger import logger


class SignalParser:
    """
    Convert raw 1D signal arrays into Signal entities.
    """

    @staticmethod
    def parse_signals(
        raw_signals: dict[str, np.ndarray],
    ) -> dict[str, Signal]:

        try:

            parsed_signals = {}

            for file_name, raw_signal in raw_signals.items():

                signal_type = next(
                    (
                        signal_name
                        for signal_name in SIGNAL_INFO
                        if signal_name.lower() in file_name.lower()
                    ),
                    None,
                )

                if signal_type is None:
                    logger.warning(
                        f"Unknown signal type: {file_name}"
                    )
                    continue

                channels = SIGNAL_INFO[signal_type]["channels"]
                dtype = SIGNAL_INFO[signal_type]["dtype"]
                sampling_rate = SIGNAL_INFO[signal_type]["sampling_rate"]

                if raw_signal.size % channels != 0:
                    raise ValueError(
                        f"{file_name} cannot be reshaped into "
                        f"{channels} channels."
                    )

                parsed_data = raw_signal.reshape(-1, channels)

                signal = Signal(
                    name=signal_type,
                    data=parsed_data,
                    dtype=dtype,
                    channels=channels,
                    sampling_rate=sampling_rate,
                )

                parsed_signals[signal_type] = signal

                logger.info(
                    f"{signal_type}: "
                    f"{raw_signal.shape} -> {parsed_data.shape}"
                )

            logger.info(
                f"Successfully parsed {len(parsed_signals)} signals."
            )

            return parsed_signals

        except Exception as error:
            raise CustomException(error, sys)