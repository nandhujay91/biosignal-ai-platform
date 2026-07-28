import sys

import numpy as np

from src.entities import Signal
from src.exceptions import CustomException
from src.logger import logger


class SignalValidator:
    """
    Validate Signal entities before preprocessing.
    """

    @staticmethod
    def validate_signals(
        parsed_signals: dict[str, Signal],
    ) -> dict[str, Signal]:

        try:

            validated_signals = {}

            for signal_name, signal in parsed_signals.items():

                if not isinstance(signal, Signal):
                    raise TypeError(f"{signal_name} is not a Signal entity.")

                if not isinstance(signal.data, np.ndarray):
                    raise TypeError(f"{signal_name}.data is not a NumPy array.")

                if signal.data.size == 0:
                    raise ValueError(f"{signal_name} is empty.")

                if signal.data.ndim != 2:
                    raise ValueError(f"{signal_name} must be a 2D array.")

                if signal.data.shape[1] != signal.channels:
                    raise ValueError(
                        f"{signal_name} expected "
                        f"{signal.channels} channels "
                        f"but found {signal.data.shape[1]}."
                    )

                if signal.data.dtype != signal.dtype:
                    raise TypeError(
                        f"{signal_name} expected "
                        f"{signal.dtype} "
                        f"but found {signal.data.dtype}."
                    )

                if np.isnan(signal.data).any():
                    raise ValueError(f"{signal_name} contains NaN values.")

                if np.isinf(signal.data).any():
                    raise ValueError(f"{signal_name} contains infinite values.")

                logger.info(f"{signal_name} validation passed.")

                validated_signals[signal_name] = signal

            logger.info(f"Successfully validated " f"{len(validated_signals)} signals.")

            return validated_signals

        except Exception as error:
            raise CustomException(error, sys)
