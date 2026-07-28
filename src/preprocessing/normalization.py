import sys

import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.entities import Signal
from src.exceptions import CustomException
from src.logger import logger


class SignalNormalization:
    """
    Normalize biosignals.
    """

    @staticmethod
    def z_score_normalize(signal: Signal) -> Signal:

        try:

            scaler = StandardScaler()

            normalized_data = scaler.fit_transform(signal.data)

            logger.info(
                f"{signal.name}: Z-Score normalization applied."
            )

            return Signal(
                name=signal.name,
                data=normalized_data.astype(np.float32),
                dtype=np.float32,
                channels=signal.channels,
                sampling_rate=signal.sampling_rate,
            )

        except Exception as error:
            raise CustomException(error, sys)

    @staticmethod
    def min_max_normalize(signal: Signal) -> Signal:

        try:

            scaler = MinMaxScaler()

            normalized_data = scaler.fit_transform(signal.data)

            logger.info(
                f"{signal.name}: Min-Max normalization applied."
            )

            return Signal(
                name=signal.name,
                data=normalized_data.astype(np.float32),
                dtype=np.float32,
                channels=signal.channels,
                sampling_rate=signal.sampling_rate,
            )

        except Exception as error:
            raise CustomException(error, sys)

    @staticmethod
    def robust_normalize(signal: Signal) -> Signal:

        try:

            scaler = RobustScaler()

            normalized_data = scaler.fit_transform(signal.data)

            logger.info(
                f"{signal.name}: Robust normalization applied."
            )

            return Signal(
                name=signal.name,
                data=normalized_data.astype(np.float32),
                dtype=np.float32,
                channels=signal.channels,
                sampling_rate=signal.sampling_rate,
            )

        except Exception as error:
            raise CustomException(error, sys)