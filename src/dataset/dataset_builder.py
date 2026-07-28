import sys
from typing import Any, cast

import numpy as np

from src.config.signal_config import SIGNAL_INFO
from src.entities import DatasetMetadata, Signal
from src.exceptions import CustomException
from src.logger import logger


class DatasetBuilder:
    """
    Build datasets and dataset metadata for representation learning.
    """

    @classmethod
    def build(
        cls,
        signals: dict[str, Signal],
        windows: dict[str, np.ndarray],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, DatasetMetadata],
    ]:

        try:

            datasets: dict[str, np.ndarray] = {}
            metadata: dict[str, DatasetMetadata] = {}

            for signal_name, signal_windows in windows.items():

                signal = signals[signal_name]

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

                dataset = signal_windows.astype(np.float32)

                datasets[signal_name] = dataset

                filter_config: dict[str, Any] = cast(
                    dict[str, Any],
                    config["filter"],
                )

                normalization = cast(
                    str,
                    config["normalization"],
                )

                metadata[signal_name] = DatasetMetadata(
                    signal_name=signal.name,
                    sampling_rate=signal.sampling_rate,
                    channels=signal.channels,
                    window_duration_seconds=window_duration,
                    window_size=window_size,
                    step_size=step_size,
                    overlap=overlap,
                    normalization=normalization,
                    filter_type=(
                        filter_config["type"] if filter_config["enabled"] else "none"
                    ),
                    dtype=str(dataset.dtype),
                    num_windows=dataset.shape[0],
                )

                logger.info(f"{signal_name}: " f"{dataset.shape} dataset created.")

            logger.info("Datasets built successfully.")

            return datasets, metadata

        except Exception as error:

            raise CustomException(error, sys)
