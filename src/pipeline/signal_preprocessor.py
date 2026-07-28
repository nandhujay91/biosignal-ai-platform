import sys
from typing import Any, cast

from src.config.signal_config import SIGNAL_INFO
from src.data_loader import BinaryReader
from src.entities import Signal
from src.exceptions import CustomException
from src.logger import logger
from src.parser import SignalParser
from src.preprocessing import (
    SignalFilters,
    SignalNormalization,
)
from src.quality import QualityManager
from src.validation import SignalValidator


class SignalPreprocessor:
    """
    Complete biosignal preprocessing pipeline.

    Pipeline
    --------
    1. Read binary files
    2. Parse signals
    3. Validate signals
    4. Apply configured preprocessing
    5. Assess signal quality
    """

    @classmethod
    def run(
        cls,
        data_directory: str,
    ) -> dict[str, Signal]:

        try:

            logger.info(
                "Starting preprocessing pipeline..."
            )

            # Read binary files
            raw_signals = BinaryReader.read_all_bin_files(
                data_directory
            )

            # Parse signals
            parsed_signals = SignalParser.parse_signals(
                raw_signals
            )

            # Validate signals
            validated_signals = SignalValidator.validate_signals(
                parsed_signals
            )

            processed_signals: dict[str, Signal] = {}

            for signal_name, signal in validated_signals.items():

                logger.info(
                    f"Processing {signal_name}..."
                )

                config: dict[str, Any] = cast(
                    dict[str, Any],
                    SIGNAL_INFO[signal_name],
                )

                # --------------------------------------------------
                # Filtering
                # --------------------------------------------------

                filter_config: dict[str, Any] = cast(
                    dict[str, Any],
                    config["filter"],
                )

                if filter_config["enabled"]:

                    signal = SignalFilters.butter_bandpass_filter(
                        signal=signal,
                        lowcut=float(
                            filter_config["lowcut"]
                        ),
                        highcut=float(
                            filter_config["highcut"]
                        ),
                        order=int(
                            filter_config["order"]
                        ),
                    )

                # --------------------------------------------------
                # Normalization
                # --------------------------------------------------

                normalization = cast(
                    str,
                    config["normalization"],
                )

                if normalization == "z_score":

                    signal = SignalNormalization.z_score_normalize(
                        signal
                    )

                elif normalization == "min_max":

                    signal = SignalNormalization.min_max_normalize(
                        signal
                    )

                elif normalization == "robust":

                    signal = SignalNormalization.robust_normalize(
                        signal
                    )

                else:

                    raise ValueError(
                        f"Unsupported normalization "
                        f"'{normalization}' "
                        f"for signal '{signal_name}'."
                    )

                # --------------------------------------------------
                # Quality Assessment
                # --------------------------------------------------

                report = QualityManager.assess(
                    signal
                )

                logger.info(
                    f"{signal_name} Quality Score: "
                    f"{report.score:.1f}"
                )

                if report.passed:

                    processed_signals[signal_name] = signal

                    logger.info(
                        f"{signal_name} accepted."
                    )

                else:

                    logger.warning(
                        f"{signal_name} rejected."
                    )

            logger.info(
                f"Pipeline completed successfully. "
                f"{len(processed_signals)} signals processed."
            )

            return processed_signals

        except Exception as error:

            raise CustomException(error, sys)