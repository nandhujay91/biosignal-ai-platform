from pathlib import Path
import sys

import numpy as np

from src.config.signal_config import SIGNAL_INFO
from src.exceptions import CustomException
from src.logger import logger
from src.utils import validate_directory_exists


class BinaryReader:
    """
    Read all binary (.bin) files from a directory.
    """

    @classmethod
    def read_all_bin_files(
        cls,
        folder_path: str | Path,
    ) -> dict[str, np.ndarray]:
        """
        Read all .bin files from a folder.

        Args:
            folder_path: Folder containing .bin files.

        Returns:
            Dictionary where:
                key   -> filename without extension
                value -> NumPy array
        """

        try:
            folder_path = Path(folder_path)

            validate_directory_exists(folder_path)

            bin_files = sorted(folder_path.glob("*.bin"))

            if not bin_files:
                raise FileNotFoundError(
                    f"No .bin files found in {folder_path}"
                )

            data = {}

            for file in bin_files:

                signal_type = next(
                    (
                        signal
                        for signal in SIGNAL_INFO
                        if signal.lower() in file.name.lower()
                    ),
                    None,
                )

                if signal_type is None:
                    logger.warning(
                        f"Unknown signal type. Skipping {file.name}"
                    )
                    continue

                dtype = SIGNAL_INFO[signal_type]["dtype"]

                logger.info(
                    f"Reading {file.name} as {dtype.__name__}"
                )

                signal = np.fromfile(
                    file,
                    dtype=dtype,
                )

                if signal.size == 0:
                    logger.warning(
                        f"{file.name} is empty."
                    )
                    continue

                data[file.stem] = signal

            logger.info(
                f"Successfully loaded {len(data)} binary files."
            )

            return data

        except Exception as error:
            raise CustomException(error, sys)