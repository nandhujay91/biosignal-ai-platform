import sys

import numpy as np
import yaml

from src.config.artifact_config import DATASET_ROOT
from src.entities import DatasetMetadata
from src.exceptions import CustomException
from src.logger import logger


class DatasetLoader:
    """
    Load saved datasets and metadata.
    """

    ROOT_DIR = DATASET_ROOT

    @classmethod
    def load(
        cls,
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, DatasetMetadata],
    ]:

        try:

            datasets = {}
            metadata = {}

            if not cls.ROOT_DIR.exists():

                raise FileNotFoundError(f"{cls.ROOT_DIR} does not exist.")

            for signal_dir in sorted(cls.ROOT_DIR.iterdir()):

                if not signal_dir.is_dir():
                    continue

                signal_name = signal_dir.name

                dataset_path = signal_dir / "windows.npy"

                metadata_path = signal_dir / "metadata.yaml"

                dataset = np.load(dataset_path)

                with open(
                    metadata_path,
                    "r",
                    encoding="utf-8",
                ) as file:

                    meta = DatasetMetadata.from_dict(yaml.safe_load(file))

                datasets[signal_name] = dataset
                metadata[signal_name] = meta

                logger.info(f"{signal_name}: " f"{dataset.shape} loaded.")

            logger.info("Datasets loaded successfully.")

            return datasets, metadata

        except Exception as error:

            raise CustomException(error, sys)
