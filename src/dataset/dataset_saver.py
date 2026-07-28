import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from src.config.artifact_config import DATASET_ROOT
from src.entities import DatasetMetadata
from src.exceptions import CustomException
from src.logger import logger


class DatasetSaver:
    """
    Save datasets and metadata for each biosignal.
    """

    ROOT_DIR = DATASET_ROOT

    @classmethod
    def save(
        cls,
        datasets: dict[str, np.ndarray],
        metadata: dict[str, DatasetMetadata],
    ) -> dict[str, Path]:

        try:

            saved_paths = {}

            cls.ROOT_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            for signal_name, dataset in datasets.items():

                signal_dir = cls.ROOT_DIR / signal_name

                signal_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                dataset_path = signal_dir / "windows.npy"
                metadata_path = signal_dir / "metadata.yaml"

                np.save(
                    dataset_path,
                    dataset,
                )

                meta = metadata[signal_name]

                meta.num_windows = int(dataset.shape[0])
                meta.created_at = datetime.now().isoformat()

                with open(
                    metadata_path,
                    "w",
                    encoding="utf-8",
                ) as file:

                    yaml.safe_dump(
                        meta.to_dict(),
                        file,
                        sort_keys=False,
                    )

                saved_paths[signal_name] = signal_dir

                logger.info(f"{signal_name} saved successfully.")

            logger.info("All datasets saved successfully.")

            return saved_paths

        except Exception as error:

            raise CustomException(error, sys)
