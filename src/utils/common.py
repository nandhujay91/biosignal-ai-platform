import json
import pickle
import random
import time
from pathlib import Path

import numpy as np

from src.logger import logger
from src.exceptions import CustomException
import sys


def get_current_timestamp() -> str:
    """
    Return the current timestamp.
    """
    return time.strftime("%Y-%m-%d %H:%M:%S")


def set_random_seed(seed: int = 42) -> None:
    """
    Set the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)

    logger.info(f"Random seed set to {seed}")


def save_json(data: dict, file_path: str | Path) -> None:
    """
    Save a dictionary to a JSON file.
    """
    try:
        file_path = Path(file_path)

        with file_path.open("w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=4)

        logger.info(f"JSON saved: {file_path}")

    except Exception as error:
        raise CustomException(error, sys)


def load_json(file_path: str | Path) -> dict:
    """
    Load a JSON file.
    """
    try:
        file_path = Path(file_path)

        with file_path.open("r", encoding="utf-8") as json_file:
            data = json.load(json_file)

        logger.info(f"JSON loaded: {file_path}")

        return data

    except Exception as error:
        raise CustomException(error, sys)


def save_pickle(data, file_path: str | Path) -> None:
    """
    Save a Python object as a pickle file.
    """
    try:
        file_path = Path(file_path)

        with file_path.open("wb") as pickle_file:
            pickle.dump(data, pickle_file)

        logger.info(f"Pickle saved: {file_path}")

    except Exception as error:
        raise CustomException(error, sys)


def load_pickle(file_path: str | Path):
    """
    Load a pickle file.
    """
    try:
        file_path = Path(file_path)

        with file_path.open("rb") as pickle_file:
            data = pickle.load(pickle_file)

        logger.info(f"Pickle loaded: {file_path}")

        return data

    except Exception as error:
        raise CustomException(error, sys)