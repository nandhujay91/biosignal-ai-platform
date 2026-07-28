from pathlib import Path
import sys

import yaml

from src.exceptions import CustomException
from src.logger import logger


def read_yaml(file_path: str | Path) -> dict:
    """
    Read a YAML file and return its contents as a dictionary.

    Args:
        file_path: Path to the YAML file.

    Returns:
        Dictionary containing the YAML contents.
    """

    try:
        file_path = Path(file_path)

        with file_path.open("r", encoding="utf-8") as yaml_file:
            data = yaml.safe_load(yaml_file)

        logger.info(f"YAML file loaded successfully: {file_path}")

        return data

    except Exception as error:
        logger.error(f"Failed to load YAML file: {file_path}")
        raise CustomException(error, sys)