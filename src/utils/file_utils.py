from pathlib import Path

from src.exceptions import CustomException
from src.logger import logger


def create_directory(path: str | Path) -> Path:
    """
    Create a directory if it does not already exist.
    """

    try:
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"Directory is ready: {directory}")

        return directory

    except Exception as error:
        raise CustomException(error, __import__("sys"))


def file_exists(path: str | Path) -> bool:
    """
    Check whether a file exists.
    """

    return Path(path).is_file()


def directory_exists(path: str | Path) -> bool:
    """
    Check whether a directory exists.
    """

    return Path(path).is_dir()


def list_files(directory: str | Path, extension: str | None = None) -> list[Path]:
    """
    List all files inside a directory.
    Optionally filter by extension.
    """

    directory = Path(directory)

    if extension is None:
        return [file for file in directory.iterdir() if file.is_file()]

    return list(directory.glob(f"*{extension}"))


def get_file_size(path: str | Path) -> int:
    """
    Return file size in bytes.
    """

    return Path(path).stat().st_size


def get_file_name(path: str | Path) -> str:
    """
    Return filename with extension.
    """

    return Path(path).name


def get_file_stem(path: str | Path) -> str:
    """
    Return filename without extension.
    """

    return Path(path).stem


def get_extension(path: str | Path) -> str:
    """
    Return file extension.
    """

    return Path(path).suffix.lower()
