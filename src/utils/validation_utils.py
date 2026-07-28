from pathlib import Path


def validate_file_exists(file_path: str | Path) -> bool:
    """
    Validate that a file exists.
    """
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    return True


def validate_directory_exists(directory_path: str | Path) -> bool:
    """
    Validate that a directory exists.
    """
    path = Path(directory_path)

    if not path.is_dir():
        raise NotADirectoryError(f"Directory not found: {path}")

    return True


def validate_extension(file_path: str | Path, extension: str) -> bool:
    """
    Validate the file extension.
    """
    path = Path(file_path)

    if path.suffix.lower() != extension.lower():
        raise ValueError(f"Expected '{extension}' file, but received '{path.suffix}'.")

    return True
