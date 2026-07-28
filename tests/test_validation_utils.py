from src.utils import (
    validate_directory_exists,
    validate_extension,
)


def test_validation_utils():

    validate_directory_exists("configs")

    validate_extension("sample.bin", ".bin")

    print("All validations passed.")


if __name__ == "__main__":
    test_validation_utils()
