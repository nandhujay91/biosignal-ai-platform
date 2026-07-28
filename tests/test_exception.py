import sys

from src.exceptions import CustomException
from src.logger import logger


def divide_numbers():
    return 10 / 0


def test_custom_exception():
    try:
        divide_numbers()

    except Exception as error:
        logger.error(CustomException(error, sys))


if __name__ == "__main__":
    test_custom_exception()