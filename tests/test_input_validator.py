from __future__ import annotations

import numpy as np

from src.validation.input_validator import (
    InputValidationError,
    InputValidator,
)


def main():

    print("Testing valid input")

    valid_input = np.zeros(
        131,
        dtype=np.float32,
    )

    result = InputValidator.validate(valid_input)

    print(
        "Valid input:",
        result,
    )

    print("\nTesting wrong feature size")

    try:

        wrong_size = np.zeros(
            50,
            dtype=np.float32,
        )

        InputValidator.validate(wrong_size)

    except InputValidationError as e:

        print(
            "Caught:",
            e,
        )

    print("\nTesting NaN input")

    try:

        nan_input = np.zeros(
            131,
            dtype=np.float32,
        )

        nan_input[10] = np.nan

        InputValidator.validate(nan_input)

    except InputValidationError as e:

        print(
            "Caught:",
            e,
        )

    print("\nInput validation test passed.")


if __name__ == "__main__":
    main()
