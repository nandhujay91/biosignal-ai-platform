from __future__ import annotations

import numpy as np


class InputValidationError(Exception):
    """
    Custom validation error.
    """
    pass



class InputValidator:
    """
    Validate inference input before model prediction.

    Expected input:
    131 features
    """

    EXPECTED_FEATURES = 131


    @classmethod
    def validate(
        cls,
        features,
    ) -> bool:


        # Convert input to numpy array

        array = np.asarray(
            features,
            dtype=np.float32,
        )


        # Check dimension

        if array.ndim != 1:

            raise InputValidationError(
                "Input must be a 1D feature vector."
            )


        # Check feature count

        if len(array) != cls.EXPECTED_FEATURES:

            raise InputValidationError(
                f"Expected {cls.EXPECTED_FEATURES} features, "
                f"received {len(array)}."
            )


        # Check missing values

        if np.isnan(array).any():

            raise InputValidationError(
                "Input contains NaN values."
            )


        # Check infinite values

        if np.isinf(array).any():

            raise InputValidationError(
                "Input contains infinite values."
            )


        return True