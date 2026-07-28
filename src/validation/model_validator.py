from __future__ import annotations

import os


class ModelValidator:
    """
    Production model validation gate.

    Checks:

    1. Model file exists
    2. Accuracy improvement
    3. Minimum accuracy threshold
    4. Loss improvement
    """

    def __init__(
        self,
        minimum_accuracy: float = 0.90,
    ):

        self.minimum_accuracy = minimum_accuracy

    def validate(
        self,
        old_accuracy: float,
        new_accuracy: float,
        old_loss: float,
        new_loss: float,
        model_path: str,
    ):

        checks = {}

        # Model file check

        checks["model_exists"] = os.path.exists(model_path)

        # Accuracy check

        checks["accuracy_improved"] = new_accuracy > old_accuracy

        checks["minimum_accuracy"] = new_accuracy >= self.minimum_accuracy

        # Loss check

        checks["loss_improved"] = new_loss < old_loss

        approved = all(checks.values())

        return {
            "approved": approved,
            "checks": checks,
            "decision": ("promote" if approved else "reject"),
        }
