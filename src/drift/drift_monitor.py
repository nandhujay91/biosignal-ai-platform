from __future__ import annotations

import pandas as pd

from src.drift.drift_detector import DriftDetector


class DriftMonitor:
    """
    Production drift monitoring logic.

    Flow:

    Reference Data
          |
          ↓
    Current Production Data
          |
          ↓
    Evidently Drift Detection
          |
          ↓
    Drift Score
          |
          ↓
    Retraining Decision
    """

    def __init__(
        self,
        drift_threshold: float = 0.3,
    ):

        self.drift_threshold = drift_threshold
        self.detector = DriftDetector()

    def check_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
    ):
        """
        Run drift detection.

        Returns:
            drift_detected
            drift_score
            threshold
            action
        """

        report = self.detector.run(
            reference_data=reference_data,
            current_data=current_data,
        )

        drift_score = 0.0

        try:

            # Evidently result
            result = report.dict()

            metrics = result.get("metrics", [])

            for metric in metrics:

                value = metric.get("value", {})

                if isinstance(value, dict):

                    possible_keys = [
                        "share_of_drifted_columns",
                        "drift_share",
                        "share",
                    ]

                    for key in possible_keys:

                        if key in value:

                            drift_score = float(value[key])

                            break

                if drift_score > 0:
                    break

        except Exception as e:

            print("Drift parsing warning:", e)

        drift_detected = drift_score >= self.drift_threshold

        action = "retrain_required" if drift_detected else "no_action"

        return {
            "drift_detected": bool(drift_detected),
            "drift_score": float(drift_score),
            "threshold": self.drift_threshold,
            "action": action,
        }
