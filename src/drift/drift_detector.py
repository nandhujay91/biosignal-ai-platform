from __future__ import annotations

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


class DriftDetector:
    """
    Detect feature distribution drift
    between training and production data.
    """

    def __init__(self):

        self.report = Report(metrics=[DataDriftPreset()])

    def run(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
    ):

        result = self.report.run(
            reference_data=reference_data,
            current_data=current_data,
        )

        return result
