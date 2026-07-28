import sys

import numpy as np

from src.entities import QualityReport, Signal
from src.exceptions import CustomException
from src.logger import logger

from .base_quality import BaseQuality


class IMUQuality(BaseQuality):
    """
    Production IMU Quality Assessment.
    """

    NAN_PENALTY = 40.0
    INFINITE_PENALTY = 40.0
    FLATLINE_PENALTY = 50.0
    LOW_VARIANCE_PENALTY = 10.0
    RANGE_PENALTY = 20.0

    MAX_ABSOLUTE_VALUE = 10.0

    def assess(self, signal: Signal) -> QualityReport:

        try:

            report = QualityReport(
                signal_name=signal.name,
                passed=True,
                score=100.0,
            )

            self._check_nan(signal, report)
            self._check_infinite(signal, report)
            self._check_flatline(signal, report)
            self._check_low_variance(signal, report)
            self._check_sensor_range(signal, report)

            self._finalize_report(report)

            logger.info(f"{signal.name}: IMU quality assessment completed.")

            return report

        except Exception as error:
            raise CustomException(error, sys)

    def _check_nan(
        self,
        signal: Signal,
        report: QualityReport,
    ) -> None:

        if np.isnan(signal.data).any():
            report.errors.append("NaN values detected")
            report.score -= self.NAN_PENALTY

    def _check_infinite(
        self,
        signal: Signal,
        report: QualityReport,
    ) -> None:

        if np.isinf(signal.data).any():
            report.errors.append("Infinite values detected")
            report.score -= self.INFINITE_PENALTY

    def _check_flatline(
        self,
        signal: Signal,
        report: QualityReport,
    ) -> None:

        std = np.std(signal.data, axis=0)

        if np.all(std < 1e-6):
            report.errors.append("Flatline signal detected")
            report.score -= self.FLATLINE_PENALTY

    def _check_low_variance(
        self,
        signal: Signal,
        report: QualityReport,
    ) -> None:

        variance = np.var(signal.data, axis=0)

        if np.any(variance < 1e-4):
            report.warnings.append("Low variance detected")
            report.score -= self.LOW_VARIANCE_PENALTY

    def _check_sensor_range(
        self,
        signal: Signal,
        report: QualityReport,
    ) -> None:

        if np.any(np.abs(signal.data) > self.MAX_ABSOLUTE_VALUE):
            report.warnings.append("Sensor values exceed expected normalized range")
            report.score -= self.RANGE_PENALTY

    def _finalize_report(
        self,
        report: QualityReport,
    ) -> None:

        report.score = max(report.score, 0.0)

        report.passed = len(report.errors) == 0
