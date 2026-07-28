import sys

from src.entities import QualityReport, Signal
from src.exceptions import CustomException

from .aux_quality import AUXQuality
from .ecg_quality import ECGQuality
from .imu_quality import IMUQuality
from .oxym_quality import OxymQuality


class QualityManager:
    """
    Factory/Dispatcher for sensor-specific quality assessment.
    """

    _QUALITY_MAP = {
        "Ephy": ECGQuality(),
        "IMU": IMUQuality(),
        "Oxym": OxymQuality(),
        "Aux": AUXQuality(),
    }

    @classmethod
    def assess(cls, signal: Signal) -> QualityReport:

        try:

            quality_checker = cls._QUALITY_MAP.get(signal.name)

            if quality_checker is None:
                raise ValueError(f"No quality checker registered for '{signal.name}'.")

            return quality_checker.assess(signal)

        except Exception as error:
            raise CustomException(error, sys)
