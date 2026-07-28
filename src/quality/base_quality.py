from abc import ABC, abstractmethod

from src.entities import QualityReport, Signal


class BaseQuality(ABC):
    """
    Abstract base class for all biosignal quality checkers.
    """

    @abstractmethod
    def assess(self, signal: Signal) -> QualityReport:
        """
        Assess the quality of a signal.

        Parameters
        ----------
        signal : Signal
            Input biosignal.

        Returns
        -------
        QualityReport
            Quality assessment result.
        """
