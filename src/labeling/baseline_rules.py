from __future__ import annotations


class BaselineLabeler:
    """
    Generate labels from biosignal baseline rules.

    Labels:
        0 -> Normal
        1 -> Alert
        2 -> Critical
    """

    NORMAL = 0
    ALERT = 1
    CRITICAL = 2

    @staticmethod
    def assign_label(
        heart_rate: float,
        spo2: float,
        signal_quality: float,
    ) -> int:
        """
        Rule-based label generation.
        """

        # Critical conditions
        if (
            spo2 < 90
            or heart_rate > 120
            or signal_quality < 0.5
        ):
            return BaselineLabeler.CRITICAL

        # Alert conditions
        if (
            spo2 < 95
            or heart_rate > 100
            or signal_quality < 0.8
        ):
            return BaselineLabeler.ALERT

        # Normal
        return BaselineLabeler.NORMAL