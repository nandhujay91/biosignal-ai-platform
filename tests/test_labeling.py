from __future__ import annotations

from src.labeling.baseline_rules import BaselineLabeler


def main() -> None:

    # Normal case
    normal = BaselineLabeler.assign_label(
        heart_rate=75,
        spo2=98,
        signal_quality=0.95,
    )

    print("Normal Label:", normal)

    assert normal == BaselineLabeler.NORMAL

    # Alert case
    alert = BaselineLabeler.assign_label(
        heart_rate=110,
        spo2=93,
        signal_quality=0.85,
    )

    print("Alert Label:", alert)

    assert alert == BaselineLabeler.ALERT

    # Critical case
    critical = BaselineLabeler.assign_label(
        heart_rate=140,
        spo2=85,
        signal_quality=0.40,
    )

    print("Critical Label:", critical)

    assert critical == BaselineLabeler.CRITICAL

    print("\nLabeling test passed successfully.")


if __name__ == "__main__":
    main()
