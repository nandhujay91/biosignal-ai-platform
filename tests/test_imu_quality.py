from src.data_loader import BinaryReader
from src.parser import SignalParser
from src.preprocessing import (
    SignalFilters,
    SignalNormalization,
)
from src.quality import IMUQuality
from src.validation import SignalValidator


def test_imu_quality():

    raw = BinaryReader.read_all_bin_files("data/test")

    parsed = SignalParser.parse_signals(raw)

    validated = SignalValidator.validate_signals(parsed)

    signal = validated["IMU"]

    signal = SignalNormalization.z_score_normalize(signal)

    report = IMUQuality().assess(signal)

    print()

    print(report)


if __name__ == "__main__":
    test_imu_quality()
