from src.data_loader import BinaryReader
from src.parser import SignalParser
from src.preprocessing import (
    SignalFilters,
    SignalNormalization,
)
from src.quality import ECGQuality
from src.validation import SignalValidator


def test_ecg_quality():

    raw = BinaryReader.read_all_bin_files("data/test")

    parsed = SignalParser.parse_signals(raw)

    validated = SignalValidator.validate_signals(parsed)

    signal = validated["Ephy"]

    signal = SignalFilters.butter_bandpass_filter(
        signal,
        lowcut=0.5,
        highcut=40.0,
    )

    signal = SignalNormalization.z_score_normalize(signal)

    report = ECGQuality().assess(signal)

    print()

    print(report)


if __name__ == "__main__":
    test_ecg_quality()
