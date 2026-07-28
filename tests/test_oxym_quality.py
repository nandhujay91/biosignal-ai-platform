from src.data_loader import BinaryReader
from src.parser import SignalParser
from src.preprocessing import SignalNormalization
from src.quality import OxymQuality
from src.validation import SignalValidator


def test_oxym_quality():

    raw = BinaryReader.read_all_bin_files("data/raw")

    parsed = SignalParser.parse_signals(raw)

    validated = SignalValidator.validate_signals(parsed)

    signal = validated["Oxym"]

    signal = SignalNormalization.z_score_normalize(signal)

    report = OxymQuality().assess(signal)

    print()

    print(report)


if __name__ == "__main__":
    test_oxym_quality()