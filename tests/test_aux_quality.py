from src.data_loader import BinaryReader
from src.parser import SignalParser
from src.preprocessing import SignalNormalization
from src.quality import AUXQuality
from src.validation import SignalValidator


def test_aux_quality():

    raw = BinaryReader.read_all_bin_files("data/raw")

    parsed = SignalParser.parse_signals(raw)

    validated = SignalValidator.validate_signals(parsed)

    signal = validated["Aux"]

    signal = SignalNormalization.z_score_normalize(signal)

    report = AUXQuality().assess(signal)

    print()

    print(report)


if __name__ == "__main__":
    test_aux_quality()