from src.data_loader import BinaryReader
from src.parser import SignalParser
from src.preprocessing import SignalNormalization
from src.quality import QualityManager
from src.validation import SignalValidator


def test_quality_manager():

    raw = BinaryReader.read_all_bin_files("data/test")

    parsed = SignalParser.parse_signals(raw)

    validated = SignalValidator.validate_signals(parsed)

    for signal in validated.values():

        normalized = SignalNormalization.z_score_normalize(signal)

        report = QualityManager.assess(normalized)

        print(report)


if __name__ == "__main__":
    test_quality_manager()
