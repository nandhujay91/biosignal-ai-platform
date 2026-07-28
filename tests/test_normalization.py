from src.data_loader import BinaryReader
from src.parser import SignalParser
from src.preprocessing import (
    SignalFilters,
    SignalNormalization,
)
from src.validation import SignalValidator


def test_normalization():

    raw_signals = BinaryReader.read_all_bin_files(
        "data/raw"
    )

    parsed_signals = SignalParser.parse_signals(
        raw_signals
    )

    validated_signals = SignalValidator.validate_signals(
        parsed_signals
    )

    ephy = validated_signals["Ephy"]

    filtered = SignalFilters.butter_bandpass_filter(
        signal=ephy,
        lowcut=0.5,
        highcut=40.0,
    )

    normalized = SignalNormalization.z_score_normalize(
        filtered
    )

    print(normalized.name)

    print(normalized.data.shape)

    print(normalized.dtype)

    print(normalized.sampling_rate)

    print("\nMean per channel")

    print(normalized.data.mean(axis=0))

    print("\nStd per channel")

    print(normalized.data.std(axis=0))

    print("\nFirst 3 Samples")

    print(normalized.data[:3])


if __name__ == "__main__":
    test_normalization()