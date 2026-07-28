from src.data_loader import BinaryReader
from src.parser import SignalParser
from src.validation import SignalValidator


def test_signal_validator():

    raw_signals = BinaryReader.read_all_bin_files(
        "data/raw"
    )

    parsed_signals = SignalParser.parse_signals(
        raw_signals
    )

    validated_signals = SignalValidator.validate_signals(
        parsed_signals
    )

    for name, signal in validated_signals.items():

        print(f"\n{name}")

        print(f"Shape          : {signal.data.shape}")
        print(f"Dtype          : {signal.dtype}")
        print(f"Channels       : {signal.channels}")
        print(f"Sampling Rate  : {signal.sampling_rate} Hz")

        print("\nFirst 3 Samples:")

        print(signal.data[:3])


if __name__ == "__main__":
    test_signal_validator()