from src.data_loader import BinaryReader
from src.parser import SignalParser


def test_signal_parser():

    raw_signals = BinaryReader.read_all_bin_files(
        "data/raw"
    )

    parsed_signals = SignalParser.parse_signals(
        raw_signals
    )

    for name, signal in parsed_signals.items():

        print(f"\n{name}")

        print(f"Shape          : {signal.data.shape}")
        print(f"Dtype          : {signal.dtype}")
        print(f"Channels       : {signal.channels}")
        print(f"Sampling Rate  : {signal.sampling_rate} Hz")

        print("\nFirst 3 Samples:")

        print(signal.data[:3])


if __name__ == "__main__":
    test_signal_parser()