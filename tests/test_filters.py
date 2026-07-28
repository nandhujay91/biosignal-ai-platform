from src.data_loader import BinaryReader
from src.parser import SignalParser
from src.preprocessing import SignalFilters
from src.validation import SignalValidator


def test_bandpass_filter():

    raw_signals = BinaryReader.read_all_bin_files("data/test")

    parsed_signals = SignalParser.parse_signals(raw_signals)

    validated_signals = SignalValidator.validate_signals(parsed_signals)

    ephy = validated_signals["Ephy"]

    filtered = SignalFilters.butter_bandpass_filter(
        signal=ephy,
        lowcut=0.5,
        highcut=40.0,
    )

    print(filtered.name)
    print(filtered.data.shape)
    print(filtered.dtype)
    print(filtered.sampling_rate)

    print("\nFirst 3 Samples")

    print(filtered.data[:3])


if __name__ == "__main__":
    test_bandpass_filter()
