from src.data_loader import BinaryReader


def test_bin_reader():

    data = BinaryReader.read_all_bin_files("data/test")

    for name, signal in data.items():

        print(f"\n{name}")

        print(type(signal))

        print(signal.shape)

        print(signal[:10])


if __name__ == "__main__":
    test_bin_reader()
