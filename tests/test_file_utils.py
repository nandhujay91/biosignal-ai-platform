from src.utils import (
    create_directory,
    directory_exists,
    get_extension,
    get_file_name,
    get_file_stem,
)


def test_file_utils():

    directory = create_directory("data/test")

    print(directory)

    print(directory_exists(directory))

    print(get_file_name("sample.bin"))

    print(get_file_stem("sample.bin"))

    print(get_extension("sample.bin"))


if __name__ == "__main__":
    test_file_utils()