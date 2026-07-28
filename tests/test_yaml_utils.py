from src.utils import read_yaml


def test_yaml():

    config = read_yaml("configs/config.yaml")

    print(config)


if __name__ == "__main__":
    test_yaml()