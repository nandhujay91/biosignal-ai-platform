from src.utils import (
    get_current_timestamp,
    save_json,
    load_json,
    set_random_seed,
)


def test_common():

    print(get_current_timestamp())

    set_random_seed(42)

    sample = {
        "name": "Embedding Model",
        "version": 1,
    }

    save_json(sample, "data/test/sample.json")

    loaded = load_json("data/test/sample.json")

    print(loaded)


if __name__ == "__main__":
    test_common()