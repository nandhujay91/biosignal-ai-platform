import numpy as np

from src.labeling.generate_labels import LabelGenerator


def main():

    ephy = np.load("artifacts/datasets/v1/Ephy/windows.npy")

    oxym = np.load("artifacts/datasets/v1/Oxym/windows.npy")

    generator = LabelGenerator()

    labels = generator.generate(
        ephy_windows=ephy,
        oxym_windows=oxym,
    )

    print(
        "Labels shape:",
        labels.shape,
    )

    print(
        "Class distribution:",
        np.unique(
            labels,
            return_counts=True,
        ),
    )

    assert labels.shape[0] == ephy.shape[0]

    print("Real label generation passed.")


if __name__ == "__main__":
    main()
