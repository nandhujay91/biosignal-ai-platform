from __future__ import annotations

import numpy as np


def main():

    labels = np.load(
        "artifacts/labels/labels.npy"
    )


    # Load features generated from dataset
    features = np.load(
        "artifacts/datasets/v1/Ephy/features.npy"
    )


    print(
        "Labels Shape:",
        labels.shape,
    )


    print(
        "Features Shape:",
        features.shape,
    )


    classes = {
        0: "Normal",
        1: "Alert",
        2: "Critical",
    }


    for class_id, class_name in classes.items():

        class_features = features[
            labels == class_id
        ]


        print("\nClass:", class_name)

        print(
            "Samples:",
            len(class_features),
        )


        print(
            "Mean:",
            np.mean(
                class_features,
                axis=0,
            )
        )


        print(
            "Std:",
            np.std(
                class_features,
                axis=0,
            )
        )


    print(
        "\nLabel analysis completed."
    )


if __name__ == "__main__":
    main()