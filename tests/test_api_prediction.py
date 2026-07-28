from __future__ import annotations

import numpy as np
import requests


def main():

    embeddings = np.load(
        "artifacts/embeddings/embeddings.npy"
    )

    features = np.load(
        "artifacts/datasets/v1/Ephy/features.npy"
    )


    fusion_input = np.concatenate(
        [
            embeddings[0],
            features[0],
        ]
    )


    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json={
            "features": fusion_input.tolist()
        },
    )


    print("Status:")
    print(response.status_code)


    print("\nResponse:")
    print(response.json())


if __name__ == "__main__":
    main()