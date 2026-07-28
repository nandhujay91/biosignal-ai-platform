from __future__ import annotations

import numpy as np

from src.inference.predictor import BiosignalPredictor


def main():

    # Load real embeddings
    embeddings = np.load("artifacts/embeddings/embeddings.npy")

    # Load real extracted features
    features = np.load("artifacts/datasets/v1/Ephy/features.npy")

    print(
        "Embeddings Shape:",
        embeddings.shape,
    )

    print(
        "Features Shape:",
        features.shape,
    )

    # Select one real sample

    embedding = embeddings[0]

    signal_features = features[0]

    # Combine embedding + handcrafted features
    # 128 + 3 = 131 features

    fusion_input = np.concatenate(
        [
            embedding,
            signal_features,
        ]
    )

    print(
        "Fusion Input Shape:",
        fusion_input.shape,
    )

    predictor = BiosignalPredictor()

    result = predictor.predict(fusion_input)

    print("\nPrediction Result")
    print("-----------------")

    print(
        "Prediction:",
        result.prediction,
    )

    print(
        "Confidence:",
        result.confidence,
    )

    print(
        "Risk Level:",
        result.risk_level,
    )

    print(
        "Recommended Action:",
        result.recommended_action,
    )

    print(
        "Model Version:",
        result.model_version,
    )

    print("\nReal prediction test passed.")


if __name__ == "__main__":
    main()
