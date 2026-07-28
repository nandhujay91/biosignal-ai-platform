from __future__ import annotations

import numpy as np

from src.explainability.shap_explainer import SHAPExplainer


def load_real_fusion_features():
    """
    Prepare fusion features for SHAP explanation.

    Expected input:

    TS2Vec embedding:
        128 dimensions

    Signal features:
        3 dimensions

    Total:
        131 dimensions
    """

    # TODO:
    # Replace this block with your real TS2Vec encoder output
    #
    # Example:
    #
    # embedding = ts2vec_model.encode(signal_window)
    #
    # Expected:
    # embedding.shape = (128,)

    embedding = np.random.rand(128).astype(np.float32)

    # Real signal-level features
    #
    # Replace with your extracted biosignal features:
    # HR, SpO2, signal quality, etc.

    signal_features = np.array(
        [
            0.72,
            0.85,
            0.91,
        ],
        dtype=np.float32,
    )

    fusion_features = np.concatenate(
        [
            embedding,
            signal_features,
        ]
    )

    if fusion_features.shape[0] != 131:

        raise ValueError(f"Expected 131 features, got {fusion_features.shape[0]}")

    return fusion_features


def main():

    print("Loading SHAP explainer...")

    explainer = SHAPExplainer()

    features = load_real_fusion_features()

    print("Fusion feature shape:", features.shape)

    result = explainer.explain(features)

    print("\nPrediction")
    print("----------------")

    print("Class:", result["prediction_class"])

    print("Confidence:", result["confidence"])

    print("\nTop Important Features")
    print("----------------------")

    for item in result["top_features"]:

        print(f"{item['feature']} => {item['impact']:.6f}")

    print("\nSHAP explanation generated successfully.")


if __name__ == "__main__":

    main()
