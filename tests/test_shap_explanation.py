from __future__ import annotations

import numpy as np

from src.explainability.shap_explainer import SHAPExplainer


def main():

    print("Loading SHAP explainer...")

    explainer = SHAPExplainer()

    # Example fusion vector
    # 128 embedding + 3 features

    features = np.random.rand(131)

    print("Feature shape:", features.shape)

    result = explainer.explain(features)

    print("\nPrediction")
    print("----------------")

    print("Class:", result["prediction_class"])

    print("Confidence:", result["confidence"])

    print("\nTop Important Features")
    print("----------------------")

    for item in result["top_features"]:

        print(item["feature"], "=>", item["impact"])

    print("\nSHAP explanation generated.")


if __name__ == "__main__":

    main()
