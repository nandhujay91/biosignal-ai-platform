from __future__ import annotations

import torch
import shap
import numpy as np

from src.classifier.classifier import EmbeddingClassifier


class SHAPExplainer:
    """
    SHAP based explanation for biosignal embedding classifier.

    Explains:
    - 128 TS2Vec embedding features
    - 3 signal features
    - total 131 fusion features
    """

    def __init__(
        self,
        model_path: str = "artifacts/model/v1/classifier.pt",
    ):

        self.model_path = model_path

        self.model = self._load_model()


        self.feature_names = (
            [
                f"embedding_{i}"
                for i in range(128)
            ]
            +
            [
                "signal_feature_0",
                "signal_feature_1",
                "signal_feature_2",
            ]
        )


    def _load_model(self):

        model = EmbeddingClassifier(
            input_dim=131,
            num_classes=3,
        )


        checkpoint = torch.load(
            self.model_path,
            map_location="cpu",
        )


        model.load_state_dict(
            checkpoint["model_state_dict"]
        )


        model.eval()

        return model



    def explain(
        self,
        features,
    ):

        """
        Generate SHAP explanation.

        Args:
            features:
                numpy array shape (131,)

        Returns:
            prediction,
            confidence,
            top contributing features
        """


        x = np.asarray(
            features,
            dtype=np.float32,
        )


        if x.shape[0] != 131:

            raise ValueError(
                f"Expected 131 features, got {x.shape[0]}"
            )



        x_tensor = torch.tensor(
            x,
            dtype=torch.float32,
        ).unsqueeze(0)



        # Background data for SHAP

        background = torch.zeros(
            (10, 131),
            dtype=torch.float32,
        )



        explainer = shap.DeepExplainer(
            self.model,
            background,
        )


        shap_values = explainer.shap_values(
            x_tensor
        )



        # Model prediction

        outputs = self.model(
            x_tensor
        )


        probabilities = torch.softmax(
            outputs,
            dim=1,
        )


        confidence, class_id = torch.max(
            probabilities,
            dim=1,
        )


        predicted_class = int(
            class_id.item()
        )



        # ----------------------------------
        # SHAP output handling
        # ----------------------------------

        if isinstance(shap_values, list):

            values = shap_values[predicted_class]

        else:

            values = shap_values



        values = np.asarray(
            values
        )


        # Remove batch dimension

        if values.ndim == 3:

            values = values[0]


        elif values.ndim == 2:

            values = values[0]



        # Multi-class output handling

        if values.ndim == 2:

            values = values[:, predicted_class]



        importance = []


        for name, value in zip(
            self.feature_names,
            values,
        ):

            importance.append(
                {
                    "feature": name,
                    "impact": float(value),
                }
            )



        importance = sorted(
            importance,
            key=lambda x: abs(x["impact"]),
            reverse=True,
        )



        return {

            "prediction_class":
                predicted_class,

            "confidence":
                float(confidence.item()),

            "top_features":
                importance[:10],
        }