from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from src.classifier.classifier import EmbeddingClassifier
from src.config import InferenceConfig
from src.inference.response import PredictionResponse
from src.validation.input_validator import InputValidator

from src.monitoring.metrics import (
    PREDICTION_COUNT,
    PREDICTION_LABELS,
    CONFIDENCE_SCORE,
    INFERENCE_TIME,
)



class BiosignalPredictor:
    """
    Production inference service.

    Loads trained classifier
    and performs prediction with:
    - configuration management
    - input validation
    - confidence handling
    - risk assessment
    - prometheus monitoring
    """



    def __init__(
        self,
        model_dir: str | None = None,
    ) -> None:


        # Load configuration

        self.config = InferenceConfig()


        if model_dir is None:

            model_dir = (
                self.config.model_directory
            )


        self.model_dir = Path(
            model_dir
        )


        self.model_path = (
            self.model_dir
            / "classifier.pt"
        )


        self.mapping_path = (
            self.model_dir
            / "class_mapping.json"
        )


        self.version_path = (
            self.model_dir
            / "version.txt"
        )


        self.class_mapping = self._load_mapping()

        self.version = self._load_version()

        self.model = self._load_model()



    def _load_mapping(self):

        with open(
            self.mapping_path,
            "r",
        ) as file:

            return json.load(file)



    def _load_version(self):

        return self.version_path.read_text().strip()



    def _load_model(self):

        model = EmbeddingClassifier(
            input_dim=self.config.input_dim,
            num_classes=self.config.num_classes,
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




    @torch.no_grad()
    def predict(
        self,
        features,
    ) -> PredictionResponse:


        # Start latency timer

        start_time = time.time()



        # ----------------------------------
        # Input validation
        # ----------------------------------

        InputValidator.validate(
            features
        )



        # ----------------------------------
        # Tensor conversion
        # ----------------------------------

        tensor = torch.tensor(
            features,
            dtype=torch.float32,
        )


        tensor = tensor.unsqueeze(0)



        # ----------------------------------
        # Model inference
        # ----------------------------------

        outputs = self.model(
            tensor
        )


        probabilities = torch.softmax(
            outputs,
            dim=1,
        )


        confidence, prediction = torch.max(
            probabilities,
            dim=1,
        )


        confidence_value = float(
            confidence.item()
        )


        prediction_index = prediction.item()



        # ----------------------------------
        # Class prediction
        # ----------------------------------

        predicted_label = self.class_mapping[
            str(prediction_index)
        ]



        required_confidence = (
            self.config.confidence_thresholds[
                predicted_label
            ]
        )



        if confidence_value < required_confidence:

            label = "Borderline"

        else:

            label = predicted_label



        # ----------------------------------
        # Risk assessment
        # ----------------------------------

        if label == "Critical":

            risk_level = "High"

            recommended_action = (
                "Immediate review required"
            )


        elif label == "Alert":

            risk_level = "Medium"

            recommended_action = (
                "Monitor patient condition"
            )


        elif label == "Borderline":

            risk_level = "Medium"

            recommended_action = (
                "Review signal quality and patient status"
            )


        else:

            risk_level = "Low"

            recommended_action = (
                "Continue normal monitoring"
            )



        # ----------------------------------
        # Prometheus metrics
        # ----------------------------------

        PREDICTION_COUNT.inc()


        PREDICTION_LABELS.labels(
            prediction=label
        ).inc()


        CONFIDENCE_SCORE.observe(
            confidence_value
        )


        INFERENCE_TIME.observe(
            time.time() - start_time
        )



        return PredictionResponse(
            prediction=label,
            confidence=confidence_value,
            risk_level=risk_level,
            recommended_action=recommended_action,
            model_version=self.version,
        )