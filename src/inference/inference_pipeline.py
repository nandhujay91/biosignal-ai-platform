from __future__ import annotations

import numpy as np
import torch

from src.embeddings.inference import TS2VecInference
from src.embeddings.ts2vec import TS2Vec
from src.features.feature_extractor import FeatureExtractor
from src.inference.predictor import BiosignalPredictor
from src.pipeline.signal_preprocessor import SignalPreprocessor
from src.windowing.window_generator import WindowGenerator


class BiosignalInferencePipeline:
    """
    End-to-end biosignal inference pipeline.

    Flow:
    Binary files
        ->
    Preprocessing
        ->
    Window generation
        ->
    TS2Vec embedding
        ->
    Feature extraction
        ->
    Fusion prediction
    """

    def __init__(
        self,
        model_dir: str = "artifacts/classifier",
    ) -> None:

        self.predictor = BiosignalPredictor(
            model_dir=model_dir,
        )

        self.feature_extractor = FeatureExtractor()

        self.ts2vec = TS2Vec(
            input_dim=8,
            hidden_dim=128,
            projection_dim=64,
            depth=8,
        )

        self.embedding_model = TS2VecInference(
            model=self.ts2vec,
        )

        self.embedding_model.load_model()

    def run(
        self,
        data_directory: str,
    ):

        # ---------------------------------
        # 1. Signal preprocessing
        # ---------------------------------

        signals = SignalPreprocessor.run(
            data_directory,
        )

        if "Ephy" not in signals:

            raise ValueError("Ephy signal is required for inference.")

        if "Oxym" not in signals:

            raise ValueError("Oxym signal is required for inference.")

        # ---------------------------------
        # 2. Generate temporal windows
        # ---------------------------------

        windows = WindowGenerator.generate(
            signals,
        )

        ephy_windows = windows["Ephy"]

        tensor = torch.tensor(
            ephy_windows,
            dtype=torch.float32,
        )

        # ---------------------------------
        # 3. Generate TS2Vec embedding
        # ---------------------------------

        embeddings = self.embedding_model.encode(
            tensor,
        )

        # Expected:
        # (batch, 128)

        embedding_vector = embeddings[0].numpy()

        # ---------------------------------
        # 4. Extract biosignal features
        # ---------------------------------

        features = self.feature_extractor.extract(
            ecg_signal=signals["Ephy"],
            oxym_signal=signals["Oxym"],
        )

        feature_vector = np.array(
            [
                features["heart_rate"],
                features["spo2"],
                features["quality_score"],
            ],
            dtype=np.float32,
        )

        # ---------------------------------
        # 5. Feature fusion
        # ---------------------------------

        fusion_vector = np.concatenate(
            [
                embedding_vector,
                feature_vector,
            ],
            axis=0,
        )

        if fusion_vector.shape[0] != 131:

            raise ValueError(f"Expected 131 features, got {fusion_vector.shape[0]}")

        # ---------------------------------
        # 6. Classification
        # ---------------------------------

        response = self.predictor.predict(
            fusion_vector,
        )

        return response
