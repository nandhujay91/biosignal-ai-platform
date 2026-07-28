from pathlib import Path

import numpy as np
import pytest

from src.inference.inference_pipeline import (
    BiosignalInferencePipeline,
)


@pytest.fixture
def inference_pipeline():

    return BiosignalInferencePipeline(
        model_dir="artifacts/classifier",
    )


def test_model_artifacts_exist():

    model_dir = Path("artifacts/classifier")

    assert (model_dir / "classifier.pt").exists()

    assert (model_dir / "class_mapping.json").exists()

    assert (model_dir / "version.txt").exists()


def test_inference_pipeline_initialization(
    inference_pipeline,
):

    assert inference_pipeline.predictor is not None

    assert inference_pipeline.feature_extractor is not None

    assert inference_pipeline.embedding_model is not None


def test_fusion_vector_dimension():

    embedding = np.zeros(
        128,
        dtype=np.float32,
    )

    features = np.zeros(
        3,
        dtype=np.float32,
    )

    fusion_vector = np.concatenate(
        [
            embedding,
            features,
        ],
        axis=0,
    )

    assert fusion_vector.shape == (131,)


def test_prediction_output(
    inference_pipeline,
):

    sample_features = np.zeros(
        131,
        dtype=np.float32,
    )

    response = inference_pipeline.predictor.predict(sample_features)

    assert response.prediction in [
        "Normal",
        "Alert",
        "Critical",
        "Borderline",
    ]

    assert 0 <= response.confidence <= 1

    assert response.model_version == "v1.0.0"


@pytest.mark.integration
def test_end_to_end_biosignal_inference():

    data_path = Path("data/test")

    if not data_path.exists():

        pytest.skip("Test biosignal data not available.")

    pipeline = BiosignalInferencePipeline(
        model_dir="artifacts/classifier",
    )

    response = pipeline.run(str(data_path))

    assert response.prediction in [
        "Normal",
        "Alert",
        "Critical",
        "Borderline",
    ]

    assert 0 <= response.confidence <= 1
