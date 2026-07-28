"""
Inference entry point.

Responsible for:
- Loading biosignal inference pipeline
- Processing raw biosignal files
- Generating embeddings
- Running classifier prediction
- Returning prediction response
"""

from pathlib import Path

from src.inference.inference_pipeline import (
    BiosignalInferencePipeline,
)
from src.logger import logger


def main() -> None:
    """
    Execute end-to-end biosignal inference.
    """

    logger.info("Starting biosignal inference.")

    model_directory = Path("artifacts/classifier")

    data_directory = Path("data/test")

    pipeline = BiosignalInferencePipeline(
        model_dir=str(model_directory),
    )

    prediction = pipeline.run(
        data_directory=str(data_directory),
    )

    logger.info(f"Prediction result: {prediction}")


if __name__ == "__main__":

    main()
