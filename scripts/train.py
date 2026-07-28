"""
Training entry point.

Responsible for:
- Loading training pipeline
- Running model training
- Saving trained models
"""

from src.logger import logger
from src.pipeline.training_pipeline import TrainingPipeline


def main() -> None:
    """
    Execute training pipeline.
    """

    logger.info("Starting training execution.")

    pipeline = TrainingPipeline()

    pipeline.run()

    logger.info("Training execution completed successfully.")


if __name__ == "__main__":

    main()
