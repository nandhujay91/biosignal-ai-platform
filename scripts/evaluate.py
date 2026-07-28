"""
Model evaluation entry point.

Responsible for:
- Loading trained model
- Running evaluation
- Generating metrics report
"""

from src.logger import logger
from src.pipeline.evaluation_pipeline import EvaluationPipeline


def main() -> None:
    """
    Execute evaluation pipeline.
    """

    logger.info("Starting model evaluation.")

    pipeline = EvaluationPipeline()

    metrics = pipeline.run()

    logger.info(f"Evaluation metrics: {metrics}")

    logger.info("Evaluation completed successfully.")


if __name__ == "__main__":

    main()
