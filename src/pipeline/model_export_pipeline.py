from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from src.logger import logger


class ModelExportPipeline:
    """
    Export trained classifier into versioned model package.
    """

    def __init__(
        self,
        source_model: str = "artifacts/classifier/classifier_best.pt",
        export_dir: str = "artifacts/model/v1",
    ) -> None:

        self.source_model = Path(source_model)

        self.export_dir = Path(export_dir)

        self.export_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def run(self):

        logger.info("Starting model export.")

        # -------------------------------------------------
        # Copy trained model
        # -------------------------------------------------

        shutil.copy(
            self.source_model,
            self.export_dir / "classifier.pt",
        )

        # -------------------------------------------------
        # Class mapping
        # -------------------------------------------------

        class_mapping = {
            "0": "Normal",
            "1": "Alert",
            "2": "Critical",
        }

        with open(
            self.export_dir / "class_mapping.json",
            "w",
        ) as file:

            json.dump(
                class_mapping,
                file,
                indent=4,
            )

        # -------------------------------------------------
        # Model configuration
        # -------------------------------------------------

        config = {
            "model_type": "EmbeddingClassifier",
            "input_dim": 131,
            "num_classes": 3,
            "classes": [
                "Normal",
                "Alert",
                "Critical",
            ],
            "version": "v1",
        }

        with open(
            self.export_dir / "model_config.yaml",
            "w",
        ) as file:

            yaml.dump(
                config,
                file,
                sort_keys=False,
            )

        # -------------------------------------------------
        # Version file
        # -------------------------------------------------

        (self.export_dir / "version.txt").write_text("v1")

        # -------------------------------------------------
        # Export evaluation metrics
        # -------------------------------------------------

        metrics_source = Path("artifacts/reports/evaluation_metrics.json")

        if metrics_source.exists():

            shutil.copy(
                metrics_source,
                self.export_dir / "metrics.json",
            )

            logger.info("Evaluation metrics exported.")

        else:

            logger.warning("Evaluation metrics not found.")

        logger.info(f"Model exported to {self.export_dir}")


if __name__ == "__main__":

    pipeline = ModelExportPipeline()

    pipeline.run()
