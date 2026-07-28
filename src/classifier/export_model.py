import json
import shutil
from pathlib import Path

import yaml

SOURCE_MODEL = "artifacts/classifier/classifier_best.pt"


OUTPUT_DIR = Path("artifacts/model/v1")


def export_model():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Copy trained model

    shutil.copy(
        SOURCE_MODEL,
        OUTPUT_DIR / "classifier.pt",
    )

    # Model configuration

    config = {
        "model_name": "biosignal_classifier",
        "version": "v1",
        "input_dimension": 131,
        "classes": 3,
        "architecture": "MLP",
        "embedding": "TS2Vec",
        "features": [
            "embedding_128",
            "heart_rate",
            "spo2",
            "quality_score",
        ],
    }

    with open(
        OUTPUT_DIR / "model_config.yaml",
        "w",
    ) as file:

        yaml.dump(
            config,
            file,
        )

    # Label mapping

    labels = {
        "0": "Normal",
        "1": "Alert",
        "2": "Critical",
    }

    with open(
        OUTPUT_DIR / "class_mapping.json",
        "w",
    ) as file:

        json.dump(
            labels,
            file,
            indent=4,
        )

    # Version

    with open(
        OUTPUT_DIR / "version.txt",
        "w",
    ) as file:

        file.write("v1")

    print("Model exported successfully.")


if __name__ == "__main__":
    export_model()
