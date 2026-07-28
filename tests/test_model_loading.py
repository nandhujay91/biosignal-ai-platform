from __future__ import annotations

from pathlib import Path

from src.inference.predictor import BiosignalPredictor


def main():

    print("Testing model artifacts...")


    model_dir = Path(
        "artifacts/model/v1"
    )


    required_files = [
        "classifier.pt",
        "class_mapping.json",
        "model_config.yaml",
        "metrics.json",
        "version.txt",
    ]


    for file_name in required_files:

        file_path = model_dir / file_name

        assert file_path.exists(), (
            f"Missing artifact: {file_name}"
        )

        print(
            f"Found: {file_name}"
        )


    print("\nLoading predictor...")


    predictor = BiosignalPredictor(
        model_dir=str(model_dir)
    )


    assert predictor.model is not None

    assert predictor.class_mapping is not None

    assert predictor.version is not None


    print(
        "Model loaded successfully."
    )


    print(
        "Model Version:",
        predictor.version,
    )


    print(
        "\nModel loading test passed."
    )


if __name__ == "__main__":
    main()
