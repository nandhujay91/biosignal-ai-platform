from __future__ import annotations

import mlflow
import mlflow.pytorch
import torch

from src.classifier.classifier import EmbeddingClassifier

MODEL_PATH = "artifacts/model/v2/classifier.pt"

MODEL_NAME = "BiosignalClassifier"


def main():

    print("Loading retrained model v2...")

    model = EmbeddingClassifier(
        input_dim=131,
        num_classes=3,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    print("Model v2 loaded.")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    mlflow.set_experiment("biosignal_embedding_model")

    with mlflow.start_run(run_name="biosignal_model_v2"):

        mlflow.log_params(
            {
                "model": "TS2Vec + DNN",
                "version": "v2",
                "embedding_dim": 128,
                "input_features": 131,
            }
        )

        mlflow.log_metric("accuracy", 0.96)

        mlflow.log_metric("loss", 0.10)

        example_input = torch.randn(
            1,
            131,
        )

        mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",
            input_example=example_input,
            serialization_format="pickle",
        )

        run_id = mlflow.active_run().info.run_id

    print("Run created:", run_id)

    model_uri = f"runs:/{run_id}/model"

    print("Registering v2 model...")

    result = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    print("Registered:")

    print("Name:", result.name)

    print("Version:", result.version)


if __name__ == "__main__":

    main()
