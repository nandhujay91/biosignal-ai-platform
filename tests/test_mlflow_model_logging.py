from __future__ import annotations

import torch
import mlflow
import mlflow.pytorch

from src.classifier.classifier import EmbeddingClassifier



def main():

    print(
        "Loading model..."
    )


    # -----------------------------
    # Load model architecture
    # -----------------------------

    model = EmbeddingClassifier(
        input_dim=131,
        num_classes=3,
    )


    model_path = (
        "artifacts/model/v1/classifier.pt"
    )


    checkpoint = torch.load(
        model_path,
        map_location="cpu",
    )


    model.load_state_dict(
        checkpoint["model_state_dict"]
    )


    model.eval()


    print(
        "Model loaded successfully."
    )



    # -----------------------------
    # MLflow configuration
    # -----------------------------

    mlflow.set_tracking_uri(
        "sqlite:///mlflow.db"
    )


    mlflow.set_experiment(
        "biosignal_embedding_model"
    )



    # -----------------------------
    # Start MLflow run
    # -----------------------------

    with mlflow.start_run(
        run_name="biosignal_model_logging"
    ):


        # -----------------------------
        # Log parameters
        # -----------------------------

        mlflow.log_params(
            {
                "model": "TS2Vec + DNN",
                "embedding_dim": 128,
                "input_features": 131,
                "num_classes": 3,
            }
        )


        # -----------------------------
        # Log metrics
        # Replace later with real metrics
        # -----------------------------

        mlflow.log_metrics(
            {
                "accuracy": 0.95,
                "loss": 0.12,
            }
        )



        # -----------------------------
        # Example input
        # -----------------------------

        example_input = torch.randn(
            1,
            131,
        )



        # -----------------------------
        # Log PyTorch model
        # -----------------------------

        mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",
            input_example=example_input,
            serialization_format="pickle",
        )



        print(
            "Model artifact logged."
        )



    print(
        "MLflow model logging completed successfully."
    )



if __name__ == "__main__":

    main()
