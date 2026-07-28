from __future__ import annotations

import mlflow
import mlflow.pytorch


class MLflowTracker:
    """
    MLflow experiment tracking manager.
    """

    def __init__(
        self,
        experiment_name="biosignal_embedding_model",
        tracking_uri="sqlite:///mlflow.db",
    ):

        mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)

    def start_run(self):

        return mlflow.start_run()

    def log_parameters(
        self,
        params: dict,
    ):

        for key, value in params.items():

            mlflow.log_param(
                key,
                value,
            )

    def log_metrics(
        self,
        metrics: dict,
    ):

        for key, value in metrics.items():

            mlflow.log_metric(
                key,
                value,
            )

    def log_model(
        self,
        model,
    ):

        mlflow.pytorch.log_model(
            model,
            "model",
        )
