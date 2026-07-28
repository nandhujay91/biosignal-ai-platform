from src.tracking.mlflow_tracker import MLflowTracker


def main():

    tracker = MLflowTracker()

    with tracker.start_run():

        tracker.log_parameters(
            {
                "model": "TS2Vec + DNN",
                "embedding_dim": 128,
                "input_features": 131,
            }
        )

        tracker.log_metrics(
            {
                "accuracy": 0.95,
                "loss": 0.12,
            }
        )

    print("MLflow tracking test passed.")


if __name__ == "__main__":
    main()
