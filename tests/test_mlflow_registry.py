from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient



MODEL_NAME = "BiosignalClassifier"



def main():

    mlflow.set_tracking_uri(
        "sqlite:///mlflow.db"
    )


    client = MlflowClient()



    # Get latest run

    experiment = client.get_experiment_by_name(
        "biosignal_embedding_model"
    )


    runs = client.search_runs(
        experiment_ids=[
            experiment.experiment_id
        ],
        order_by=[
            "attributes.start_time DESC"
        ],
        max_results=1,
    )


    latest_run = runs[0]


    run_id = latest_run.info.run_id


    print(
        "Latest Run:",
        run_id
    )



    model_uri = (
        f"runs:/{run_id}/model"
    )


    print(
        "Registering model..."
    )


    result = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )


    print(
        "Model Registered"
    )


    print(
        "Name:",
        result.name
    )


    print(
        "Version:",
        result.version
    )



if __name__ == "__main__":
    main()
