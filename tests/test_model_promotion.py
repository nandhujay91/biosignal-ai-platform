from __future__ import annotations

import mlflow

from mlflow.tracking import MlflowClient



MODEL_NAME = "BiosignalClassifier"



def get_latest_versions():

    client = MlflowClient()


    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )


    versions = sorted(
        versions,
        key=lambda x: int(x.version)
    )


    return versions



def promote_model(version):

    client = MlflowClient()


    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="production",
        version=str(version.version),
    )


    print(
        f"Version {version.version} promoted to production"
    )



def main():

    mlflow.set_tracking_uri(
        "sqlite:///mlflow.db"
    )


    print(
        "Checking model versions..."
    )


    versions = get_latest_versions()


    if len(versions) < 2:

        raise Exception(
            "Need at least Version 1 and Version 2"
        )



    current_model = versions[-2]

    candidate_model = versions[-1]



    print(
        "Current Production Candidate:",
        current_model.version
    )


    print(
        "New Candidate:",
        candidate_model.version
    )



    # Metrics from MLflow runs

    client = MlflowClient()


    current_run = client.get_run(
        current_model.run_id
    )


    candidate_run = client.get_run(
        candidate_model.run_id
    )



    current_accuracy = (
        current_run.data.metrics
        .get(
            "accuracy",
            0.0
        )
    )


    candidate_accuracy = (
        candidate_run.data.metrics
        .get(
            "accuracy",
            0.0
        )
    )



    print(
        "\nAccuracy comparison"
    )

    print(
        "Version",
        current_model.version,
        ":",
        current_accuracy
    )

    print(
        "Version",
        candidate_model.version,
        ":",
        candidate_accuracy
    )



    if candidate_accuracy > current_accuracy:

        print(
            "\nNew model is better"
        )


        promote_model(
            candidate_model
        )


    else:

        print(
            "\nKeeping current production model"
        )



if __name__ == "__main__":

    main()