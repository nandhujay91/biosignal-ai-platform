from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient



MODEL_NAME = "BiosignalClassifier"



def main():

    mlflow.set_tracking_uri(
        "sqlite:///mlflow.db"
    )


    client = MlflowClient()



    print(
        "Checking registered model..."
    )


    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )


    if not versions:

        raise Exception(
            "No registered model found"
        )


    latest_version = max(
        int(v.version)
        for v in versions
    )


    print(
        "Latest version:",
        latest_version
    )



    print(
        "Promoting model to Production..."
    )


    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="production",
        version=latest_version,
    )


    print(
        "Model promoted successfully."
    )


    print(
        f"{MODEL_NAME} version {latest_version} is now Production"
    )



if __name__ == "__main__":
    main()