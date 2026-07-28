from __future__ import annotations

import mlflow

from mlflow.tracking import MlflowClient



MODEL_NAME = "BiosignalClassifier"



def get_model_metric(version):

    """
    Get accuracy metric from MLflow run.
    """

    client = MlflowClient()


    run_id = version.run_id


    run = client.get_run(
        run_id
    )


    accuracy = run.data.metrics.get(
        "accuracy",
        0.0
    )


    return accuracy



def compare_models():


    mlflow.set_tracking_uri(
        "sqlite:///mlflow.db"
    )


    client = MlflowClient()



    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )


    if len(versions) < 2:

        raise Exception(
            "Need at least two model versions"
        )



    versions = sorted(
        versions,
        key=lambda x: int(x.version)
    )


    v1 = versions[-2]

    v2 = versions[-1]



    v1_accuracy = get_model_metric(
        v1
    )


    v2_accuracy = get_model_metric(
        v2
    )



    print(
        "Version 1 Accuracy:",
        v1_accuracy
    )


    print(
        "Version 2 Accuracy:",
        v2_accuracy
    )



    if v2_accuracy > v1_accuracy:


        decision = (
            "Promote Version 2"
        )


    else:


        decision = (
            "Keep Version 1"
        )



    print(
        "\nDecision:"
    )

    print(
        decision
    )



def main():

    compare_models()



if __name__ == "__main__":

    main()