from __future__ import annotations

import numpy as np
import pandas as pd

from src.drift.drift_monitor import DriftMonitor



FEATURE_COUNT = 131



def create_columns():

    return (
        [
            f"embedding_{i}"
            for i in range(128)
        ]
        +
        [
            "signal_feature_0",
            "signal_feature_1",
            "signal_feature_2",
        ]
    )



def create_reference_data():

    np.random.seed(42)


    data = np.random.normal(
        loc=0,
        scale=1,
        size=(
            500,
            FEATURE_COUNT
        )
    )


    return pd.DataFrame(
        data,
        columns=create_columns()
    )



def create_production_data():

    """
    Simulate real production drift
    """


    np.random.seed(100)


    data = np.random.normal(
        loc=1.5,
        scale=2,
        size=(
            500,
            FEATURE_COUNT
        )
    )


    return pd.DataFrame(
        data,
        columns=create_columns()
    )



def main():

    print(
        "Creating reference data..."
    )

    reference_data = create_reference_data()



    print(
        "Creating production data..."
    )

    current_data = create_production_data()



    print(
        "Running drift monitor..."
    )


    monitor = DriftMonitor(
        drift_threshold=0.3
    )


    result = monitor.check_drift(
        reference_data,
        current_data,
    )



    print(
        "\nDrift Result"
    )

    print(
        "----------------"
    )


    print(
        "Drift detected:",
        result["drift_detected"]
    )


    print(
        "Drift score:",
        result["drift_score"]
    )


    print(
        "Threshold:",
        result["threshold"]
    )


    print(
        "Action:",
        result["action"]
    )


    print(
        "\nDrift monitoring test completed."
    )



if __name__ == "__main__":

    main()
