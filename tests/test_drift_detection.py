from __future__ import annotations

import os
import numpy as np
import pandas as pd

from src.drift.drift_detector import DriftDetector



FEATURE_COUNT = 131



def create_feature_columns():

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

    """
    Training dataset distribution.

    Represents:
    - TS2Vec embeddings (128)
    - Signal features (3)

    Shape:
        100 samples x 131 features
    """


    np.random.seed(42)


    data = np.random.normal(
        loc=0,
        scale=1,
        size=(
            100,
            FEATURE_COUNT
        ),
    )


    return pd.DataFrame(
        data,
        columns=create_feature_columns()
    )



def create_current_data():

    """
    Production dataset distribution.

    Simulates production drift:
    - shifted mean
    - increased variance
    """


    np.random.seed(100)


    data = np.random.normal(
        loc=0.5,
        scale=1.2,
        size=(
            100,
            FEATURE_COUNT
        ),
    )


    return pd.DataFrame(
        data,
        columns=create_feature_columns()
    )



def main():


    print(
        "Creating reference dataset..."
    )


    reference_data = create_reference_data()



    print(
        "Creating production dataset..."
    )


    current_data = create_current_data()



    print(
        "Reference shape:",
        reference_data.shape
    )


    print(
        "Current shape:",
        current_data.shape
    )



    print(
        "Running drift detection..."
    )


    detector = DriftDetector()



    result = detector.run(
        reference_data=reference_data,
        current_data=current_data,
    )



    print(
        "Drift report generated."
    )



    os.makedirs(
        "reports",
        exist_ok=True
    )



    report_path = (
        "reports/drift_report.html"
    )


    result.save_html(
        report_path
    )


    print(
        "Saved:",
        report_path
    )



if __name__ == "__main__":
    main()
