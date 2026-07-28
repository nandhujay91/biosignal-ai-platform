from src.labeling import LabelGenerator


def main():

    features = [
        {
            "heart_rate": 75,
            "spo2": 98,
            "quality_score": 100,
        },
        {
            "heart_rate": 110,
            "spo2": 93,
            "quality_score": 85,
        },
        {
            "heart_rate": 140,
            "spo2": 85,
            "quality_score": 40,
        },
    ]


    generator = LabelGenerator()

    labels = generator.generate(
        features
    )

    print("Generated Labels:")
    print(labels)


    assert labels.tolist() == [
        0,
        1,
        2,
    ]

    print(
        "Label generation test passed successfully."
    )


if __name__ == "__main__":
    main()