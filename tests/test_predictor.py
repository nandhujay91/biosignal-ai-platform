from src.inference.predictor import BiosignalPredictor


def main():

    predictor = BiosignalPredictor()


    sample = [
        0.1
    ] * 131


    result = predictor.predict(
        sample
    )


    print(result)


    print(
        "Predictor test passed."
    )


if __name__ == "__main__":
    main()