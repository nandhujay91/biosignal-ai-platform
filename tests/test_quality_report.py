from src.entities import QualityReport


def test_quality_report():

    report = QualityReport(
        signal_name="Ephy",
        passed=True,
        score=98.7,
    )

    print(report)


if __name__ == "__main__":
    test_quality_report()
