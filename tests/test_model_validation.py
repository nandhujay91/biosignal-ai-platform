from __future__ import annotations

from src.validation.model_validator import ModelValidator


def main():

    print("Starting model validation...")

    validator = ModelValidator(minimum_accuracy=0.90)

    # Existing production model v1

    old_accuracy = 0.95
    old_loss = 0.12

    # New candidate model v2

    new_accuracy = 0.96
    new_loss = 0.10

    model_path = "artifacts/model/v2/classifier.pt"

    print("Checking model:", model_path)

    result = validator.validate(
        old_accuracy=old_accuracy,
        new_accuracy=new_accuracy,
        old_loss=old_loss,
        new_loss=new_loss,
        model_path=model_path,
    )

    print("\nValidation Result")

    print("----------------")

    print("Approved:", result["approved"])

    print("Decision:", result["decision"])

    print("\nChecks:")

    for name, value in result["checks"].items():

        print(name, ":", value)

    print("\nModel validation completed.")


if __name__ == "__main__":

    main()
