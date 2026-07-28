import torch

from src.embeddings import MetricTracker


def main():

    tracker = MetricTracker()

    tracker.update_train(torch.tensor(1.2))
    tracker.update_train(torch.tensor(0.8))
    tracker.update_train(torch.tensor(1.0))

    tracker.update_validation(torch.tensor(0.6))
    tracker.update_validation(torch.tensor(0.4))

    print("Average Train Loss :", tracker.average_train_loss)
    print("Average Validation Loss :", tracker.average_validation_loss)


if __name__ == "__main__":
    main()
