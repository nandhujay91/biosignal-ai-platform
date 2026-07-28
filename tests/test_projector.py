import torch

from src.embeddings import ProjectionHead


def main():

    projector = ProjectionHead(
        hidden_dim=128,
        projection_dim=64,
    )

    x = torch.randn(
        4,
        320,
        128,
    )

    y = projector(x)

    print("Input Shape :", x.shape)
    print("Output Shape:", y.shape)


if __name__ == "__main__":
    main()
