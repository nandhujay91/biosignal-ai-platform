import torch

from src.embeddings import TS2Vec


def main():

    model = TS2Vec(
        input_dim=8,
        hidden_dim=128,
        projection_dim=64,
        depth=8,
    )

    x = torch.randn(
        4,
        320,
        8,
    )

    loss = model(x)

    print("Loss:", loss.item())


if __name__ == "__main__":
    main()
