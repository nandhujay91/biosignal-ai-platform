import torch

from src.embeddings.encoder import TS2VecEncoder


def main():

    model = TS2VecEncoder(
        input_dim=8,
        hidden_dim=128,
        depth=8,
    )

    x = torch.randn(
        4,
        1280,
        8,
    )

    y = model(x)

    print("Input :", x.shape)

    print("Output:", y.shape)


if __name__ == "__main__":

    main()