import torch

from src.embeddings import (
    HierarchicalContrastiveLoss,
)


def main():

    loss_fn = HierarchicalContrastiveLoss()

    z1 = torch.randn(
        8,
        320,
        128,
    )

    z2 = torch.randn(
        8,
        320,
        128,
    )

    loss = loss_fn(
        z1,
        z2,
    )

    print(loss)


if __name__ == "__main__":

    main()
