import torch

from src.embeddings import TemporalMasking


def main():

    x = torch.randn(
        2,
        20,
        8,
    )

    masked = TemporalMasking.random_mask(
        x,
        mask_ratio=0.25,
    )

    print("Input Shape :", x.shape)
    print("Output Shape:", masked.shape)

    zeros = (masked == 0).sum()

    print("Masked Values:", zeros.item())


if __name__ == "__main__":

    main()
