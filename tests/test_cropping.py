import torch

from src.embeddings import RandomTemporalCrop


def main():

    x = torch.randn(
        4,
        320,
        8,
    )

    cropped = RandomTemporalCrop.crop(
        x,
        min_crop_ratio=0.5,
    )

    print("Input Shape :", x.shape)
    print("Output Shape:", cropped.shape)

    print(
        "Crop Length:",
        cropped.shape[1],
    )


if __name__ == "__main__":

    main()
