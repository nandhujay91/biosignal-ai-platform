import torch

from src.embeddings import TS2VecAugmentations


def main():

    x = torch.randn(
        4,
        320,
        8,
    )

    view1, view2 = TS2VecAugmentations.generate_views(
        x,
        crop_ratio=0.5,
        mask_ratio=0.2,
    )

    print("Input Shape :", x.shape)
    print("View 1 Shape:", view1.shape)
    print("View 2 Shape:", view2.shape)

    print("Views Equal :", torch.equal(view1, view2))


if __name__ == "__main__":

    main()
