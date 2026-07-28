import torch

from src.embeddings import LearningRateScheduler


def main():

    model = torch.nn.Linear(10, 2)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
    )

    scheduler = LearningRateScheduler(
        optimizer=optimizer,
        epochs=10,
    )

    print(f"Initial LR: {scheduler.get_lr():.8f}")

    for epoch in range(10):

        # Dummy forward pass
        x = torch.randn(4, 10)
        y = model(x)

        # Dummy loss
        loss = y.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # Update weights first
        scheduler.step()  # Then update learning rate

        print(
            f"Epoch {epoch + 1:2d} | "
            f"Loss: {loss.item():.6f} | "
            f"LR: {scheduler.get_lr():.8f}"
        )


if __name__ == "__main__":
    main()
