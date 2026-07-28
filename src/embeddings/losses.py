import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalContrastiveLoss(nn.Module):
    """
    Hierarchical Contrastive Loss used in TS2Vec.

    Computes contrastive loss across multiple temporal resolutions.
    """

    def __init__(
        self,
        temperature: float = 0.2,
    ) -> None:

        super().__init__()

        self.temperature = temperature

    def _instance_contrastive_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Instance-level InfoNCE loss.

        Input:
            (batch, time, hidden)
        """

        batch, time, hidden = z1.shape

        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        z1 = z1.reshape(batch * time, hidden)
        z2 = z2.reshape(batch * time, hidden)

        logits = torch.matmul(z1, z2.T)

        logits /= self.temperature

        labels = torch.arange(
            logits.size(0),
            device=logits.device,
        )

        return F.cross_entropy(
            logits,
            labels,
        )

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Multi-scale hierarchical contrastive loss.
        """

        loss = 0.0

        levels = 0

        while z1.size(1) > 1:

            loss += self._instance_contrastive_loss(
                z1,
                z2,
            )

            z1 = F.avg_pool1d(
                z1.transpose(1, 2),
                kernel_size=2,
                stride=2,
            ).transpose(1, 2)

            z2 = F.avg_pool1d(
                z2.transpose(1, 2),
                kernel_size=2,
                stride=2,
            ).transpose(1, 2)

            levels += 1

        loss += self._instance_contrastive_loss(
            z1,
            z2,
        )

        levels += 1

        return loss / levels