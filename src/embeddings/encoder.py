from __future__ import annotations

import torch
from torch import nn


class SamePadConv1D(nn.Module):
    """
    1D convolution with SAME padding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:

        super().__init__()

        padding = ((kernel_size - 1) * dilation) // 2

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.conv(x)


class ResidualBlock(nn.Module):
    """
    Dilated residual block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        self.conv1 = SamePadConv1D(
            in_channels,
            out_channels,
            kernel_size,
            dilation,
        )

        self.conv2 = SamePadConv1D(
            out_channels,
            out_channels,
            kernel_size,
            dilation,
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

        # Important for MyPy:
        # Conv1d and Identity are both nn.Module
        self.residual: nn.Module

        if in_channels != out_channels:

            self.residual = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
            )

        else:

            self.residual = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        residual = self.residual(x)

        out = self.conv1(x)

        out = self.relu(out)

        out = self.dropout(out)

        out = self.conv2(out)

        out = out + residual

        out = self.relu(out)

        return out


class DilatedResidualEncoder(nn.Module):
    """
    Stack of residual blocks.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        depth: int = 8,
    ) -> None:

        super().__init__()

        layers: list[nn.Module] = []

        in_channels = input_dim

        for i in range(depth):

            dilation = 2**i

            layers.append(
                ResidualBlock(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    dilation=dilation,
                )
            )

            in_channels = hidden_dim

        self.network = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.network(x)


class TS2VecEncoder(nn.Module):
    """
    Encoder backbone used by TS2Vec.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        depth: int = 8,
    ) -> None:

        super().__init__()

        self.encoder = DilatedResidualEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            depth=depth,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Input:
            (batch, length, channels)

        Output:
            (batch, length, hidden_dim)
        """

        x = x.transpose(
            1,
            2,
        )

        x = self.encoder(x)

        x = x.transpose(
            1,
            2,
        )

        return x
