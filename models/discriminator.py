"""PyTorch multiscale Discriminator implementation for FaciesGAN.

This module provides a concrete `Discriminator` class that performs
multiscale critic evaluation on facies tensors. The implementation is
kept framework-specific and mirrors the generator-side structure with a
per-scale module list and progressive scale construction.
"""

from __future__ import annotations

from typing import Self, cast

import torch
import torch.nn as nn

from .custom_layer import SPADEDiscriminator


class Discriminator(nn.Module):
    """Convolutional critic for facies images (WGAN-GP compatible).

    The discriminator produces a single-channel feature map of scores; for a
    given input tensor of shape ``(B, C, H, W)`` the output shape is
    ``(B, 1, H_out, W_out)`` where ``H_out``/``W_out`` depend on padding and
    kernel sizes. Higher values indicate more-realistic patches.

    Parameters
    ----------
    num_layer : int
        Number of convolutional layers in the discriminator.
    kernel_size : int
        Size of convolutional kernels.
    padding_size : int
        Padding size for convolutions.
    input_channels : int
        Number of input facies channels.

    Attributes
    ----------
    discs : nn.ModuleList
        List of per-scale discriminator modules.
    """

    def __init__(
        self,
        num_layer: int,
        kernel_size: int,
        padding_size: int,
        input_channels: int,
    ) -> None:
        """Initialize the convolutional discriminator.

        Parameters
        ----------
        num_layer : int
            Number of convolutional layers in the discriminator.
        kernel_size : int
            Convolution kernel size.
        padding_size : int
            Padding applied to convolutions.
        input_channels : int
            Number of input image channels.
        """
        self.num_layer = num_layer
        self.kernel_size = kernel_size
        self.padding_size = padding_size
        self.input_channels = input_channels

        nn.Module.__init__(self)

        # Use nn.ModuleList so that .train()/.eval(), .to(device),
        # and .state_dict() propagate to all per-scale disc blocks.
        self.discs = nn.ModuleList()

    def __call__(self, scale: int, input_tensor: torch.Tensor) -> torch.Tensor:
        """Call the discriminator's forward method."""
        return nn.Module.__call__(self, scale, input_tensor)

    def eval(self) -> Self:
        """Set the module in evaluation mode.

        Returns
        -------
        Self
            The discriminator instance in evaluation mode.
        """
        return cast(Self, nn.Module.eval(self))

    def forward(self, scale: int, input_tensor: torch.Tensor) -> torch.Tensor:
        """Discriminate input tensor and return score map tensor."""
        return self.discs[scale](input_tensor)

    def create_scale(self, num_features: int, min_num_features: int) -> None:
        """Append a new per-scale block to the discriminator implementation.

        Parameters
        ----------
        num_features : int
            Number of features in the first convolutional layer of the block.
        min_num_features : int
            Minimum number of features used when reducing channels.
        """
        spade_disc = SPADEDiscriminator(
            num_layer=self.num_layer,
            kernel_size=self.kernel_size,
            padding_size=self.padding_size,
            num_features=num_features,
            min_num_features=min_num_features,
            input_channels=self.input_channels,
        )
        self.discs.append(spade_disc)
