"""Discriminator module used by FaciesGAN.

Provides a multi-layer convolutional discriminator with a small API that
returns per-pixel critic scores (suitable for WGAN-style losses).
"""

from typing import Self

import torch
import torch.nn as nn

from .custom_layer import SPADEDiscriminator


class Discriminator(nn.Module):
    """Convolutional critic for facies images (WGAN-GP compatible).

    The discriminator produces a single-channel feature map of scores; for a
    given input tensor of shape ``(B, C, H, W)`` the output shape is
    ``(B, 1, H_out, W_out)``. Higher values indicate more-realistic patches.
    """

    def __init__(
        self,
        num_layer: int,
        kernel_size: int,
        padding_size: int,
        input_channels: int,
    ) -> None:
        """Initialize the convolutional discriminator."""
        super().__init__()

        # Base configuration logic (inlined)
        self.num_layer = num_layer
        self.kernel_size = kernel_size
        self.padding_size = padding_size
        self.input_channels = input_channels

        # Use nn.ModuleList so that .train()/.eval(), .to(device),
        # and .state_dict() propagate to all per-scale disc blocks.
        self.discs = nn.ModuleList()

    def eval(self) -> Self:
        """Set the module in evaluation mode."""
        return super().eval()

    def forward(self, scale: int, input_tensor: torch.Tensor) -> torch.Tensor:
        """Discriminate input tensor and return score map tensor."""
        return self.discs[scale](input_tensor)

    def create_scale(self, num_features: int, min_num_features: int) -> None:
        """Append a new per-scale block to the discriminator pyramid."""
        spade_disc = SPADEDiscriminator(
            num_layer=self.num_layer,
            kernel_size=self.kernel_size,
            padding_size=self.padding_size,
            num_features=num_features,
            min_num_features=min_num_features,
            input_channels=self.input_channels,
        )
        self.discs.append(spade_disc)
