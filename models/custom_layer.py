"""Custom neural network layers used by the generator and discriminator.

This module implements building blocks such as :class:`ConvBlock`, SPADE
normalization and SPADE-based generator blocks.
"""

from typing import List, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .palette import PALETTE_ONE_HOT_4_TANH


class ConvBlock(nn.Sequential):
    """Convolutional block with Conv2D, InstanceNorm, and LeakyReLU.

    A standard building block for both generator and discriminator networks,
    combining convolution, instance normalization, and activation.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Size of the convolutional kernel.
    padding : int
        Amount of padding to add.
    stride : int
        Stride of the convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        stride: int,
    ) -> None:
        """Initialize a ConvBlock sequential module."""
        super().__init__()

        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
        )
        self.add_module("norm", nn.InstanceNorm2d(out_channels, affine=True))
        self.add_module("LeakyRelu", nn.LeakyReLU(0.2, inplace=True))


class SPADE(nn.Module):
    """Spatially-Adaptive Denormalization (SPADE).

    Instead of using standard batch/instance normalization where the scaling
    (gamma) and shifting (beta) parameters are learned scalars, SPADE computes
    these parameters as a function of an external conditioning tensor (e.g.,
    well locations or seismic data). This allows the normalization to preserve
    spatial semantics that would otherwise be lost during standard normalization.
    """

    def __init__(
        self,
        norm_nc: int,
        cond_nc: int,
        hidden_nc: int = 64,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        self.norm = nn.InstanceNorm2d(norm_nc, affine=True)

        padding = kernel_size // 2

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(cond_nc, hidden_nc, kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mlp_gamma = nn.Conv2d(
            hidden_nc, norm_nc, kernel_size=kernel_size, padding=padding
        )
        self.mlp_beta = nn.Conv2d(
            hidden_nc, norm_nc, kernel_size=kernel_size, padding=padding
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(x)

        if cond.shape[2:] != x.shape[2:]:
            cond = F.interpolate(
                cond, size=x.shape[2:], mode="bilinear", align_corners=True
            )

        actv = self.mlp_shared(cond)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        return normalized * (1 + gamma) + beta


class SPADEConvBlock(nn.Module):
    """Convolutional block with SPADE normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        kernel_size: int,
        padding: int,
        stride: int,
        spade_hidden: int = 64,
    ) -> None:
        super().__init__()

        self.spade = SPADE(in_channels, cond_channels, spade_hidden, kernel_size)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.spade(x, cond)
        x = self.activation(x)
        x = self.conv(x)
        return x


class SPADEGenerator(nn.Module):
    """SPADE-based generator block for the coarsest scale."""

    def __init__(
        self,
        num_layer: int,
        kernel_size: int,
        padding_size: int,
        num_features: int,
        min_num_features: int,
        output_channels: int,
        input_channels: int,
    ) -> None:
        super().__init__()

        self.init_conv = nn.Conv2d(
            input_channels,
            num_features,
            kernel_size=kernel_size,
            padding=padding_size,
        )

        self.spade_blocks = nn.ModuleList()

        curr_features = num_features
        for i in range(num_layer - 2):
            out_ch = max(int(num_features / pow(2, (i + 1))), min_num_features)
            self.spade_blocks.append(
                SPADEConvBlock(
                    curr_features, out_ch, input_channels, kernel_size, padding_size, 1
                )
            )
            curr_features = out_ch

        self.tail = nn.Sequential(
            nn.Conv2d(
                curr_features,
                output_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding_size,
            ),
            nn.Tanh(),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(cond)
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"[DEBUG] SPADEGenerator: NaN/Inf detected before LeakyReLU. cond range: [{cond.min().item()}, {cond.max().item()}]")
        x = F.leaky_relu(x, 0.2)

        for spade_block in self.spade_blocks:
            x = spade_block(x, cond)

        return self.tail(x)


class SPADEDiscriminator(nn.Module):
    """Convolutional discriminator with minibatch stddev.

    Implements a multi-scale critic that uses 'minibatch standard deviation'
    to prevent mode collapse by providing the discriminator with information
    about the variation within a batch of samples.
    """

    def __init__(
        self,
        num_features: int,
        min_num_features: int,
        num_layer: int,
        kernel_size: int,
        padding_size: int,
        input_channels: int,
        minibatch_stddev_group_size: int = 4,
        minibatch_stddev_epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        self.minibatch_stddev_group_size = minibatch_stddev_group_size
        self.minibatch_stddev_epsilon = minibatch_stddev_epsilon

        self.head = ConvBlock(
            input_channels, num_features, kernel_size, padding_size, 1
        )

        self.body = nn.Sequential(
            *[
                ConvBlock(
                    max(num_features // (2**i), min_num_features),
                    max(num_features // (2 ** (i + 1)), min_num_features),
                    kernel_size,
                    padding_size,
                    1,
                )
                for i in range(num_layer - 2)
            ]
        )

        output_channels = max(num_features // (2 ** (num_layer - 2)), min_num_features)
        self.tail = nn.Conv2d(
            output_channels + 1,
            1,
            kernel_size=kernel_size,
            stride=1,
            padding=padding_size,
        )

    def _append_minibatch_stddev(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()

            gathered: List[torch.Tensor] = [
                torch.zeros_like(x) for _ in range(world_size)
            ]
            dist.all_gather(gathered, x.detach())  # type: ignore
            gathered[rank] = x
            global_batch = torch.cat(gathered, dim=0)
        else:
            global_batch = x

        if global_batch.shape[0] == 1:
            stddev = torch.zeros((1, 1, 1, 1), device=x.device, dtype=x.dtype)
        else:
            stddev = global_batch.float().var(dim=0, unbiased=False, keepdim=True)
            stddev = torch.sqrt(stddev + self.minibatch_stddev_epsilon)
            stddev = stddev.mean(dim=1, keepdim=True).to(dtype=x.dtype)

        stddev = stddev.expand(batch_size, 1, x.shape[2], x.shape[3])
        return torch.cat([x, stddev], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.head(x)
        scores = self.body(scores)
        scores = self._append_minibatch_stddev(scores)
        scores = self.tail(scores)
        return scores


class FaciesQuantization(nn.Module):
    """Snaps continuous Tanh outputs to categorical one-hot vectors."""

    def __init__(self, num_classes: int = 4, temperature: float = 0.5) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature

        if num_classes == 4:
            palette = torch.tensor(PALETTE_ONE_HOT_4_TANH, dtype=torch.float32)
        else:
            palette = torch.eye(num_classes) * 2.0 - 1.0
        self.register_buffer("pure_colors", palette)

    def _sq_distances_nchw(self, x: torch.Tensor) -> torch.Tensor:
        colors = cast(torch.Tensor, self.pure_colors)  # (K, C)
        x_sq = (x * x).sum(dim=1, keepdim=True)
        # Reshape colors squared sum to (1, K, 1, 1) for correct NCHW broadcasting
        c_sq = colors.pow(2).sum(dim=1, keepdim=True).view(1, -1, 1, 1)
        dot = torch.einsum("bchw,kc->bkhw", x, colors)
        return x_sq + c_sq - 2 * dot

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply soft-argmax quantization to the input tensor."""

        colors = self.pure_colors
        distances = self._sq_distances_nchw(x)

        if not self.training:
            indices = distances.argmin(dim=1, keepdim=True)
            one_hot = torch.zeros_like(distances).scatter_(1, indices, 1.0)
            return torch.einsum("bkhw,kc->bchw", one_hot, colors)

        weights = F.softmax(-distances / self.temperature, dim=1)
        return torch.einsum("bkhw,kc->bchw", weights, colors)
