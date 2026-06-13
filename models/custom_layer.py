"""Custom neural network layers used by the generator and discriminator.

This module implements building blocks such as :class:`ConvBlock`, SPADE
normalization and SPADE-based generator blocks. These helpers are
convenience modules that keep layer definitions and small forward
utilities close to the model implementations.

All public classes in this module are documented with NumPy-style
docstrings describing parameters and returns.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from config import DomainConfig
from device import device_manager


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
        """Initialize a ConvBlock sequential module.

        Parameters are documented on the class level.
        """
        super(ConvBlock, self).__init__()

        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,  # InstanceNorm2d(affine=True) has its own bias
            ),
        )
        self.add_module("norm", nn.InstanceNorm2d(out_channels, affine=True))
        self.add_module("LeakyRelu", nn.LeakyReLU(0.2, inplace=True))


class SPADE(nn.Module):
    """Spatially-Adaptive Denormalization (SPADE) layer.

    Modulates normalized feature maps using spatially-varying scale (gamma)
    and bias (beta) learned from a conditioning input (noise + wells).

    This allows the noise to have a stronger, more meaningful impact on the
    generation by learning how to transform the features based on the noise
    at each spatial location.

    Reference: Park et al., "Semantic Image Synthesis with Spatially-Adaptive
    Normalization", CVPR 2019.

    Parameters
    ----------
    norm_nc : int
        Number of channels in the feature map to be normalized.
    cond_nc : int
        Number of channels in the conditioning input (noise + wells).
    hidden_nc : int, optional
        Number of hidden channels in the SPADE mlp. Defaults to 64.
    kernel_size : int, optional
        Kernel size for convolutions. Defaults to 3.
    """

    def __init__(
        self,
        norm_nc: int,
        cond_nc: int,
        hidden_nc: int = 64,
        kernel_size: int = 3,
    ) -> None:
        """Initialize the SPADE normalization module.

        Parameters are documented on the class level.
        """
        super().__init__()  # type: ignore

        self.norm = nn.InstanceNorm2d(norm_nc, affine=True)

        padding = kernel_size // 2

        # Shared convolution for processing conditioning input
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(cond_nc, hidden_nc, kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Separate convolutions for gamma (scale) and beta (bias)
        self.mlp_gamma = nn.Conv2d(
            hidden_nc, norm_nc, kernel_size=kernel_size, padding=padding
        )
        self.mlp_beta = nn.Conv2d(
            hidden_nc, norm_nc, kernel_size=kernel_size, padding=padding
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply SPADE normalization.

        Parameters
        ----------
        x : torch.Tensor
            Feature map to normalize, shape (B, norm_nc, H, W).
        cond : torch.Tensor
            Conditioning input (noise + wells), shape (B, cond_nc, H', W').
            Will be resized to match x if needed.

        Returns
        -------
        torch.Tensor
            Modulated feature map with same shape as x.
        """
        # Normalize the input features
        normalized = self.norm(x)

        # Resize conditioning to match feature map size if needed
        if cond.shape[2:] != x.shape[2:]:
            cond = F.interpolate(
                cond, size=x.shape[2:], mode="bilinear", align_corners=True
            )

        # Generate spatially-varying gamma and beta from conditioning
        activation = self.mlp_shared(cond)
        gamma = self.mlp_gamma(activation)
        beta = self.mlp_beta(activation)

        # Apply modulation: out = gamma * normalized + beta
        return normalized * (1 + gamma) + beta


class SPADEConvBlock(nn.Module):
    """Convolutional block with SPADE normalization for noise-conditioned generation.

    Replaces BatchNorm with SPADE to allow noise to modulate features at each
    spatial location, enabling more diverse outputs from different noise inputs.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    cond_channels : int
        Number of conditioning channels (noise + wells).
    kernel_size : int
        Size of the convolutional kernel.
    padding : int
        Amount of padding to add.
    stride : int
        Stride of the convolution.
    spade_hidden : int, optional
        Hidden channels in SPADE mlp. Defaults to 64.
    """

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
        """Initialize the SPADEConvBlock used inside SPADEGenerator.

        Parameters are documented on the class level.
        """
        super().__init__()  # type: ignore

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
        """Forward pass with SPADE conditioning.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map.
        cond : torch.Tensor
            Conditioning input (noise + wells).

        Returns
        -------
        torch.Tensor
            Output feature map.
        """
        x = self.spade(x, cond)
        x = self.activation(x)
        x = self.conv(x)
        return x


class SPADEGenerator(nn.Module):
    """SPADE-based generator block for the coarsest scale.

    Uses SPADE normalization to inject noise into the generation process,
    allowing the network to learn how noise should modulate features at
    each spatial location. This produces more diverse outputs compared
    to simple concatenation.

    Parameters
    ----------
    num_layer : int
        Number of convolutional layers.
    kernel_size : int
        Size of convolutional kernels.
    padding_size : int
        Padding size for convolutions.
    stride : int
        Stride of the convolution.
    num_features : int
        Number of features in the first layer.
    min_num_features : int
        Minimum number of features.
    output_channels : int
        Number of output image channels (e.g., 3 for RGB facies).
    input_channels : int
        Number of conditioning channels (noise + wells) supplied to the
        generator (also used as SPADE conditioning channels).
    """

    def __init__(
        self,
        num_layer: int,
        kernel_size: int,
        padding_size: int,
        stride: int,
        num_features: int,
        min_num_features: int,
        output_channels: int,
        input_channels: int,
    ) -> None:
        """Initialize SPADEGenerator parameters and layers.

        Parameters are documented on the class level.
        """
        super().__init__()  # type: ignore

        self.init_conv = nn.Conv2d(
            input_channels,
            num_features,
            kernel_size=kernel_size,
            padding=padding_size,
        )

        # SPADE blocks for the body
        self.spade_blocks = nn.ModuleList()

        curr_features = num_features
        for i in range(num_layer - 2):
            out_ch = max(int(num_features / pow(2, (i + 1))), min_num_features)
            self.spade_blocks.append(
                SPADEConvBlock(
                    curr_features,
                    out_ch,
                    input_channels,
                    kernel_size,
                    padding_size,
                    stride,
                )
            )
            curr_features = out_ch

        # Final output layer with coordination block
        self.tail = nn.Sequential(
            ConvBlock(
                curr_features,
                curr_features,
                kernel_size=1,
                padding=0,
                stride=1,
            ),
            nn.Conv2d(
                curr_features,
                output_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding_size,
            ),
            nn.Tanh(),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """Generate output from conditioning input using SPADE modulation.

        Parameters
        ----------
        cond : torch.Tensor
            Conditioning input containing noise and well data,
            shape (B, cond_channels, H, W).

        Returns
        -------
                torch.Tensor
                    Generated facies image, shape (B, output_channels, H, W).
        """
        # Initial feature extraction from conditioning
        x = self.init_conv(cond)
        x = F.leaky_relu(x, 0.2)

        # Apply SPADE blocks - each block uses conditioning to modulate features
        for spade_block in self.spade_blocks:
            x = spade_block(x, cond)

        return self.tail(x)


class SPADEDiscriminator(nn.Module):
    """Convolutional discriminator with minibatch standard-deviation regularization.

    This discriminator is a standard convolutional encoder that appends a
    minibatch-standard-deviation feature map before the final conv to help the
    network detect mode collapse and encourage diversity. It is compatible with
    distributed training: when DDP is active the minibatch statistic is
    computed across the global batch gathered from all ranks.

    Parameters
    ----------
    num_features : int
        Number of features in the first convolutional layer.
    min_num_features : int
        Minimum number of features used when reducing channels.
    num_layer : int
        Number of convolutional layers.
    kernel_size : int
        Convolution kernel size.
    padding_size : int
        Padding applied to convolutions.
    stride : int
        Stride of the convolution.
    input_channels : int
        Number of input image channels.
    minibatch_stddev_group_size : int, optional
        Group size used to compute minibatch standard deviation. The
        implementation falls back to the full batch when the batch is
        smaller than this value. Default is 4.
    minibatch_stddev_epsilon : float, optional
        Numerical stability constant added before the square root.
        Default is DomainConfig.EPSILON.
    """

    def __init__(
        self,
        num_features: int,
        min_num_features: int,
        num_layer: int,
        kernel_size: int,
        padding_size: int,
        stride: int,
        input_channels: int,
        minibatch_stddev_group_size: int = 4,
        minibatch_stddev_epsilon: float = DomainConfig.EPSILON,
    ) -> None:
        """Initialize the convolutional discriminator.

        Parameters
        ----------
        num_features : int
            Number of features in the first convolutional layer.
        min_num_features : int
            Minimum number of features used when reducing channels.
        num_layer : int
            Number of convolutional layers.
        kernel_size : int
            Convolution kernel size.
        padding_size : int
            Padding applied to convolutions.
        input_channels : int
            Number of input image channels.
        minibatch_stddev_group_size : int, optional
            Group size used to compute minibatch standard deviation. The
            implementation falls back to the full batch when the batch is
            smaller than this value. Default is 4.
        minibatch_stddev_epsilon : float, optional
            Numerical stability constant added before the square root.
            Default is DomainConfig.EPSILON.
        """

        nn.Module.__init__(self)  # type: ignore
        self.minibatch_stddev_group_size = minibatch_stddev_group_size
        self.minibatch_stddev_epsilon = minibatch_stddev_epsilon

        self.head = ConvBlock(
            input_channels,
            num_features,
            kernel_size,
            padding_size,
            stride,
        )

        self.body = nn.Sequential(
            *[
                ConvBlock(
                    max(num_features // (2**i), min_num_features),
                    max(num_features // (2 ** (i + 1)), min_num_features),
                    kernel_size,
                    padding_size,
                    stride,
                )
                for i in range(num_layer - 2)
            ]
        )

        output_channels = max(num_features // (2 ** (num_layer - 2)), min_num_features)
        self.tail = nn.Conv2d(
            output_channels + 1,
            1,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding_size,
        )

    def _append_minibatch_stddev(self, x: torch.Tensor) -> torch.Tensor:
        """Append a minibatch standard-deviation feature map.

        The statistic is computed across the full global batch when DDP is
        active, then averaged over channels/spatial dimensions so the
        discriminator gets one extra map describing how much variation exists
        within the batch.
        """
        batch_size = x.shape[0]
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()

            gathered = [torch.zeros_like(x) for _ in range(world_size)]
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
        """Discriminate input facies images.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor containing facies images.

        Returns
        -------
        torch.Tensor
            Discrimination scores with shape ``(B, 1, H_out, W_out)``. The
            returned tensor is not reduced to a scalar so callers can compute
            patch-wise or global losses as required.
        """
        scores = self.head(x)
        scores = self.body(scores)
        scores = self._append_minibatch_stddev(scores)
        scores = self.tail(scores)
        return scores


class FaciesQuantization(nn.Module):
    """Quantize output to a small set of pure colors.

    During training this module performs a soft (differentiable) assignment
    of each pixel to a small palette of pure colors using a temperature-
    scaled softmax over negative squared distances. During evaluation, it
    performs a hard nearest-color lookup to produce discrete colors.

    The palette is registered as a buffer and expects generator outputs in
    the ``normalization_range`` (usually [-1, 1] for tanh output).

    By default, the palette contains 4 pure colors (K=4) registered under the
    buffer name ``pure_colors`` with shape ``(4, 3)``. Change the buffer if a
    different palette size is required.
    """

    temperature: torch.Tensor
    pure_colors: torch.Tensor

    def __init__(
        self,
        temperature: float = 0.5,
        normalization_range: tuple[float, ...] = (-1.0, 1.0),
    ) -> None:
        """Create a ColorQuantization module.

        Parameters
        ----------
        temperature : float, optional
            Softmax temperature used during training for soft assignments.
            Higher values produce softer (more differentiable) assignments
            enabling better gradient flow; lower values produce sharper
            (more discrete) outputs. Default is 0.5.
        normalization_range : tuple of float, optional
            Target range for the quantized colors (default: (-1.0, 1.0)).
        """
        super().__init__()  # type: ignore
        self.register_buffer(
            "temperature",
            torch.tensor(
                temperature,
                dtype=torch.float32,
                device=device_manager.device,
            ),
        )

        from models.palette import PALETTE_RGB

        lo, hi = float(min(normalization_range)), float(max(normalization_range))
        colors_rgb = torch.tensor(PALETTE_RGB, dtype=torch.float32)
        colors_norm = colors_rgb * (hi - lo) + lo

        self.register_buffer(
            "pure_colors",
            colors_norm.to(device=device_manager.device),
        )

    def _sq_distances_nchw(self, x: torch.Tensor) -> torch.Tensor:
        """Compute squared distances from each pixel to each palette color.

        Operates entirely in NCHW layout using ``einsum`` to avoid the
        two expensive ``permute().contiguous()`` round-trips that the
        previous implementation required.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Squared-distance tensor of shape ``(B, K, H, W)`` where *K*
            is the palette size.
        """
        colors = self.pure_colors  # (K, 3)
        # ||x||^2 per-pixel: (B, 1, H, W)
        x_sq = (x * x).sum(dim=1, keepdim=True)
        # ||c||^2 per-color: (K,) → (1, K, 1, 1) for broadcasting
        c_sq = colors.pow(2).sum(dim=1, keepdim=True).view(1, -1, 1, 1)
        # x·c per color: (B, K, H, W)
        dot = torch.einsum("bchw,kc->bkhw", x, colors)
        return x_sq + c_sq - 2 * dot

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize RGB output to pure colors.

        All computation stays in NCHW layout via ``einsum``, eliminating
        the two ``permute().contiguous()`` copies per forward pass that
        the original implementation paid.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 3, H, W) with values in [-1, 1].

        Returns
        -------
        torch.Tensor
            Quantized tensor with same shape.
        """
        colors = self.pure_colors  # (K, 3)
        distances = self._sq_distances_nchw(x)  # (B, K, H, W)

        if not self.training:
            # Hard quantization — one-hot argmin then einsum back to (B,3,H,W)
            indices = distances.argmin(dim=1, keepdim=True)  # (B, 1, H, W)
            one_hot = torch.zeros_like(distances).scatter_(1, indices, 1.0)
            return torch.einsum("bkhw,kc->bchw", one_hot, colors)

        # Soft (differentiable) quantization during training
        weights = F.softmax(-distances / self.temperature, dim=1)  # (B, K, H, W)
        return torch.einsum("bkhw,kc->bchw", weights, colors)
