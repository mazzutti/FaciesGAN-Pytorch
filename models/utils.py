from typing import Any

import torch
from apex.normalization import FusedLayerNorm  # type: ignore[import]

from config import DomainConfig
from device import device_manager
from enums import ChannelKey, DeviceType, LossFn, SplitKey
from options import TrainingOptions


def weights_init(m: torch.nn.Module) -> None:
    """Initialize neural network layer weights using normal distributions.

    Applies standard weight initialization strategies for convolutional and
    normalization layers. Conv2d layers use N(0, 0.02), while BatchNorm2d
    and InstanceNorm2d use N(1, 0.02) for weights and zero for biases.

    Skips normalization layers without affine parameters (e.g., InstanceNorm2d
    with affine=False).

    Parameters
    ----------
    m : nn.Module
        The neural network module to initialize.
    """
    if isinstance(m, torch.nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
    elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.InstanceNorm2d, FusedLayerNorm)):
        # Only initialize if affine parameters exist
        if getattr(m, "weight", None) is not None:
            m.weight.data.normal_(1.0, 0.02)
        if getattr(m, "bias", None) is not None:
            m.bias.data.fill_(0)


def calc_gradient_penalty(
    discriminator: torch.nn.Module,
    real_data: torch.Tensor,
    fake_data: torch.Tensor,
    LAMBDA: float,
) -> torch.Tensor:
    """Calculate gradient penalty for WGAN-GP training.

    Implements the gradient penalty term used in Wasserstein GAN with
    Gradient Penalty (WGAN-GP) to enforce the Lipschitz constraint.

    Parameters
    ----------
    discriminator : nn.Module
        The discriminator model.
    real_data : torch.Tensor
        Real data samples from the dataset.
    fake_data : torch.Tensor
        Generated fake data samples.
    LAMBDA : float
        Gradient penalty coefficient (typically 10.0).

    Returns
    -------
    torch.Tensor
        Calculated gradient penalty scalar value.
    """
    # Sample one alpha per batch item so every sample has its own interpolation
    # point — required by WGAN-GP for an unbiased gradient penalty estimate.
    dev = device_manager.device
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=dev).expand_as(real_data)
    interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
    disc_interpolates: torch.Tensor = discriminator(interpolates)

    gradients: torch.Tensor = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(1, dtype=disc_interpolates.dtype, device=dev).expand_as(
            disc_interpolates
        ),
        create_graph=True,
        only_inputs=True,
    )[0]

    # Compute the scale-invariant RMS gradient norm per sample.
    # Dividing by sqrt(C*H*W) normalises the L2 norm to a per-dimension RMS,
    # so the 1-Lipschitz target is the same regardless of spatial resolution.
    # Without this, the aggregate L2 norm grows as O(sqrt(n_dims)), causing
    # the GP to explode at large scales (e.g. 256×256) even when per-pixel
    # gradients are small.
    # Use reshape instead of view: tensor may be channels_last (non-contiguous).
    n_dims = float(gradients.shape[1] * gradients.shape[2] * gradients.shape[3])
    gradient_norms = gradients.reshape(batch_size, -1).norm(2, dim=1) / (n_dims**0.5) - 1  # type: ignore
    gradient_penalty = (gradient_norms**2).mean() * LAMBDA  # type: ignore

    return gradient_penalty  # type: ignore


def load(path: str) -> Any:
    """Load a torch file from disk, automatically mapping to the managed device.
    
    Uses the device_manager to ensure tensors are loaded onto the correct 
    local rank or CPU, regardless of where they were originally saved.
    """
    import os

    if not os.path.exists(path):
        return None
    return torch.load(path, map_location=device_manager.device, weights_only=False)


def interpolate(tensor: torch.Tensor, size: tuple[int, ...]) -> torch.Tensor:
    """Resize the input tensor to the given size using bilinear interpolation.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor to be resized.
    size : tuple[int, ...]
        The target spatial dimensions for the resized tensor (height, width).

    Returns
    -------
    torch.Tensor
        The resized tensor with the specified dimensions.
    """
    return torch.nn.functional.interpolate(
        tensor, size=size, mode="bilinear", align_corners=True
    )


def calculate_channels(options: TrainingOptions) -> dict[ChannelKey, int]:
    """Calculate input and output channel counts for models.

    Centralizes the logic that determines how many channels the generator
    outputs and the discriminator receives, based on facies and rock
    physics configuration.

    Returns
    -------
    dict[ChannelKey, int]
        Dictionary with keys:
        - 'facies': Number of facies classes.
        - 'rock_physics': Number of rock physics channels (3 if enabled).
        - 'generator_out': Total channels output by the generator.
        - 'discriminator_in': Total channels input to the discriminator.
    """
    num_facies_channels = options.num_facies_channels
    num_rp = 0
    if options.use_rock_physics:
        from enums import DataFiles

        num_rp = len(DataFiles.generator_output_rock_physics())

    # Total channels produced by the generator (e.g., 3 Facies + 3 Rock Physics = 6)
    total_out: int = num_facies_channels + num_rp

    # The noise tensor must accommodate the target output channels and any
    # conditioning channels (wells, seismic).
    noise_channels = (
        max(DomainConfig.NOISE_CHANNELS, total_out)
        + (num_facies_channels if options.use_wells else 0)
        + (1 if options.use_seismic else 0)
    )

    return {
        ChannelKey.FACIES: num_facies_channels,
        ChannelKey.ROCK_PHYSICS: num_rp,
        ChannelKey.GENERATOR_OUT: total_out,
        ChannelKey.DISCRIMINATOR_IN: total_out,
        ChannelKey.NOISE: noise_channels,
    }


def generate_noise(
    size: tuple[int, ...],
    num_samp: int = 1,
    scale: float = 1.0,
) -> torch.Tensor:
    """Generate a random noise tensor with specified dimensions.

    On CUDA the tensor is created in ``channels_last`` memory format so
    downstream convolutions (which use ``channels_last`` weights) avoid an
    implicit layout conversion on every forward call.

    Parameters
    ----------
    size : tuple[int, ...]
        Shape of the noise tensor as (channels, height, width).
    num_samp : int, optional
        Number of samples (batch size) to generate. Defaults to 1.
    scale : float, optional
        Scale factor applied to spatial dimensions (height, width).
        Dimensions are divided by scale. Defaults to 1.0.

    Returns
    -------
    torch.Tensor
        Random tensor sampled from standard normal distribution with shape
        (num_samp, channels, height/scale, width/scale).
    """
    dev = device_manager.device
    shape = (num_samp, size[0], *[round(s / scale) for s in size[1:]])
    if dev.type == DeviceType.CUDA and len(shape) == 4:
        # Allocate directly in channels_last layout — avoids a copy
        # compared to torch.randn(...).to(memory_format=channels_last).
        noise = torch.empty(shape, device=dev, memory_format=torch.channels_last)
        noise.normal_()
    else:
        noise = torch.randn(*shape, device=dev)
    if scale != 1:
        noise = interpolate(noise, size[1:])
    return noise


def split_facies_rp(
    tensor: torch.Tensor,
    num_facies: int,
    has_rp: bool = False,
    has_wells: bool = False,
    has_seismic: bool = False,
    channels_last: bool = False,
) -> dict[str, torch.Tensor | None]:
    """Split a multichannel tensor into its constituent components.

    Explicitly extracts components based on provided flags. The order is
    assumed to be [Facies | Rock Physics | Wells | Seismic].

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor with shape (B, C, H, W) or (B, H, W, C).
    num_facies : int
        Number of facies channels (e.g., 4).
    has_rp : bool, optional
        Whether the tensor contains the generator's rock physics channels
        (Ip, Is, Vp/Vs). Default: False.
    has_wells : bool, optional
        Whether the tensor contains well-conditioning channels (default: False).
        Uses num_facies as the well channel count.
    has_seismic : bool, optional
        Whether the tensor contains a 1-channel seismic attribute (default: False).
    channels_last : bool, optional
        If True, assumes (B, H, W, C) layout. Default is False.

    Returns
    -------
    dict[str, torch.Tensor | None]
        Dictionary with keys: 'facies', 'rock_physics', 'wells', 'seismic'.
    """
    if channels_last:
        dim = -1
    else:
        if tensor.ndim == 5:
            # (B, T, C, H, W)
            dim = 2
        elif tensor.ndim == 4:
            # (B, C, H, W)
            dim = 1
        else:
            # (C, H, W)
            dim = 0

    total_ch = tensor.shape[dim]

    res: dict[str, torch.Tensor | None] = {
        "facies": None,
        "rock_physics": None,
        "wells": None,
        "seismic": None,
    }

    def _slice(start: int, length: int) -> torch.Tensor:
        if channels_last:
            return tensor[..., start : start + length]
        if tensor.ndim == 5:
            # (B, T, C, H, W)
            return tensor[:, :, start : start + length, ...]
        if tensor.ndim == 4:
            # (B, C, H, W)
            return tensor[:, start : start + length, ...]
        # (C, H, W)
        return tensor[start : start + length, ...]

    # 1. Facies (always first)
    res["facies"] = _slice(0, num_facies)
    curr = num_facies

    # 2. Rock Physics (if flagged and present)
    num_rp = 0
    if has_rp:
        from enums import DataFiles

        num_rp = len(DataFiles.generator_output_rock_physics())

    if has_rp and total_ch >= curr + num_rp:
        res["rock_physics"] = _slice(curr, num_rp)
        curr += num_rp

    # 3. Wells (if flagged and present)
    if has_wells and total_ch >= curr + num_facies:
        res["wells"] = _slice(curr, num_facies)
        curr += num_facies

    # 4. Seismic (if flagged and present)
    if has_seismic and total_ch >= curr + 1:
        res["seismic"] = _slice(curr, 1)
        curr += 1

    return res


__all__ = ["SplitKey", "ChannelKey", "LossFn", "weights_init", "calc_gradient_penalty"]
