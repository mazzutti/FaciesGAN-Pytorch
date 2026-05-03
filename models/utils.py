"""Model utilities and shared functions.

This module provides framework-specific helpers for interpolation, padding,
and channel bookkeeping used by the generator and discriminator.
"""

from typing import Any, cast

import torch
import torch.nn.functional as F

from config import DomainConfig, PhysicsConfig
from enums import FeatureKey, LossFunction
from options import TrainingOptions


def weights_init(m: torch.nn.Module) -> None:
    """Initialize network weights using standard GAN normal distributions.

    - Convolutions: mean=0.0, std=0.02
    - Normalization: mean=1.0, std=0.02, bias=0
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight = getattr(m, "weight", None)
        if isinstance(weight, torch.Tensor):
            torch.nn.init.normal_(weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1 or classname.find("InstanceNorm") != -1:
        weight = getattr(m, "weight", None)
        bias = getattr(m, "bias", None)
        if isinstance(weight, torch.Tensor):
            torch.nn.init.normal_(weight.data, 1.0, 0.02)
        if isinstance(bias, torch.Tensor):
            torch.nn.init.constant_(bias.data, 0)


def calc_gradient_penalty(
    netD: torch.nn.Module,
    real_data: torch.Tensor,
    fake_data: torch.Tensor,
    LAMBDA: float,
    device: torch.device,
) -> torch.Tensor:
    """Calculate WGAN-GP gradient penalty."""
    alpha = torch.rand(1, 1, device=device).expand_as(real_data)
    interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).requires_grad_(True)
    disc_interpolates: torch.Tensor | list[torch.Tensor] = netD(interpolates)

    # Handle both single-scale and multi-scale discriminator outputs
    if isinstance(disc_interpolates, list):
        disc_interpolates = disc_interpolates[-1]

    gradients: torch.Tensor = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(
            1, dtype=disc_interpolates.dtype, device=device
        ).expand_as(disc_interpolates),
        create_graph=True,
        only_inputs=True,
    )[0]

    # compute the L2 norm of gradients for each sample and apply the penalty
    gradients = cast(torch.Tensor, gradients.norm(2, dim=1) - 1)  # type: ignore
    gradient_penalty = (gradients**2).mean() * LAMBDA

    return gradient_penalty


def load(path: str, device: torch.device, as_type: type[Any] = torch.Tensor) -> Any:
    """Load a torch or numpy file from disk and move it to the target device."""
    import os

    if not os.path.exists(path):
        return None
    data = torch.load(path, map_location=device)
    return data


def interpolate(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resample tensor to target size using bilinear interpolation.

    Convenience wrapper around ``F.interpolate`` with standard settings
    for the FaciesGAN pyramid.
    """
    return F.interpolate(x, size=size, mode="bilinear", align_corners=True)


def calculate_channels(options: TrainingOptions) -> dict[str, int]:
    """Calculate input and output channel counts for models.

    Centralizes the logic that determines how many channels the generator
    outputs and the discriminator receives, based on facies and rock
    physics configuration.

    Returns
    -------
    dict[str, int]
        Dictionary with keys:
        - 'facies': Number of facies classes.
        - 'rock_physics': Number of rock physics channels (3 if enabled).
        - 'generator_out': Total channels output by the generator.
        - 'discriminator_in': Total channels input to the discriminator.
    """
    num_facies = options.num_facies_classes
    num_rp = 0
    if getattr(options, "use_rock_physics", False):
        from datasets.data_files import DataFiles

        num_rp = len(DataFiles.generator_output_rock_physics())

    return {
        "facies": num_facies,
        "rock_physics": num_rp,
        "generator_out": num_facies + num_rp,
        "discriminator_in": num_facies + num_rp,
    }


def calculate_noise_channels(options: TrainingOptions) -> int:
    """Calculate total input noise channels for the generator.

    Calculates the base noise channel count (max of noise_channels and
    output channels) and adds extra channels for conditioning (wells,
    seismic) when enabled.

    Parameters
    ----------
    options : TrainingOptions
        Training configuration containing hyperparams and feature flags.

    Returns
    -------
    int
        Total number of input noise channels.
    """
    counts = calculate_channels(options)
    total_out = counts["generator_out"]

    return (
        max(options.noise_channels, total_out)
        + (options.num_facies_classes if options.use_wells else 0)
        + (1 if options.use_seismic else 0)
    )


def total_variation_loss(img: torch.Tensor) -> torch.Tensor:
    """
    Compute Total Variation (TV) loss for a 4D tensor.

    Args:
        img: Tensor of shape [Batch, Canais, Profundidade, Traços]

    Returns:
        Scalar tensor representing the TV loss.
    """
    # Vertical differences (along depth axis)
    tv_z = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).mean()

    # Horizontal differences (along trace axis)
    tv_x = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).mean()

    return tv_z + tv_x


class NoiseBufferManager:
    """Manages pre-allocated noise buffers to minimize GPU allocations.

    Reuses cached zero-padded buffers for batched noise generation during
    training, eliminating thousands of small allocations per epoch.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.buffers: dict[tuple[int, ...], torch.Tensor] = {}

    def get_buffer(
        self,
        key: tuple[int, ...],
        shape: tuple[int, ...],
    ) -> torch.Tensor:
        """Retrieve or allocate a zeroed noise buffer for the given shape."""
        buf = self.buffers.get(key)
        if buf is None:
            buf = torch.empty(
                *shape,
                device=self.device,
                memory_format=torch.channels_last,
            ).zero_()
            self.buffers[key] = buf
        return buf

    def clear(self) -> None:
        """Clear all cached buffers."""
        self.buffers.clear()


def load_framework_state_dict(
    module: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> None:
    """Load a state dict into a module, handling DDP and compile prefixes.

    Strips or adds the ``_orig_mod.`` prefix as needed to ensure compatibility
    between compiled and uncompiled versions of the model.
    """
    from .facies_gan import unwrap_ddp

    target = unwrap_ddp(module)
    model_keys = set(target.state_dict().keys())
    ck_keys = set(state_dict.keys())
    prefix = "_orig_mod."

    # Checkpoint has prefix but model does not → strip it
    if any(k.startswith(prefix) for k in ck_keys) and not any(
        k.startswith(prefix) for k in model_keys
    ):
        state_dict = {
            (k[len(prefix) :] if k.startswith(prefix) else k): v
            for k, v in state_dict.items()
        }
    # Model has prefix but checkpoint does not → add it
    elif any(k.startswith(prefix) for k in model_keys) and not any(
        k.startswith(prefix) for k in ck_keys
    ):
        state_dict = {f"{prefix}{k}": v for k, v in state_dict.items()}

    target.load_state_dict(state_dict)


def generate_noise(
    shape: tuple[int, ...],
    num_samp: int = 1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generate a batch of Gaussian noise tensors with the specified shape."""
    return torch.randn(num_samp, *shape, device=device)


def split_facies_rp(
    tensor: torch.Tensor,
    num_facies: int,
    has_rp: bool = False,
    has_wells: bool = False,
    has_seismic: bool = False,
    channels_last: bool = False,
) -> dict[str, torch.Tensor | None]:
    """Split a multi-channel tensor into its constituent components.

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
    dict[FeatureKey, torch.Tensor | None]
        Dictionary with keys from FeatureKey (e.g., FeatureKey.FACIES).
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
        FeatureKey.FACIES: None,
        FeatureKey.ROCK_PHYSICS: None,
        FeatureKey.WELLS: None,
        FeatureKey.SEISMIC: None,
    }

    curr = 0

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
    res[FeatureKey.FACIES] = _slice(0, num_facies)
    curr = num_facies

    # 2. Rock Physics (if flagged and present)
    num_rp = 0
    if has_rp:
        from datasets.data_files import DataFiles

        num_rp = len(DataFiles.generator_output_rock_physics())

    if has_rp and total_ch >= curr + num_rp:
        res[FeatureKey.ROCK_PHYSICS] = _slice(curr, num_rp)
        curr += num_rp

    # 3. Wells (if flagged and present)
    if has_wells and total_ch >= curr + num_facies:
        res[FeatureKey.WELLS] = _slice(curr, num_facies)
        curr += num_facies

    # 4. Seismic (if flagged and present)
    if has_seismic and total_ch >= curr + 1:
        res[FeatureKey.SEISMIC] = _slice(curr, 1)
        curr += 1

    return res


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
    eps: float = DomainConfig.EPSILON,
) -> torch.Tensor:
    """Compute the multi-class Dice loss.

    Parameters
    ----------
    inputs : torch.Tensor
        Predicted probabilities or logits of shape (B, C, H, W).
    targets : torch.Tensor
        Ground truth one-hot encoded labels of shape (B, C, H, W).
    smooth : float, optional
        Smoothing factor to prevent zero division. Default is 1.0.
    eps : float, optional
        Small epsilon for numerical stability. Default is 1e-7.

    Returns
    -------
    torch.Tensor
        Scalar Dice loss.
    """
    # Reshape to (B, C, -1)
    inputs = inputs.flatten(2)
    targets = targets.flatten(2)

    intersection = (inputs * targets).sum(-1)
    cardinality = inputs.sum(-1) + targets.sum(-1)

    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth + eps)
    dice_loss = 1.0 - dice_score

    return dice_loss.mean()


def masked_cross_entropy(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute cross-entropy loss only at masked locations.

    Parameters
    ----------
    inputs : torch.Tensor
        Predicted logits of shape (B, C, H, W).
    targets : torch.Tensor
        Ground truth one-hot encoded labels of shape (B, C, H, W).
    mask : torch.Tensor
        Binary spatial mask of shape (B, 1, H, W).

    Returns
    -------
    torch.Tensor
        Scalar masked cross-entropy loss.
    """
    # Convert one-hot targets to class indices
    target_indices = torch.argmax(targets, dim=1)

    # Compute per-pixel loss without reduction
    loss = F.cross_entropy(inputs, target_indices, reduction="none")

    # Apply mask (mask is 1 where we have data)
    masked_loss = loss * mask.squeeze(1)

    # Average only over masked pixels
    denom = mask.sum()
    if denom > 0:
        return masked_loss.sum() / denom
    else:
        # Return a zero scalar that preserves gradients
        return inputs.sum() * 0.0


def rms_normalize(x: torch.Tensor, eps: float = DomainConfig.EPSILON) -> torch.Tensor:
    """Apply Root-Mean-Square (RMS) normalization to a tensor."""
    return x / (torch.sqrt(torch.mean(x**2)) + eps)


def calculate_synthetic_seismic(
    ip_norm: torch.Tensor,
    wavelet_t: torch.Tensor,
    dt_wavelet: torch.Tensor,
    dz_pixel: float | torch.Tensor,
    vp_mean: torch.Tensor,
    vp_min: torch.Tensor,
    ip_min: torch.Tensor,
    ip_max: torch.Tensor,
    seis_min: torch.Tensor,
    seis_max: torch.Tensor,
    fixed_kernel_size: int = PhysicsConfig.FIXED_KERNEL_SIZE,
) -> torch.Tensor:
    """Perform Geophysical Modeling to produce normalized synthetic seismic.

    Parameters
    ----------
    ip_norm : torch.Tensor
        Normalized P-Impedance in [-1, 1] range. Shape (B, 1, Z, W).
    wavelet_t : torch.Tensor
        Base time-domain wavelet.
    dt_wavelet : float
        Time sampling interval (s).
    dz_pixel : float or torch.Tensor
        Depth sampling interval (m).
    vp_mean : torch.Tensor
        Mean velocity (m/s) used for dynamic wavelet resampling.
    vp_min : torch.Tensor
        Minimum velocity value for Vp clamping (m/s).
    ip_min, ip_max : torch.Tensor
        Min/Max values for Ip denormalization.
    seis_min, seis_max : torch.Tensor
        Min/Max values for Seismic normalization.
    fixed_kernel_size : int, optional
        Fixed size for the convolution kernel. Default is PhysicsConfig.FIXED_KERNEL_SIZE.

    Returns
    -------
    torch.Tensor
        Normalized synthetic seismic tensor in [-1, 1] range.
    """
    from physics.seismic import resample_wavelet_to_depth, torch_ip_to_reflectivity

    # Ensure all stats are tensors on the correct device
    ip_min = torch.as_tensor(ip_min, device=ip_norm.device, dtype=ip_norm.dtype)
    ip_max = torch.as_tensor(ip_max, device=ip_norm.device, dtype=ip_norm.dtype)
    vp_min = torch.as_tensor(vp_min, device=ip_norm.device, dtype=ip_norm.dtype)
    dz_pixel = torch.as_tensor(dz_pixel, device=ip_norm.device, dtype=ip_norm.dtype)
    seis_min = torch.as_tensor(seis_min, device=ip_norm.device, dtype=ip_norm.dtype)
    seis_max = torch.as_tensor(seis_max, device=ip_norm.device, dtype=ip_norm.dtype)

    # 1. Denormalize to Physical Units ([-1, 1] -> [min, max])
    ip_phys = ((ip_norm + 1) / 2) * (ip_max - ip_min) + ip_min

    # 2. Compute Reflectivity (RC)
    rc = torch_ip_to_reflectivity(ip_phys)

    # 3. Resample Wavelet to Depth (Zero-Sync approach)
    wavelet_z = resample_wavelet_to_depth(
        wavelet_t,
        vp_mean,
        dt_wavelet,
        dz_pixel,
        vp_min=vp_min,
        fixed_size=fixed_kernel_size,
    )

    # 4. Synthetic Modeling (Convolution)
    padding_z = fixed_kernel_size // 2
    synth = F.conv2d(rc, wavelet_z, padding=(padding_z, 0))

    # 5. Normalization using Dataset Statistics -> [-1, 1]
    synth = 2.0 * (synth - seis_min) / (seis_max - seis_min + DomainConfig.EPSILON) - 1.0
    return synth.clamp(-1.0, 1.0)


def calculate_physics_loss(
    gen_ip_norm: torch.Tensor,
    real_seismic: torch.Tensor,
    wavelet_t: torch.Tensor,
    dt_wavelet: torch.Tensor,
    dz_pixel: float | torch.Tensor,
    vp_mean: torch.Tensor,
    vp_min: torch.Tensor,
    vp_max: torch.Tensor,
    ip_min: torch.Tensor,
    ip_max: torch.Tensor,
    seis_min: torch.Tensor,
    seis_max: torch.Tensor,
    loss_fn: str = LossFunction.HUBER,
    fixed_kernel_size: int = PhysicsConfig.FIXED_KERNEL_SIZE,
) -> torch.Tensor:
    """Calculate Geophysical Consistency Loss (Physics Loss).

    Parameters
    ----------
    gen_ip_norm : torch.Tensor
        Generated P-Impedance in [-1, 1] range. Shape (B, 1, Z, W).
    real_seismic : torch.Tensor
        Real seismic data at the current scale.
    wavelet_t : torch.Tensor
        Base time-domain wavelet.
    dt_wavelet : torch.Tensor
        Time sampling interval (s).
    dz_pixel : float or torch.Tensor
        Depth sampling interval (m).
    vp_mean : torch.Tensor
        Mean velocity (m/s) used to deform the wavelet.
    vp_min : torch.Tensor
        Minimum velocity value for Vp clamping (m/s).
    vp_max : torch.Tensor
        Maximum velocity value for Vp clamping (m/s).
    ip_min, ip_max : torch.Tensor
        Min/Max values for Ip denormalization.
    seis_min, seis_max : torch.Tensor
        Min/Max values for Seismic normalization.
    loss_fn : str, optional
        Loss function to use ('huber' or 'mse'). Default is 'huber'.
    fixed_kernel_size : int, optional
        Fixed size for the convolution kernel to avoid recompilation. Default is PhysicsConfig.FIXED_KERNEL_SIZE.

    Returns
    -------
    torch.Tensor
        Scalar physics loss.
    """
    # 1. Generate Synthetic Seismic
    synth = calculate_synthetic_seismic(
        gen_ip_norm,
        wavelet_t,
        dt_wavelet,
        dz_pixel,
        vp_mean,
        vp_min,
        ip_min,
        ip_max,
        seis_min,
        seis_max,
        fixed_kernel_size=fixed_kernel_size,
    )

    # 2. Match shapes if necessary (e.g. if seismic_pyramid has different Z)
    if synth.shape != real_seismic.shape:
        synth = F.interpolate(
            synth, size=(real_seismic.shape[2], real_seismic.shape[3])
        )

    # 3. RMS Normalization for Waveform Comparison
    synth_norm = rms_normalize(synth)
    real_seismic_norm = rms_normalize(real_seismic)

    if loss_fn == LossFunction.HUBER:
        loss = F.huber_loss(synth_norm, real_seismic_norm)
    else:
        loss = F.mse_loss(synth_norm, real_seismic_norm)
    return loss
