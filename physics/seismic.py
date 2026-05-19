"""Seismic modeling utilities for synthetic data generation.

Provides functions for Ricker wavelet generation, reflectivity calculation,
and synthetic seismogram modeling from P-Impedance data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import fftconvolve  # type: ignore[import]

from config import DomainConfig, PhysicsConfig
from device import device_manager

if TYPE_CHECKING:
    from physics.physics import PhysicsState


def ricker_wavelet(
    f_peak: float, dt: float, length: float = PhysicsConfig.WAVELET_LENGTH
) -> np.ndarray:
    """
    Generate a Ricker (zero-phase) wavelet.

    Parameters
    ----------
    f_peak : float
        Peak frequency in Hz.
    dt : float
        Sampling interval in seconds.
    length : float, optional
        Total length of the wavelet in seconds. Default is PhysicsConfig.WAVELET_LENGTH.

    Returns
    -------
    np.ndarray
        1D array of wavelet samples.
    """
    t = np.arange(-length / 2, length / 2, dt)
    pi_sq = np.pi**2
    f_sq = f_peak**2
    t_sq = t**2

    # Ricker wavelet formula: (1 - 2*pi^2*f^2*t^2) * exp(-pi^2*f^2*t^2)
    term1 = 1 - 2 * pi_sq * f_sq * t_sq
    term2 = np.exp(-pi_sq * f_sq * t_sq)
    return term1 * term2


def normal_incidence_reflection(ip: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Calculate normal-incidence reflection coefficients from P-Impedance.

    Formula: R = (IP_{i+1} - IP_i) / (IP_{i+1} + IP_i)

    Parameters
    ----------
    ip : np.ndarray
        P-Impedance array.
    axis : int, optional
        Axis along which to calculate reflectivity (depth/time axis).
        Default is 0.

    Returns
    -------
    np.ndarray
        Reflectivity array of the same shape as input.
    """
    # Create slices for i and i+1 along the specified axis
    s_i = [slice(None)] * ip.ndim
    s_i_plus_1 = [slice(None)] * ip.ndim

    s_i[axis] = slice(0, -1)
    s_i_plus_1[axis] = slice(1, None)

    ip_i = ip[tuple(s_i)]
    ip_i_plus_1 = ip[tuple(s_i_plus_1)]

    rc = (ip_i_plus_1 - ip_i) / (ip_i_plus_1 + ip_i + DomainConfig.EPSILON)

    # Pad with a zero at the end of the axis to maintain shape
    padding_shape = list(ip.shape)
    padding_shape[axis] = 1
    padding = np.zeros(padding_shape, dtype=ip.dtype)

    return np.concatenate([rc, padding], axis=axis)


def apply_wavelet_to_ip(
    ip: np.ndarray,
    f_peak: float = PhysicsConfig.WAVELET_F_PEAK,
    dt: float = PhysicsConfig.WAVELET_DT,
    axis: int = 0,
) -> np.ndarray:
    """
    Applies a Ricker wavelet to P-Impedance data.

    Parameters
    ----------
    ip : np.ndarray
        P-Impedance data.
    f_peak : float, optional
        Wavelet peak frequency (Hz). Default is PhysicsConfig.WAVELET_F_PEAK.
    dt : float, optional
        Sampling interval (s). Default is PhysicsConfig.WAVELET_DT.
    axis : int, optional
        Axis along which to apply the wavelet (depth/time axis).
        Default is 0.

    Returns
    -------
    np.ndarray
        Synthetic seismogram of the same shape as IP.
    """
    # 1. Calculate reflectivity
    rc = normal_incidence_reflection(ip, axis=axis)

    # 2. Generate wavelet
    wavelet = ricker_wavelet(f_peak, dt)

    # 3. Convolve
    # Handle multidimensional input by applying along the specified axis
    def _conv1d(trace: np.ndarray) -> np.ndarray:
        return fftconvolve(trace, wavelet, mode="same")

    return np.apply_along_axis(_conv1d, axis=axis, arr=rc)


def torch_ricker_wavelet(
    f_peak: float,
    dt: float,
    length: float = PhysicsConfig.WAVELET_LENGTH,
) -> torch.Tensor:
    """
    Generate a Ricker (zero-phase) wavelet as a torch Tensor.

    Parameters
    ----------
    f_peak : float
        Peak frequency in Hz.
    dt : float
        Sampling interval in seconds.
    length : float, optional
        Total length in seconds. Default is PhysicsConfig.WAVELET_LENGTH.

    Returns
    -------
    torch.Tensor
        1D wavelet tensor.
    """
    dev = device_manager.device
    t = torch.arange(-length / 2, length / 2, dt, device=dev)
    pi_sq = torch.pi**2
    f_sq = f_peak**2
    t_sq = t**2

    term1 = 1 - 2 * pi_sq * f_sq * t_sq
    term2 = torch.exp(-pi_sq * f_sq * t_sq)
    return term1 * term2


def ip_to_reflectivity(ip: torch.Tensor, padding_value: torch.Tensor) -> torch.Tensor:
    """
    Calculate normal-incidence reflection coefficients from P-Impedance (Torch).

    Parameters
    ----------
    ip : torch.Tensor
        P-Impedance tensor (B, C, H, W). Assumes vertical axis is H.
    padding_value : float
        Value to pad the bottom of the RC tensor with.

    Returns
    -------
    torch.Tensor
        Reflectivity tensor of the same shape as input.
    """
    # Calculate RC along the H axis (dim 2)
    rc = (ip[:, :, 1:, :] - ip[:, :, :-1, :]) / (
        ip[:, :, 1:, :] + ip[:, :, :-1, :] + DomainConfig.EPSILON
    )

    # Pad at the bottom to maintain shape.
    # Broadcast the tensor to form a slice of shape (B, C, 1, W)
    pad_slice = torch.full_like(rc[:, :, :1, :], 0.0) + padding_value
    return torch.cat([rc, pad_slice], dim=2)


def resample_wavelet_to_depth(
    vp_mean: torch.Tensor,
    physics_state: "PhysicsState",
    dz_pixel: torch.Tensor | None = None,
) -> torch.Tensor:
    """Resample a time-domain wavelet to depth using a zero-sync grid_sample approach.

    Uses a fixed-size buffer and dynamic coordinate mapping to allow the
    entire operation to stay on the GPU, avoiding stalls from .item() calls.

    Parameters
    ----------
    vp_mean : torch.Tensor
        Average velocity (m/s).
    dz_pixel : torch.Tensor | None
        Depth sampling interval (m).
    physics_state : PhysicsState
        Object holding physical parameters and state.

    Returns
    -------
    torch.Tensor
        Depth-domain wavelet kernel. Shape (1, 1, fixed_size, 1).
    """
    device = physics_state.wavelet_t.device
    dtype = physics_state.wavelet_t.dtype

    if dz_pixel is None:
        dz_pixel = PhysicsConfig.DZ_PIXEL

    # Ensure everything is a tensor on the correct device
    v = torch.as_tensor(vp_mean, device=device, dtype=dtype).clamp(
        min=physics_state.vp_min
    )
    dz = torch.as_tensor(dz_pixel, device=device, dtype=dtype)
    wavelet_len = len(physics_state.wavelet_t)

    # Calculate the 'zoom' factor for the grid
    # How many depth pixels would the full time-wavelet occupy?
    # target_len_z = (Vp * T_total) / (2 * dz)
    total_time = (wavelet_len - 1) * physics_state.wavelet_dt
    target_len_z = (v * total_time) / (2 * dz)

    # Create a grid of depth indices centered at zero: [- (N-1)/2, (N-1)/2]
    # We use a fixed size to avoid recompilation and syncs.
    indices = torch.linspace(
        -(physics_state.fixed_kernel_size - 1) / 2,
        (physics_state.fixed_kernel_size - 1) / 2,
        steps=physics_state.fixed_kernel_size,
        device=device,
    )

    # Convert depth index to normalized time coordinate for grid_sample.
    # x_grid maps our indices to [-1, 1] relative to target_len_z (wavelet width in depth).
    grid_x = indices / (target_len_z / 2)

    # grid_sample expects (B, H, W, 2) or (B, C, D, H, W, 3).
    # For 1D sampling from (B, C, 1, L), we use (B, 1, fixed_size, 2).
    # The x-coordinate maps to the last dimension (L).
    grid = torch.stack([grid_x, torch.zeros_like(grid_x)], dim=-1).reshape(
        1, 1, physics_state.fixed_kernel_size, 2
    )

    # Input wavelet as (B, C, H, W) -> (1, 1, 1, L)
    w_input = physics_state.wavelet_t.view(1, 1, 1, -1)

    # Resample
    w_z: torch.Tensor = F.grid_sample(
        w_input,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    # Normalize by L1-norm (sum of absolute values) to ensure consistent convolution
    # gain across different sampling densities (dz).
    w_l1 = torch.sum(torch.abs(w_z.to(torch.float32)))
    w_norm = w_l1 + DomainConfig.EPSILON
    w_z = (w_z.to(torch.float32) / w_norm).to(dtype)  # type: ignore[assignment]

    # Return as (OutC, InC, H, W) -> (1, 1, fixed_size, 1)
    return w_z.view(1, 1, physics_state.fixed_kernel_size, 1)  # type: ignore[return-value]


def calculate_synthetic_seismic(
    ip_norm: torch.Tensor,
    vp_mean: torch.Tensor,
    dz_pixel: torch.Tensor,
    physics_state: PhysicsState,
    clip_output: bool = False,
) -> torch.Tensor:
    """Perform Geophysical Modeling to produce normalized synthetic seismic.

    Parameters
    ----------
    ip_norm : torch.Tensor
        Normalized P-Impedance in ``norm_range``. Shape (B, 1, Z, W).
    vp_mean : torch.Tensor
        Mean velocity (m/s) used for dynamic wavelet resampling.
    dz_pixel : torch.Tensor
        Depth sampling interval (m).
    physics_state : PhysicsState
        Object containing all physics-related buffers and logic.
    clip_output : bool, optional
        Whether to clamp the output to normalization_range. Must be False
        during training to prevent dead gradients. Defaults to False.

    Returns
    -------
    torch.Tensor
        Normalized synthetic seismic tensor in ``normalization_range``.
    """

    # 1. Denormalize IP from ``norm_range`` to physical units (e.g., GPa·m/s).
    ip_phys = (ip_norm - physics_state.norm_min) / (
        physics_state.norm_max - physics_state.norm_min + DomainConfig.EPSILON
    ) * (physics_state.ip_max - physics_state.ip_min) + physics_state.ip_min

    # 2. Compute Reflectivity (RC)
    rc = ip_to_reflectivity(ip_phys, padding_value=physics_state.padding_value)

    # 3. Resample Wavelet to Depth (Zero-Sync approach)
    wavelet_z = resample_wavelet_to_depth(
        vp_mean,
        physics_state,
        dz_pixel,
    )

    # 4. Synthetic Modeling (Convolution)
    padding_z = physics_state.fixed_kernel_size // 2
    synth = torch.nn.functional.conv2d(rc, wavelet_z, padding=(int(padding_z), 0))

    # Apply global physical gain calibration to align synthetic amplitudes with dataset scale
    synth = synth * PhysicsConfig.SEISMIC_GAIN

    # 5. Normalization using Dataset Statistics, then remap to normalization_range.
    synth = (synth - physics_state.seis_min) / (
        physics_state.seis_max - physics_state.seis_min + DomainConfig.EPSILON
    )
    lo = torch.min(physics_state.norm_min, physics_state.norm_max)
    hi = torch.max(physics_state.norm_min, physics_state.norm_max)
    synth = synth * (hi - lo) + lo
    
    if clip_output:
        return torch.clamp(synth, lo, hi)
    return synth
