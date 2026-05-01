"""Seismic modeling utilities for synthetic data generation.

Provides functions for Ricker wavelet generation, reflectivity calculation,
and synthetic seismogram modeling from P-Impedance data.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import fftconvolve  # type: ignore[import]


def ricker_wavelet(f_peak: float, dt: float, length: float = 0.128) -> np.ndarray:
    """
    Generate a Ricker (zero-phase) wavelet.

    Parameters
    ----------
    f_peak : float
        Peak frequency in Hz.
    dt : float
        Sampling interval in seconds.
    length : float, optional
        Total length of the wavelet in seconds. Default is 0.128.

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


def ip_to_reflectivity(ip: np.ndarray, axis: int = 0) -> np.ndarray:
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

    rc = (ip_i_plus_1 - ip_i) / (ip_i_plus_1 + ip_i + 1e-6)

    # Pad with a zero at the end of the axis to maintain shape
    padding_shape = list(ip.shape)
    padding_shape[axis] = 1
    padding = np.zeros(padding_shape, dtype=ip.dtype)

    return np.concatenate([rc, padding], axis=axis)


def apply_wavelet_to_ip(
    ip: np.ndarray, f_peak: float = 8.0, dt: float = 0.001, axis: int = 0
) -> np.ndarray:
    """
    Applies a Ricker wavelet to P-Impedance data.

    Parameters
    ----------
    ip : np.ndarray
        P-Impedance data.
    f_peak : float, optional
        Wavelet peak frequency (Hz). Default is 8.0.
    dt : float, optional
        Sampling interval (s). Default is 0.001.
    axis : int, optional
        Axis along which to apply the wavelet (depth/time axis).
        Default is 0.

    Returns
    -------
    np.ndarray
        Synthetic seismogram of the same shape as IP.
    """
    # 1. Calculate reflectivity
    rc = ip_to_reflectivity(ip, axis=axis)

    # 2. Generate wavelet
    wavelet = ricker_wavelet(f_peak, dt)

    # 3. Convolve
    # Handle multidimensional input by applying along the specified axis
    def _conv1d(trace: np.ndarray) -> np.ndarray:
        return fftconvolve(trace, wavelet, mode="same")

    return np.apply_along_axis(_conv1d, axis=axis, arr=rc)


def torch_ricker_wavelet(
    f_peak: float, dt: float, length: float = 0.128, device: torch.device | None = None
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
        Total length in seconds. Default is 0.128.
    device : torch.device, optional
        Target device for the tensor.

    Returns
    -------
    torch.Tensor
        1D wavelet tensor.
    """
    t = torch.arange(-length / 2, length / 2, dt, device=device)
    pi_sq = torch.pi**2
    f_sq = f_peak**2
    t_sq = t**2

    term1 = 1 - 2 * pi_sq * f_sq * t_sq
    term2 = torch.exp(-pi_sq * f_sq * t_sq)
    return term1 * term2


def torch_ip_to_reflectivity(ip: torch.Tensor) -> torch.Tensor:
    """
    Calculate normal-incidence reflection coefficients from P-Impedance (Torch).

    Parameters
    ----------
    ip : torch.Tensor
        P-Impedance tensor (B, C, H, W). Assumes vertical axis is H.

    Returns
    -------
    torch.Tensor
        Reflectivity tensor of the same shape as input.
    """
    # Calculate RC along the H axis (dim 2)
    rc = (ip[:, :, 1:, :] - ip[:, :, :-1, :]) / (
        ip[:, :, 1:, :] + ip[:, :, :-1, :] + 1e-6
    )

    # Pad with a zero at the bottom to maintain shape
    return F.pad(rc, (0, 0, 0, 1), mode="constant", value=0)


def resample_wavelet_to_depth(
    wavelet_t: torch.Tensor,
    vp_mean: float | torch.Tensor,
    dt: torch.Tensor,
    dz_pixel: float | torch.Tensor,
    vp_min: float | torch.Tensor = 2000.0,
    fixed_size: int = 255,
) -> torch.Tensor:
    """Resample a time-domain wavelet to depth using a zero-sync grid_sample approach.

    Uses a fixed-size buffer and dynamic coordinate mapping to allow the
    entire operation to stay on the GPU, avoiding stalls from .item() calls.

    Parameters
    ----------
    wavelet_t : torch.Tensor
        Time-domain wavelet (1D). Shape (L,).
    vp_mean : float or torch.Tensor
        Average velocity (m/s).
    dt : float
        Time sampling interval (s).
    dz_pixel : float or torch.Tensor
        Depth sampling interval (m).
    vp_min : float or torch.Tensor, optional
        Minimum velocity to clamp vp_mean (m/s). Default is 2000.0.
    fixed_size : int, optional
        Fixed output size for the kernel buffer. Default is 256.

    Returns
    -------
    torch.Tensor
        Depth-domain wavelet kernel. Shape (1, 1, fixed_size, 1).
    """
    device = wavelet_t.device
    dtype = wavelet_t.dtype

    # Ensure everything is a tensor on the correct device
    v = torch.as_tensor(vp_mean, device=device, dtype=dtype).clamp(min=vp_min)
    dz = torch.as_tensor(dz_pixel, device=device, dtype=dtype)
    wavelet_len = len(wavelet_t)

    # Calculate the 'zoom' factor for the grid
    # How many depth pixels would the full time-wavelet occupy?
    # target_len_z = (Vp * T_total) / (2 * dz)
    total_time = (wavelet_len - 1) * dt
    target_len_z = (v * total_time) / (2 * dz)

    # Create a grid of depth indices centered at zero: [- (N-1)/2, (N-1)/2]
    # We use a fixed size to avoid recompilation and syncs.
    indices = torch.linspace(
        -(fixed_size - 1) / 2, (fixed_size - 1) / 2, steps=fixed_size, device=device
    )

    # Convert depth index to normalized time coordinate for grid_sample.
    # x_grid maps our indices to [-1, 1] relative to target_len_z (wavelet width in depth).
    grid_x = indices / (target_len_z / 2)

    # grid_sample expects (B, H, W, 2) or (B, C, D, H, W, 3).
    # For 1D sampling from (B, C, 1, L), we use (B, 1, fixed_size, 2).
    # The x-coordinate maps to the last dimension (L).
    grid = torch.stack([grid_x, torch.zeros_like(grid_x)], dim=-1).reshape(
        1, 1, fixed_size, 2
    )

    # Input wavelet as (B, C, H, W) -> (1, 1, 1, L)
    w_input = wavelet_t.view(1, 1, 1, -1)

    # Resample
    w_z: torch.Tensor = F.grid_sample(
        w_input,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    # Normalize energy
    w_z = w_z / (torch.norm(w_z) + 1e-6)  # type: ignore[assignment]

    # Return as (OutC, InC, H, W) -> (1, 1, fixed_size, 1)
    return w_z.view(1, 1, fixed_size, 1)  # type: ignore[return-value]
