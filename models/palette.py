"""Centralized color palette definitions for the FaciesGAN project.

This module provides a single source of truth for the facies color palettes
used across the codebase, including RGB [0, 1], Tanh RGB [-1, 1], and
One-Hot representations.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from enums import FaciesClass

# ── Facies Class Names ───────────────────────────────────────────────
FACIES_NAMES: dict[FaciesClass, str] = {
    FaciesClass.FLOODPLAIN: "Floodplain",
    FaciesClass.POINT_BAR: "Point bar",
    FaciesClass.CHANNEL: "Channel",
    FaciesClass.BOUNDARY: "Boundary",
}

# ── Standard RGB Palette [0, 1] ──────────────────────────────────────
# High-contrast palette for visualization (TensorBoard, Matplotlib)
PALETTE_RGB: list[list[float]] = [
    [0.0, 0.0, 0.0],  # 0: Floodplain (Black)
    [1.0, 0.0, 0.0],  # 1: Point bar (Red)
    [0.0, 0.0, 1.0],  # 2: Channel (Blue)
    [0.0, 1.0, 0.0],  # 3: Boundary (Green)
]

# ── Tanh Normalized Palette [-1, 1] ─────────────────────────────────
# Palette used by the GAN and Triton kernels in normalized RGB space
PALETTE_TANH: list[list[float]] = [
    [-1.0, -1.0, -1.0],  # 0: Black
    [ 1.0, -1.0, -1.0],  # 1: Red
    [-1.0, -1.0,  1.0],  # 2: Blue
    [-1.0,  1.0, -1.0],  # 3: Green
]

# ── One-Hot Tanh Palette [-1, 1] ────────────────────────────────────
# Identity-like palette where each class has its own high channel.
# Used when the model outputs N channels (one per facies).
PALETTE_ONE_HOT_4_TANH: list[list[float]] = [
    [ 1.0, -1.0, -1.0, -1.0],
    [-1.0,  1.0, -1.0, -1.0],
    [-1.0, -1.0,  1.0, -1.0],
    [-1.0, -1.0, -1.0,  1.0],
]

# Flattened (K*C,) = 16 values for optimized kernels (One-Hot mode)
PALETTE_ONE_HOT_4_TANH_FLAT: list[float] = [c for color in PALETTE_ONE_HOT_4_TANH for c in color]

# Squared norms ||c||² for each color (all equal to 4.0 for one-hot 4-class)
PALETTE_ONE_HOT_4_TANH_SQ_NORM: list[float] = [4.0, 4.0, 4.0, 4.0]

# Flattened (K*C,) = 12 values for optimized kernels (RGB mode)
PALETTE_TANH_FLAT: list[float] = [c for color in PALETTE_TANH for c in color]

# Squared norms ||c||² for each color (all equal to 3.0 for this palette)
PALETTE_TANH_SQ_NORM: list[float] = [3.0, 3.0, 3.0, 3.0]


def get_palette_tensor(
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    tanh: bool = False,
    one_hot: bool = False,
) -> torch.Tensor:
    """Return the palette as a PyTorch tensor.

    Parameters
    ----------
    device : torch.device, optional
        Target device for the tensor.
    dtype : torch.dtype, optional
        Data type for the tensor (default: float32).
    tanh : bool, optional
        If True, return the [-1, 1] palette. Otherwise [0, 1].
    one_hot : bool, optional
        If True, return the 4-class one-hot palette.

    Returns
    -------
    torch.Tensor
        Palette tensor.
    """
    if one_hot:
        p = PALETTE_ONE_HOT_4_TANH if tanh else np.eye(4).tolist()
    else:
        p = PALETTE_TANH if tanh else PALETTE_RGB
    return torch.tensor(p, dtype=dtype, device=device)


def get_palette_numpy(tanh: bool = False, one_hot: bool = False) -> NDArray[np.float32]:
    """Return the palette as a NumPy array.

    Parameters
    ----------
    tanh : bool, optional
        If True, return the [-1, 1] palette. Otherwise [0, 1].
    one_hot : bool, optional
        If True, return the 4-class one-hot palette.

    Returns
    -------
    NDArray[np.float32]
        Palette array.
    """
    if one_hot:
        p = PALETTE_ONE_HOT_4_TANH if tanh else np.eye(4).tolist()
    else:
        p = PALETTE_TANH if tanh else PALETTE_RGB
    return np.array(p, dtype=np.float32)
