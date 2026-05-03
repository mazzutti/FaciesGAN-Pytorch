from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray


def norm(x: torch.Tensor) -> torch.Tensor:
    """Normalize tensor from [0, 1] to [-1, 1] range.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor with values in [0, 1] range.

    Returns
    -------
    torch.Tensor
        Normalized tensor with values clamped to [-1, 1].
    """
    out = (x - 0.5) * 2
    return out.clamp(-1, 1)


def denorm(
    tensor: torch.Tensor | NDArray[np.float32], ceiling: bool = False
) -> torch.Tensor | NDArray[np.float32]:
    """Denormalize tensor from [-1, 1] to [0, 1] range.

    Parameters
    ----------
    tensor : torch.Tensor | NDArray[np.float32]
        Input tensor or array with values in [-1, 1] range.
    ceiling : bool, optional
        Whether to set all positive values to 1. Currently not implemented.
        Defaults to False.

    Returns
    -------
    torch.Tensor | NDArray[np.float32]
        Denormalized tensor or array with values clamped to [0, 1].
    """
    if isinstance(tensor, torch.Tensor):
        tensor = (tensor + 1.0) / 2.0
        tensor = tensor.clamp(0, 1)
    else:
        tensor = (tensor + np.float32(1.0)) / np.float32(2.0)
        tensor = np.clip(tensor, 0, 1).astype(np.float32, copy=False)
    return tensor
