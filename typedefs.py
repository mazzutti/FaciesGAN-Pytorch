"""Common type definitions for the FaciesGAN project.

This module centralizes type aliases used for static analysis and runtime
type checking to ensure consistency across the codebase.
"""

from __future__ import annotations

from pathlib import Path
from typing import IO, Any, NamedTuple

import torch
from torch.utils.data import DataLoader

# Filesystem paths or stream-like objects
FileLike = str | Path | IO[Any]


# Batch container for multiscale pyramids
class Batch(NamedTuple):
    """Container for a training batch of multiscale pyramids.

    Fields are per-scale tuples of tensors. `Batch` is a NamedTuple so it
    subclasses `tuple` and is compatible with PyTorch's default collate.

    Attributes
    ----------
    facies : tuple[torch.Tensor, ...]
        Per-scale facies tensors (one tensor per pyramid scale).
    wells : tuple[torch.Tensor, ...]
        Per-scale well conditioning tensors (one tensor per scale).
        Empty tuple if wells are not used.
    masks : tuple[torch.Tensor, ...]
        Per-scale mask tensors indicating well locations (one per scale).
        Empty tuple if masks are not used.
    seismic : tuple[torch.Tensor, ...]
        Per-scale seismic conditioning tensors (one per scale).
        Empty tuple if seismic is not used.
    """

    facies: tuple[torch.Tensor, ...]
    wells: tuple[torch.Tensor, ...]
    masks: tuple[torch.Tensor, ...]
    seismic: tuple[torch.Tensor, ...]


PyramidsBatch = tuple[
    torch.Tensor,  # indices
    dict[int, torch.Tensor],  # facies
    dict[int, torch.Tensor],  # wells
    dict[int, torch.Tensor],  # masks
    dict[int, torch.Tensor],  # seismic
]
"""Output type returned by TorchDataPrefetcher after batch preparation.

A tuple of five elements:

    - indices: Sample indices tensor of shape (batch_size,).
    - facies: Dict mapping scale index to facies tensor of shape (N, C, H, W).
    - wells: Dict mapping scale index to well tensor of shape (N, C, H, W).
    - masks: Dict mapping scale index to mask tensor of shape (N, 1, H, W).
    - seismic: Dict mapping scale index to seismic tensor of shape (N, C, H, W).
"""

# Raw batch yielded by the DataLoader (either just the Batch or (index_tensor, Batch))
RawBatch = tuple[torch.Tensor, Batch] | Batch

# Interface for data loader
IDataLoader = DataLoader[RawBatch]
"""Type alias for a PyTorch DataLoader that yields RawBatch items.

Used as a type hint for data loaders that are compatible with both standard
iteration and index-aware prefetching systems.
"""
