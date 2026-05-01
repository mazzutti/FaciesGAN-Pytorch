"""Common type definitions and containers for multi-scale pyramids.

This module centralizes the :class:`Batch` named tuple and other core types
used by the datasets and data prefetching systems. By isolating these types,
we avoid circular dependencies between the main dataset module and other
packages like models or trainers.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch.utils.data import DataLoader


# Batch container for multi-scale pyramids
class Batch(NamedTuple):
    """Container for a training batch of multi-scale pyramids.

    Fields are per-scale tuples of tensors. `Batch` is a NamedTuple so it
    subclasses `tuple` and is compatible with PyTorch's default collate.
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

# Interface for data loader
IDataLoader = DataLoader[tuple[int, Batch] | Batch]
