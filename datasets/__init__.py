"""Public dataset symbols used across the project.

This module exposes the framework-specific dataset and any helper factories.
Common types like Batch, IDataLoader, and PyramidsBatch should be imported
directly from `typedefs`.
"""

from .dataset import PyramidsDataset
from .utils import get_global_stats

__all__ = [
    "PyramidsDataset",
    "get_global_stats",
]
