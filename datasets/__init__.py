"""Public dataset symbols used across the project.

This module exposes a small, stable public API for the `datasets` package
so other modules can import `Batch`, the framework-specific dataset and any
helper factories without reaching into submodules.
"""

from .dataset import TorchPyramidsDataset
from .pyramids_batch import Batch, IDataLoader, PyramidsBatch
from .utils import get_global_stats

__all__ = [
    "TorchPyramidsDataset",
    "Batch",
    "PyramidsBatch",
    "IDataLoader",
    "get_global_stats",
]
