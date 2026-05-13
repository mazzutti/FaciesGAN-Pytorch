"""Centralised NVIDIA Apex imports.

Apex is a hard dependency — import failure crashes immediately with a
clear error message.  This module re-exports the subset of Apex used
throughout FaciesGAN so every other module can ``from apex_utils import …``
without duplicating try/except logic.

Only the kernel-fused components that exist in modern Apex builds are
exported here: ``FusedAdam`` (optimizer) and ``FusedLayerNorm``
(normalization).  Legacy APIs such as ``apex.amp`` and ``apex.parallel``
were removed from Apex; PyTorch's native ``torch.amp`` is used instead.
"""

from apex.normalization import FusedLayerNorm  # type: ignore[import]
from apex.optimizers import FusedAdam  # type: ignore[import]
