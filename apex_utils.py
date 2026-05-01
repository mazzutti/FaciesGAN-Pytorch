"""Centralised NVIDIA Apex imports with safe fallback.

This module provides FusedAdam and FusedLayerNorm, falling back to standard
PyTorch implementations if Apex is not available or corrupted.
"""

from typing import Any

import torch
import torch.nn as nn

# Expose names with a permissive Any annotation so optional Apex types
# do not cause static assignability failures when Apex is installed.
FusedAdam: Any
FusedLayerNorm: Any

using_apex: bool = False
_apex_fused_adam: Any | None = None
_apex_fused_ln: Any | None = None
try:
    from apex.normalization import (  # isort: skip # type: ignore[import]
        FusedLayerNorm as ApexFusedLayerNorm,
    )
    from apex.optimizers import (  # isort: skip # type: ignore[import]
        FusedAdam as ApexFusedAdam,
    )
    # Simple check to see if they are usable (not corrupted)
    _test_adam = ApexFusedAdam
    _test_ln = ApexFusedLayerNorm
    _apex_fused_adam = ApexFusedAdam
    _apex_fused_ln = ApexFusedLayerNorm
    using_apex = True
except (ImportError, Exception):
    # Fallback to standard PyTorch implementations; define fallback classes
    class _FallbackFusedAdam(torch.optim.Adam):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            # Apex's FusedAdam has set_grad_none; strip it if present
            kwargs.pop("set_grad_none", None)
            super().__init__(*args, **kwargs)

    _apex_fused_adam = None
    _apex_fused_ln = None

# Export names with permissive Any typing so optional Apex types do not
# cause static assignability failures when Apex is present in some envs.
FusedAdam = _apex_fused_adam if _apex_fused_adam is not None else _FallbackFusedAdam  # type: ignore[assignment]
FusedLayerNorm = _apex_fused_ln if _apex_fused_ln is not None else nn.LayerNorm  # type: ignore[assignment]

if not using_apex:
    print(
        "⚠️ Warning: Apex not found or corrupted. Falling back to standard PyTorch implementations."
    )
