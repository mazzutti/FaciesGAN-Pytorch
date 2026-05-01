"""Multi-scale interpolation package for FaciesGAN.

This package provides various strategies for generating multi-resolution
pyramids from categorical (facies), continuous (seismic, rock physics),
sparse (well locations), and binary (mask) data. All interpolators are
designed to preserve physical and statistical integrity across scales.
"""

from .base import BaseInterpolator
from .config import InterpolatorConfig
from .mask import MaskInterpolator
from .numeric import NumericInterpolator
from .seismic import SeismicInterpolator
from .well import WellInterpolator

__all__ = [
    "BaseInterpolator",
    "InterpolatorConfig",
    "MaskInterpolator",
    "NumericInterpolator",
    "SeismicInterpolator",
    "WellInterpolator",
]
