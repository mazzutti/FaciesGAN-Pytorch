"""Interpolator configuration dataclasses.

This module contains configuration dataclasses used by all interpolator
implementations (numeric, seismic, well, neural). They provide a
compact way to pass rendering options and geometry parameters.
"""

from dataclasses import dataclass

from config import DomainConfig, PhysicsConfig
from enums import InterpolationStrategy


@dataclass
class InterpolatorConfig:
    """Configuration for multiscale interpolation.

    This dataclass provides parameters for configuring interpolator
    instances during multiscale pyramid generation.

    Attributes
    ----------
    num_classes : int
        Number of output facies classes. Defaults to 4.
    scale : float
        Scale parameter (sigma) for Fourier feature encoding (used by
        NeuralSmoother). Defaults to 1.0.
    upsample : int
        Upsampling factor applied to base geometry dimensions.
        Defaults to 4.
    chunk_size : int
        Batch size for coordinate-wise evaluation (used by NeuralSmoother).
        Defaults to 65536.
    geometry : tuple[int, int]
        Base spatial dimensions (Height, Width) of the data.
    channels_last : bool
        If True, produces tensors in (H, W, C) layout. Defaults to False.
    use_mode_filter : bool
        If True, uses majority-vote pooling for downscaling categorical
        data. Defaults to True.
    strategy : str
        Interpolation strategy to use for numeric data. Supported values
        are ``InterpolationStrategy.CATEGORICAL`` (mode-filter) and ``InterpolationStrategy.CONTINUOUS``
        (Backus averaging). Defaults to ``InterpolationStrategy.CATEGORICAL``.
    data_min : float, optional
        Minimum value for data normalization. Defaults to None.
    data_max : float, optional
        Maximum value for data normalization. Defaults to None.
    normalization_range : tuple[float, float]
        Inclusive output range ``(min, max)`` applied after normalization
        for continuous data. Defaults to ``(0.0, 1.0)``.
    """

    num_classes: int = DomainConfig.NUM_FACIES
    scale: float = 1.0
    upsample: int = 4
    chunk_size: int = 65536
    geometry: tuple[int, int] = PhysicsConfig.INTERP_GEOMETRY
    channels_last: bool = False
    use_mode_filter: bool = True
    strategy: str = InterpolationStrategy.CATEGORICAL
    data_min: float | None = None
    data_max: float | None = None
    normalization_range: tuple[float, float] = (0.0, 1.0)
