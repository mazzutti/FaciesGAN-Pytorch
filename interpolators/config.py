"""Interpolator configuration dataclasses.

This module contains configuration dataclasses used by all interpolator
implementations (numeric, seismic, well, neural). They provide a
compact way to pass rendering options and geometry parameters.
"""

from dataclasses import dataclass


@dataclass
class InterpolatorConfig:
    """Configuration for multi-scale interpolation.

    This dataclass provides parameters for configuring interpolator
    instances during multi-scale pyramid generation.

    Attributes
    ----------
    num_classes : int
        Number of output facies classes. Defaults to 4.
    strategy : str
        Interpolation strategy to use for numeric data. Supported values
        are ``"categorical"`` (mode-filter) and ``"continuous"``
        (Backus averaging). Defaults to ``"categorical"``.
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
    """

    num_classes: int = 4
    scale: float = 1.0
    upsample: int = 4
    chunk_size: int = 65536
    geometry: tuple[int, int] = (150, 120)
    channels_last: bool = False
    use_mode_filter: bool = True
    strategy: str = "categorical"
    data_min: float | None = None
    data_max: float | None = None
