"""Configuration constants used across the training and model modules.

This module centralizes filesystem and checkpoint filename constants so
other modules can import a single stable source of truth for file
locations and conventional filenames used by the training and model
serialization code.

Notes
-----
These are plain string constants and are intended to be imported as
required, for example::

        from config import OUTPUTS_DIR, G_FILE

Do not put runtime logic in this module; it only contains constants.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class _DomainConfig:
    NUM_FACIES: int = 4
    NUM_FACIES_CHANNELS: int = 3
    NOISE_CHANNELS: int = 3
    RANDOM_SEED: int = 42
    LOSS_SCALE_MIN: float = 1e-4
    EPSILON: float = 1e-6


@dataclass(frozen=True)
class _PhysicsConfig:
    FIXED_KERNEL_SIZE: int = 255
    INTERP_GEOMETRY: tuple[int, int] = (150, 120)
    WAVELET_F_PEAK: float = 8.0
    WAVELET_DT: float = 0.001
    WAVELET_LENGTH: float = 0.128
    VP_MIN: float = 2000.0
    VELOCITY_SCALE: float = 1000.0


@dataclass(frozen=True)
class _LoggingConfig:
    SCALAR_LOG_INTERVAL: int = 1
    IMAGE_LOG_INTERVAL: int = 100
    EMA_SYNC_INTERVAL: int = 50


@dataclass(frozen=True)
class VariantConfig:
    use_wells: bool
    use_seismic: bool
    use_rock_physics: bool
    label: str


DomainConfig = _DomainConfig()
PhysicsConfig = _PhysicsConfig()
LoggingConfig = _LoggingConfig()
