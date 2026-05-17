"""Configuration constants used across the training and model modules.

This module centralizes filesystem and checkpoint filename constants so
other modules can import a single stable source of truth for file
locations and conventional filenames used by the training and model
serialization code.

Notes
-----
These are plain string constants and are intended to be imported as
required, for example::

        from config import DirectoryConfig, CheckpointFilenames

Do not put runtime logic in this module; it only contains constants.
"""

from dataclasses import dataclass

import torch

from device import device_manager


@dataclass(frozen=True)
class _DomainConfig:
    NUM_FACIES: int = 4
    NUM_FACIES_CHANNELS: int = 3
    NOISE_CHANNELS: int = 3
    RANDOM_SEED: int = 42
    LOSS_SCALE_MIN: float = 1e-4
    EPSILON: float = 1e-8

    @property
    def ZERO_SCALAR(self) -> torch.Tensor:
        return torch.tensor(0.0, device=device_manager.device)


@dataclass(frozen=True)
class _PhysicsConfig:
    FIXED_KERNEL_SIZE: int = 255
    INTERP_GEOMETRY: tuple[int, int] = (150, 120)
    WAVELET_F_PEAK: float = 8.0
    WAVELET_DT: float = 0.001
    WAVELET_LENGTH: float = 0.128
    VP_MIN: float = 2000.0
    VP_MS_SCALE: float = 1000.0

    @property
    def DZ_PIXEL(self) -> torch.Tensor:
        return torch.tensor(1.0, device=device_manager.device)


@dataclass(frozen=True)
class _LoggingConfig:
    SCALAR_LOG_INTERVAL: int = 1
    IMAGE_LOG_INTERVAL: int = 100
    EMA_SYNC_INTERVAL: int = 50


@dataclass(frozen=True)
class _CheckpointFilenames:
    """Singleton container for standard FaciesGAN checkpoint filenames."""

    EPOCH_CKPT: str = "epoch_checkpoint.pth"
    COMPLETED_EPOCH: str = "completed_epoch.txt"
    GENERATOR: str = "generator.pth"
    DISCRIMINATOR: str = "discriminator.pth"
    MASKS: str = "masks.pth"
    REC_NOISE: str = "rec_noise.pth"
    NOISE_AMP: str = "noise_amp.txt"
    SHAPE: str = "shape.pth"
    OPT_G: str = "opt_G.pth"
    OPT_D: str = "opt_D.pth"
    SCH_G: str = "sch_G.pth"
    SCH_D: str = "sch_D.pth"
    OPTIONS: str = "options.json"
    STATS: str = "stats.json"
    EMBEDDINGS: str = "cached_embeddings.npz"


@dataclass(frozen=True)
class _DirectoryConfig:
    """Conventional directory paths for data and outputs."""

    JOBLIB_CACHE: str = ".cache/joblib"
    OUTPUTS: str = "outputs/py"
    DATA: str = "data"
    CACHE: str = "./.cache"
    TENSORBOARD_LOGS: str = "tensorboard_logs"
    CHECKPOINT: str = "outputs/py"
    DEFAULT_DATA: str = "./data"


@dataclass(frozen=True)
class _ExperimentPaths:
    """Conventional subdirectories for experiment outputs."""

    FACIES: str = "real_x_generated_facies"
    IP: str = "real_x_generated_ip"
    IS: str = "real_x_generated_is"
    VP_VS: str = "real_x_generated_vp_vs"


DomainConfig = _DomainConfig()
PhysicsConfig = _PhysicsConfig()
LoggingConfig = _LoggingConfig()
CheckpointFilenames = _CheckpointFilenames()
DirectoryConfig = _DirectoryConfig()
ExperimentPaths = _ExperimentPaths()
