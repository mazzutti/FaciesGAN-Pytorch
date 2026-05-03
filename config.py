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

OUTPUTS_DIR = "outputs/py"
DATA_DIR = "data"
CACHE_DIR = "./.cache"
TENSORBOARD_LOGS_DIR = "tensorboard_logs"
CHECKPOINT_PATH = OUTPUTS_DIR
OPT_FILE = "options.json"

# ── Structured Configurations ────────────────────────────────────────

@dataclass(frozen=True)
class _DomainConfig:
    NUM_FACIES: int = 4
    NOISE_CHANNELS: int = 4
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

DomainConfig = _DomainConfig()
PhysicsConfig = _PhysicsConfig()
LoggingConfig = _LoggingConfig()

# Optimizer / scheduler checkpoint filenames
OPT_G_FILE = "opt_G.pth"
OPT_D_FILE = "opt_D.pth"
SCH_G_FILE = "sch_G.pth"
SCH_D_FILE = "sch_D.pth"
OUTPUT_FACIES_PATH = "real_x_generated_facies"
OUTPUT_IP_PATH = "real_x_generated_ip"
OUTPUT_IS_PATH = "real_x_generated_is"
OUTPUT_VP_VS_PATH = "real_x_generated_vp_vs"


# Epoch-level checkpoint (saved inside each scale directory)
EPOCH_CKPT_FILE = "epoch_checkpoint.pth"

# Small metadata file recording the last completed epoch for a scale
# group.  Written when the group finishes so that resume can detect the
# actual training progress even after the epoch checkpoint is removed.
COMPLETED_EPOCH_FILE = "completed_epoch.txt"

# Model and auxiliary filenames used by the facies GAN implementation
G_FILE = "generator.pth"
D_FILE = "discriminator.pth"
M_FILE = "masks.pth"
REC_FILE = "rec_noise.pth"
AMP_FILE = "noise_amp.txt"
SHAPE_FILE = "shape.pth"
