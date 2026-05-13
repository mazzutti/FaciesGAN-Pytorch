"""Shared constants for the experiments module."""

import torch

from enums import EmbeddingMethod as EmbeddingMethod
from enums import ExperimentVariant as ExperimentVariant
from enums import VariantConfig as VariantConfig

EMBEDDINGS_FILE = "cached_embeddings.npz"

__all__ = [
    "AMP_FILE",
    "CACHE_DIR",
    "CHECKPOINT_PATH",
    "COMPLETED_EPOCH_FILE",
    "DATA_DIR",
    "DEFAULT_DATA_DIR",
    "D_FILE",
    "EMBEDDINGS_FILE",
    "EPOCH_CKPT_FILE",
    "EmbeddingMethod",
    "ExperimentVariant",
    "G_FILE",
    "JOBLIB_CACHE_DIR",
    "M_FILE",
    "OPT_D_FILE",
    "OPT_FILE",
    "OPT_G_FILE",
    "OUTPUTS_DIR",
    "OUTPUT_FACIES_PATH",
    "OUTPUT_IP_PATH",
    "OUTPUT_IS_PATH",
    "OUTPUT_VP_VS_PATH",
    "REC_FILE",
    "SCH_D_FILE",
    "SCH_G_FILE",
    "SHAPE_FILE",
    "STATS_FILENAME",
    "TENSORBOARD_LOGS_DIR",
    "VP_MS_SCALE",
    "VariantConfig",
]

ZERO_SCALAR: torch.Tensor = torch.tensor(0.0)
DZ_PIXEL: torch.Tensor = torch.tensor(5.0)

# Default base directory for datasets
DEFAULT_DATA_DIR = "./data"

# Filename for the precomputed global statistics JSON
STATS_FILENAME = "stats.json"

# Directory used by joblib.Memory for caching pyramid computations
JOBLIB_CACHE_DIR = ".cache/joblib"

# Scale factor to convert VP/VS from km/s (stored units) to m/s
VP_MS_SCALE = 1000.0

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


OUTPUTS_DIR = "outputs/py"
DATA_DIR = "data"
CACHE_DIR = "./.cache"
TENSORBOARD_LOGS_DIR = "tensorboard_logs"
CHECKPOINT_PATH = OUTPUTS_DIR
OPT_FILE = "options.json"
