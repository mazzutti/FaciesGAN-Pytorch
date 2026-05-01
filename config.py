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

# Directory and option filename used by the top-level scripts
OUTPUTS_DIR = "outputs/py"
DATA_DIR = "data"
CHECKPOINT_PATH = OUTPUTS_DIR
OPT_FILE = "options.json"

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
