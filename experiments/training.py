"""Training orchestration for experiments."""

import logging
import subprocess
import sys
from pathlib import Path

# Removed legacy constants import
from config import CheckpointFilenames

logger = logging.getLogger(__name__)


def _has_resumable_scale_artifacts(variant_output: str, scale: int) -> bool:
    """Return True if a scale has real resume artifacts, not just an empty folder."""
    scale_dir = Path(variant_output) / str(scale)
    if not scale_dir.is_dir():
        return False

    required_any = (
        CheckpointFilenames.NOISE_AMP,
        CheckpointFilenames.GENERATOR,
        CheckpointFilenames.EPOCH_CKPT,
        CheckpointFilenames.COMPLETED_EPOCH,
    )
    return any((scale_dir / f).is_file() for f in required_any)


def find_last_completed_scale(variant_output: str) -> int:
    """Return the highest scale index with resumable checkpoint artifacts."""
    last_done = -1
    for i in range(20):
        if _has_resumable_scale_artifacts(variant_output, i):
            last_done = i
        else:
            # Stop at the first missing/non-resumable scale to preserve
            # contiguous scale progression assumptions.
            break
    return last_done


def read_completed_epochs(variant_output: str, scale: int) -> int:
    """Read the number of completed epochs for a specific scale from disk."""
    path = Path(variant_output) / str(scale) / CheckpointFilenames.COMPLETED_EPOCH
    if path.is_file():
        try:
            with path.open("r") as f:
                return int(f.read().strip())
        except (ValueError, OSError):
            return 0
    return 0


def train_variant(args: list[str], nproc: int) -> None:
    """Launch a training run using torchrun for DDP."""
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        "main.py",
    ] + args

    print(f"Executing: {' '.join(cmd)}")
    # We use subprocess.run and check the exit code.
    # We don't use shell=True for security and argument handling.
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Variant training failed with exit code {result.returncode}: {' '.join(cmd)}"
        )
