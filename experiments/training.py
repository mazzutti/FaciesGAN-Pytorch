"""Training orchestration for experiments."""

import os
import subprocess
import sys


def find_last_completed_scale(variant_output: str) -> int:
    """Check how many scale checkpoints already exist in the variant folder."""
    last_done = -1
    # We look for scale folders like "0", "1", "2", etc.
    # Note: scale folders are created by the trainer.
    for i in range(20):
        scale_dir = os.path.join(variant_output, str(i))
        if os.path.isdir(scale_dir):
            last_done = i
        else:
            break
    return last_done


def read_completed_epochs(variant_output: str, scale: int) -> int:
    """Read the number of completed epochs for a specific scale from disk."""
    from config import COMPLETED_EPOCH_FILE

    path = os.path.join(variant_output, str(scale), COMPLETED_EPOCH_FILE)
    if os.path.isfile(path):
        try:
            with open(path, "r") as f:
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
        print(f"\nVariant training failed with exit code {result.returncode}")
        # We don't exit(1) immediately to allow the loop to try other variants
        # if the user wants, but usually, a DDP failure is fatal for the session.
