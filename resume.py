"""Resume training entrypoint.

This script provides a small command-line wrapper to resume or fine-tune
previously saved training checkpoints. It parses a few resume related
arguments, restores training options from the checkpoint `options.json`,
initializes logging, and delegates to :class:`Trainer` to continue
training from the requested scale or checkpoint path.

Example
-------
Run with ``--checkpoint-path /path/to/checkpoint --num-iter 100`` to fine-tune.
"""

import argparse
import glob
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

from config import CheckpointFilenames, ExperimentPaths
from device import device_manager
from log import init_output_logging
from options import ResumeOptions
from training.trainer import Trainer

logger = logging.getLogger(__name__)

# from types import SimpleNamespace


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fine-tuning", action="store_true", help="fine-tune the models"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        help="checkpoint path to continue the training",
        required=True,
    )
    parser.add_argument("--num-iter", type=int, help="number of epochs for fine-tuning")
    parser.add_argument(
        "--start-scale", type=int, default=0, help="start scale for fine-tuning"
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="epoch to resume training from within the current scale group",
    )

    # Number of parallel scales to process at once when resuming
    parser.add_argument(
        "--num-parallel-scales",
        "--num_parallel_scales",
        type=int,
        default=2,
        help="number of scales to train in parallel when resuming",
    )

    arguments = parser.parse_args(namespace=ResumeOptions())

    if arguments.fine_tuning and arguments.num_iter is None:
        raise ValueError("Number of iterations required for fine-tuning.")

    # Load the saved input parameter options for the trained models
    checkpoint_path = Path(arguments.checkpoint_path)

    with (checkpoint_path / CheckpointFilenames.OPTIONS).open("r") as f:
        options = json.load(f, object_hook=lambda x: SimpleNamespace(**x))

    options.out_path = arguments.checkpoint_path

    # Forward resume-specific options into the loaded options namespace
    options.start_epoch = getattr(arguments, "start_epoch", 0)

    init_output_logging(str(Path(options.out_path) / "log.txt"))

    if arguments.fine_tuning:
        print(f"Fine-Tuning: {arguments.num_iter} iter")
        options.num_iter = arguments.num_iter

    # Global device initialization using DeviceManager singleton
    device_manager.initialize(
        gpu_id=options.gpu_device,
        use_cpu=getattr(options, "use_cpu", False),
        manual_seed=options.manual_seed,
    )

    trainer = Trainer(
        options,
        arguments.fine_tuning,
        arguments.checkpoint_path,
    )

    if arguments.fine_tuning:
        trainer.load(arguments.checkpoint_path, arguments.start_scale - 1)
    else:
        # Get last saved scale path
        last_scale = max(map(int, next(os.walk(arguments.checkpoint_path))[1]))  # type: ignore
        last_scale_path = checkpoint_path / str(last_scale)

        # If the last scale folder was created, but no models were saved, remove the folder
        has_ckpt = (last_scale_path / CheckpointFilenames.GENERATOR).is_file() or (
            last_scale_path / CheckpointFilenames.EPOCH_CKPT
        ).is_file()

        if not has_ckpt:
            for file in glob.glob(str(last_scale_path / ExperimentPaths.FACIES / "*")):
                os.remove(file)
            os.removedirs(str(last_scale_path / ExperimentPaths.FACIES))
            for file in glob.glob(str(last_scale_path / "*")):
                os.remove(file)
            os.removedirs(str(last_scale_path))

        trainer.load(arguments.checkpoint_path)

    trainer.train()
