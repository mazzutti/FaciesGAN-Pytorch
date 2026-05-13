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
import os
import random
from types import SimpleNamespace

import torch
import torch.distributed as dist

from config import CheckpointFilenames, ExperimentPaths
from enums import DdpBackend, DeviceType
from log import init_output_logging
from options import ResumeOptions
from training.trainer import Trainer

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
    with open(
        os.path.join(arguments.checkpoint_path, CheckpointFilenames.OPTIONS), "r"
    ) as f:
        options = json.load(f, object_hook=lambda x: SimpleNamespace(**x))

    options.out_path = arguments.checkpoint_path

    # Forward resume-specific options into the loaded options namespace
    options.start_epoch = getattr(arguments, "start_epoch", 0)

    init_output_logging(os.path.join(options.out_path, "log.txt"))

    if options.manual_seed is not None:
        random.seed(options.manual_seed)
        torch.manual_seed(options.manual_seed)  # type: ignore

    if arguments.finetuning:
        print("Fine-Tuning: %d iter\n" % arguments.num_iter)
        options.num_iter = arguments.num_iter

    device = torch.device(
        f"{DeviceType.CUDA}:{options.gpu_device}"
        if torch.cuda.is_available()
        else (
            f"{DeviceType.MPS}:{options.gpu_device}"
            if torch.backends.mps.is_available()
            else DeviceType.CPU
        )
    )

    # ── Detect distributed (torchrun) ────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    distributed = local_rank >= 0

    if distributed:
        dist.init_process_group(backend=DdpBackend.NCCL)
        torch.cuda.set_device(local_rank)
        device = torch.device(f"{DeviceType.CUDA}:{local_rank}")

    trainer = Trainer(
        options,
        arguments.fine_tuning,
        arguments.checkpoint_path,
        device,
        distributed=distributed,
    )

    if arguments.fine_tuning:
        trainer.load(arguments.checkpoint_path, arguments.start_scale - 1)
    else:
        # Get last saved scale path
        last_scale = max(map(int, next(os.walk(arguments.checkpoint_path))[1]))  # type: ignore
        last_scale_path = os.path.join(arguments.checkpoint_path, str(last_scale))

        # If the last scale folder was created, but no models were saved, remove the folder
        has_ckpt = os.path.isfile(os.path.join(last_scale_path, CheckpointFilenames.GENERATOR)) or \
                   os.path.isfile(os.path.join(last_scale_path, CheckpointFilenames.EPOCH_CKPT))

        if not has_ckpt:
            for file in glob.glob(
                os.path.join(last_scale_path, ExperimentPaths.FACIES, "*")
            ):
                os.remove(file)
            os.removedirs(os.path.join(last_scale_path, ExperimentPaths.FACIES))
            for file in glob.glob(os.path.join(last_scale_path, "*")):
                os.remove(file)
            os.removedirs(last_scale_path)

        trainer.load(arguments.checkpoint_path)

    trainer.train()
