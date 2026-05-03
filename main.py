"""Main entry point for parallel multi-scale FaciesGAN training.

This script provides the command-line interface and initialization for
training a FaciesGAN model with parallel scale processing. Multiple pyramid
scales can be trained simultaneously for faster overall training.
"""

import atexit
import json
import os
import random
import signal
from argparse import ArgumentParser
from datetime import datetime

# Suppress torch.compile symbolic-shape C++ warnings (pow_by_natural etc.)
# Must be set before importing torch.  Apex AMP (O1) is compatible with
# torch.compile; Apex issues its own diagnostics separately.
os.environ.setdefault("TORCH_LOGS", "-dynamo")
os.environ.setdefault("TORCHDYNAMO_VERBOSE", "0")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:128",
)


# ---------------------------------------------------------------------------
# Silence harmless resource_tracker KeyError tracebacks at exit.
#
# loky (used by joblib.Memory for pyramid caching) creates POSIX shared-
# memory segments and cleans them up on its own.  Python 3.12's
# multiprocessing.resource_tracker daemon still has them registered and
# prints noisy KeyError tracebacks when it later tries to double-clean.
#
# The tracker is a separate process so we cannot monkey-patch it.  Instead,
# we register an atexit handler that terminates the tracker daemon before
# it enters the cleanup-and-warn code path.  Because loky already unlinked
# every segment it created, nothing actually leaks.
# ---------------------------------------------------------------------------
def _silence_resource_tracker() -> None:
    try:
        from multiprocessing.resource_tracker import (
            _resource_tracker,  # type: ignore[attr-defined]; type: ignore[import-untyped]
        )

        if _resource_tracker._pid is not None:  # type: ignore[union-attr]
            os.kill(_resource_tracker._pid, signal.SIGKILL)  # type: ignore[arg-type]
            os.waitpid(_resource_tracker._pid, 0)  # type: ignore[arg-type]
            _resource_tracker._pid = None  # type: ignore[assignment]
        if _resource_tracker._fd is not None:  # type: ignore[union-attr]
            os.close(_resource_tracker._fd)  # type: ignore[arg-type]
            _resource_tracker._fd = None  # type: ignore[assignment]
    except Exception:
        pass


atexit.register(_silence_resource_tracker)

import torch
import torch.distributed as dist
from dateutil import tz  # type: ignore[import-untyped]

from config import OPT_FILE, PhysicsConfig
from log import init_output_logging
from options import TrainingOptions
from training import Trainer


def get_arguments() -> ArgumentParser:
    """Parse command-line arguments for parallel FaciesGAN training.

    Returns
    -------
    ArgumentParser
        Configured argument parser with all training options.
    """
    parser = ArgumentParser()

    # workspace:
    parser.add_argument("--use-cpu", action="store_true", help="use cpu")
    parser.add_argument("--gpu-device", type=int, help="which GPU to use", default=0)
    parser.add_argument("--input-path", help="input facie path", required=True)

    # load, input, save configurations:
    parser.add_argument("--manual-seed", type=int, help="manual seed")
    parser.add_argument(
        "--output-path", help="output folder path", default="resuslts/py/"
    )
    parser.add_argument(
        "--output-fullpath",
        help="Set exact output path (overrides automatic timestamp prefix).",
        default=None,
    )
    parser.add_argument("--stop-scale", type=int, help="stop scale", default=6)
    parser.add_argument(
        "--start-scale",
        type=int,
        help="scale to start/resume training from (default: 0)",
        default=0,
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        help="epoch to resume training from within the current scale group (default: 0)",
        default=0,
    )
    parser.add_argument(
        "--num-facies-classes",
        "--facies-channels",
        "--num-img-channels",
        type=int,
        dest="num_facies_classes",
        help="number of one-hot encoded facies classes (channels)",
        default=4,
    )
    parser.add_argument(
        "--noise-channels",
        type=int,
        help="number of noise channels to generate per scale",
        default=3,
    )
    parser.add_argument(
        "--img-color-range",
        type=int,
        nargs=2,
        help="range of values in the input facie",
        default=[0, 255],
    )
    parser.add_argument(
        "--crop-size", type=int, help="crop size to train the facie", default=256
    )
    parser.add_argument(
        "--batch-size",
        default=1,
        type=int,
        help="Total batch size - e.g: num_gpus = 2, batch_size = 128 then, effectively, 64",
    )

    # networks hyperparameters:
    parser.add_argument(
        "--num-features",
        type=int,
        help="initial number of features in each layer",
        default=32,
    )
    parser.add_argument(
        "--min-num-features",
        type=int,
        help="minimal number of features in each layer",
        default=32,
    )
    parser.add_argument("--kernel-size", type=int, help="kernel size", default=3)
    parser.add_argument(
        "--num-layers", type=int, help="number of layers in each scale", default=5
    )
    parser.add_argument("--stride", help="stride", default=1)
    parser.add_argument("--padding-size", type=int, help="net pad size", default=0)

    # pyramid parameters:
    parser.add_argument(
        "--noise-amp", type=float, help="adaptive noise cont weight", default=0.1
    )
    parser.add_argument(
        "--min-noise-amp",
        type=float,
        help="minimum noise amplitude floor for diversity",
        default=0.5,
    )
    parser.add_argument(
        "--scale0-noise-amp",
        type=float,
        help="noise amplitude at scale 0 (controls structural diversity)",
        default=1.0,
    )

    # Parallel training specific parameters:
    parser.add_argument(
        "--num-parallel-scales",
        type=int,
        help="Number of scales to train in parallel (default: 2)",
        default=2,
    )

    # profiling
    parser.add_argument(
        "--use-profiler",
        action="store_true",
        help="Enable PyTorch profiler and export a chrome trace to the output path",
    )

    # optimization hyperparameters:
    parser.add_argument(
        "--num-iter",
        type=int,
        default=2000,
        help="number of full dataset passes (each pass shuffles the dataset independently)",
    )
    parser.add_argument("--gamma", type=float, help="scheduler gamma", default=0.9)
    parser.add_argument(
        "--lr-g", type=float, default=5e-4, help="learning rate, default=5e-4"
    )
    parser.add_argument(
        "--lr-d", type=float, default=5e-4, help="learning rate, default=5e-4"
    )
    parser.add_argument(
        "--lr-decay",
        type=int,
        default=1000,
        help="number of epochs (or steps if --lr-decay-unit=step) before lr decay (used by discriminator StepLR)",
    )
    parser.add_argument(
        "--lr-decay-unit",
        type=str,
        choices=["epoch", "step", "batch"],
        default="epoch",
        help="unit for --lr-decay: 'epoch' decays per-batch epoch count (reset each batch), "
        "'step' decays by global optimisation step across all batches, "
        "'batch' decays once every N dataset batches (schedulers accumulate across batches, default: epoch)",
    )
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=400,
        help="ReduceLROnPlateau patience: epochs with no improvement before reducing generator LR (default: 400)",
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=1e-4,
        help="minimum learning rate for generator ReduceLROnPlateau (default: 1e-4)",
    )
    parser.add_argument(
        "--lr-smoothing-alpha",
        type=float,
        default=0.95,
        help="EMA smoothing factor for generator loss fed to ReduceLROnPlateau (0=no smoothing, 0.99=very smooth, default: 0.95)",
    )
    parser.add_argument(
        "--lr-g-factor",
        type=float,
        default=0.8,
        help="Factor by which the generator LR is reduced on plateau (default: 0.8)",
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="beta1 for adam. default=0.5"
    )
    parser.add_argument(
        "--generator-steps", type=int, help="Generator inner steps", default=3
    )
    parser.add_argument(
        "--discriminator-steps", type=int, help="Discriminator inner steps", default=3
    )
    parser.add_argument(
        "--scale0-disc-steps-multiplier",
        type=int,
        default=1,
        help="Extra D-step multiplier for scale 0 only (default: 1, i.e. no extra steps).",
    )
    parser.add_argument(
        "--scale0-loss-multiplier",
        type=float,
        default=1.0,
        help="Extra loss multiplier applied to rec and rock physics losses at scale 0 (default: 1.0).",
    )
    parser.add_argument(
        "--gradient-loss-penalty",
        type=float,
        help="gradient penalty weight",
        default=0.1,
    )
    parser.add_argument(
        "--facies-rec-loss-penalty",
        "--reconstruction-loss-penalty",
        type=float,
        dest="facies_rec_loss_penalty",
        help="reconstruction loss weight",
        default=10,
    )
    parser.add_argument(
        "--gp-interval",
        type=int,
        help=(
            "Gradient penalty lazy regularization interval. Compute the "
            "expensive WGAN-GP penalty every N discriminator steps instead "
            "of every step (StyleGAN2-style). Weight is scaled by N to "
            "compensate. Default 16."
        ),
        default=16,
    )
    parser.add_argument(
        "--num-diversity-samples",
        type=int,
        help="number of diverse samples to generate per G-step for diversity loss (default: 3)",
        default=3,
    )
    parser.add_argument(
        "--diversity-loss-penalty",
        type=float,
        help="additional scalar multiplier applied to the generator diversity loss (default: 1.0)",
        default=1.0,
    )
    parser.add_argument(
        "--adversarial-loss-penalty",
        type=float,
        help="scalar multiplier applied to the generator adversarial loss -E[D(fake)] (default: 1.0)",
        default=1.0,
    )
    parser.add_argument(
        "--save-interval", type=int, help="save log interval", default=100
    )
    parser.add_argument(
        "--checkpoint-interval",
        "--checkpoint_interval",
        type=int,
        help="Interval (in epochs) between saving training state checkpoints for resume (default: 1).",
        default=1,
    )
    parser.add_argument(
        "--num-real-facies",
        type=int,
        help="Number of real facies to use in the grid plot",
        default=5,
    )
    parser.add_argument(
        "--num-generated-per-real",
        type=int,
        help="Number of generated facies per real facies to use in the grid plot",
        default=5,
    )
    parser.add_argument(
        "--num-train-pyramids",
        type=int,
        help="Number of train pyramids to use in the FaciesGAN training",
        default=200,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number of workers for data loading (0 = main process only). "
        "Defaults to min(4, cpu_count//2) for parallel data prep.",
        default=min(4, max(1, (os.cpu_count() or 1) // 2)),
    )

    parser.add_argument(
        "--use-wells",
        action="store_true",
        help="enable using wells during data loading (filter by --wells-mask-columns if set)",
    )

    parser.add_argument(
        "--wells-mask-columns",
        type=int,
        help="list of well indices to train the model from",
        nargs="+",
        default=tuple(),
    )

    parser.add_argument(
        "--well-loss-penalty",
        type=float,
        help="weight multiplier for well/mask reconstruction loss",
        default=10.0,
    )

    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="max gradient norm for generator clipping (default: 1.0; set to 0 to disable)",
    )

    parser.add_argument(
        "--use-seismic",
        action="store_true",
        help="enable using seismic data during data loading",
    )

    parser.add_argument(
        "--use-rock-physics",
        "--use-rock_physics",
        action="store_true",
        dest="use_rock_physics",
        help="Train with rock physics volumes (Ip, Is, Vp/Vs) as additional output channels.",
    )
    parser.add_argument(
        "--rock-physics-loss-penalty",
        "--rock_physics-loss-penalty",
        type=float,
        dest="rock_physics_loss_penalty",
        default=1.0,
        help="Extra loss multiplier for rock physics reconstruction (default: 1.0).",
    )
    parser.add_argument(
        "--rec-rock-physics-loss-penalty",
        type=float,
        dest="rec_rock_physics_loss_penalty",
        default=1.0,
        help="Extra loss multiplier for rock-physics reconstruction on the generated volume (default: 1.0).",
    )
    parser.add_argument(
        "--tv-loss-penalty",
        type=float,
        dest="tv_loss_penalty",
        default=1.0,
        help="Scalar multiplier for the total-variation smoothness loss (default: 1.0).",
    )
    parser.add_argument(
        "--elastic-loss-penalty",
        type=float,
        dest="elastic_loss_penalty",
        default=1.0,
        help="Scalar multiplier for the elastic-consistency loss (default: 1.0).",
    )
    parser.add_argument(
        "--physics-loss-penalty",
        type=float,
        dest="physics_loss_penalty",
        default=1.0,
        help="Scalar multiplier for the seismic physics loss (default: 1.0).",
    )
    parser.add_argument(
        "--dz-pixel",
        type=float,
        dest="dz_pixel",
        default=5.0,
        help="Vertical resolution in meters per pixel (default: 5.0).",
    )
    parser.add_argument(
        "--wavelet-f-peak",
        type=float,
        dest="wavelet_f_peak",
        default=PhysicsConfig.WAVELET_F_PEAK,
        help=f"Peak frequency of the Ricker wavelet in Hz (default: {PhysicsConfig.WAVELET_F_PEAK}).",
    )
    parser.add_argument(
        "--wavelet-dt",
        type=float,
        dest="wavelet_dt",
        default=PhysicsConfig.WAVELET_DT,
        help=f"Wavelet sampling interval in seconds (default: {PhysicsConfig.WAVELET_DT}).",
    )

    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="disable dataset shuffling (useful for reproducible debugging)",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="disable TensorBoard logging during training",
    )
    parser.add_argument(
        "--no-plot-outputs",
        action="store_true",
        help="disable generated output visualizations (facies and rock physics) during training",
    )

    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="disable torch.compile on CUDA (enabled by default for Ampere+ GPUs)",
    )
    parser.add_argument(
        "--compile-backend",
        action="store_true",
        help="enable torch.compile on CUDA (enabled by default)",
    )
    parser.add_argument(
        "--hand-off-to-c",
        action="store_true",
        help="Hand off orchestration to compiled C library via ctypes (thin wrapper)",
    )
    parser.add_argument(
        "--gradient-checkpoint",
        action="store_true",
        help=(
            "Enable activation checkpointing on generator blocks.  Trades "
            "~30%% extra compute for significantly lower peak GPU memory, "
            "allowing larger batch sizes.  Incompatible with torch.compile "
            "(compile is automatically disabled when this flag is set)."
        ),
    )
    parser.add_argument(
        "--amp-dtype",
        type=str,
        choices=["fp16", "bf16"],
        default="bf16",
        help=(
            "AMP compute dtype for CUDA autocast. "
            "Use bf16 on Ampere+ for improved stability/perf tradeoff."
        ),
    )

    return parser


def _kill_child_processes() -> None:
    """Kill all descendant processes of the current PID.

    Walks ``/proc`` to find every process whose parent (ppid) matches our
    pid, then sends ``SIGTERM`` first (giving children a chance to drain
    in-flight GPU / NCCL operations) and falls back to ``SIGKILL`` if
    they are still alive after a short grace period.  Killing a process
    with SIGKILL while CUDA kernels or NCCL collectives are in-flight
    can leave the GPU driver in a bad state ("GPU has fallen off the
    bus").
    """
    import time

    my_pid = os.getpid()
    children: list[int] = []
    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/stat") as f:
                    parts = f.read().split()
                ppid = int(parts[3])
                child_pid = int(entry)
                if ppid == my_pid and child_pid != my_pid:
                    children.append(child_pid)
            except (OSError, IndexError, ValueError):
                continue
    except OSError:
        pass

    # Phase 1: SIGTERM — let children flush GPU work
    for pid in children:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass

    # Brief grace period for GPU kernels to drain
    if children:
        time.sleep(0.5)

    # Phase 2: SIGKILL any survivors
    for pid in children:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def main() -> None:
    """Run parallel FaciesGAN training.

    Handles argument parsing, environment setup, logger initialization, and
    constructs the :class:`Trainer` instance that performs the training
    loop. This entry point is intended for CLI usage and is also used in
    the ``if __name__ == '__main__'`` guard.
    """
    argument_parser = get_arguments()
    options = argument_parser.parse_args(namespace=TrainingOptions())

    # Handle --no-shuffle flag
    if hasattr(options, "no_shuffle") and options.no_shuffle:
        options.shuffle = False

    # Handle --no-tensorboard flag
    if hasattr(options, "no_tensorboard") and options.no_tensorboard:
        options.enable_tensorboard = False

    # Handle --no-plot-outputs flag
    if hasattr(options, "no_plot_outputs") and options.no_plot_outputs:
        options.enable_plot_outputs = False

    # Handle --no-compile flag (overrides --compile-backend if both are passed)
    if hasattr(options, "no_compile") and options.no_compile:
        options.compile_backend = False
    # Otherwise, compile_backend value from args or default (False) is used

    # Handle --gradient-checkpoint (incompatible with torch.compile)
    if hasattr(options, "gradient_checkpoint") and options.gradient_checkpoint:
        options.gradient_checkpointing = True
        # torch.compile reorders saved tensors in a way that breaks
        # checkpoint recomputation metadata — disable it when
        # checkpointing is active.
        options.compile_backend = False
    else:
        if not hasattr(options, "gradient_checkpointing"):
            options.gradient_checkpointing = False

    if options.manual_seed is not None:
        random.seed(options.manual_seed)
        torch.manual_seed(options.manual_seed)  # type: ignore

    # ── Detect distributed (torchrun) ────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    distributed = local_rank >= 0
    rank = 0

    # Ensure SIGTERM (sent by torchrun when a peer dies) actually kills us
    # instead of being swallowed by debugpy or hanging in NCCL collectives.
    signal.signal(signal.SIGTERM, lambda *_: os._exit(1))  # type: ignore

    # Set OMP_NUM_THREADS before torchrun can default it to 1.
    if "OMP_NUM_THREADS" not in os.environ:
        omp_threads = max(1, (os.cpu_count() or 1) // max(1, torch.cuda.device_count()))
        os.environ["OMP_NUM_THREADS"] = str(omp_threads)

    if distributed:
        # Tell NCCL to surface errors asynchronously instead of blocking
        # forever when a peer dies or a collective is mismatched.  With
        # this flag an unrecoverable NCCL error raises a Python exception
        # rather than hanging the whole job.
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        device_id = (
            torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else None
        )
        # Use a 5-minute timeout so a DDP desync surfaces as an error
        # instead of hanging silently for the default 30 minutes.
        from datetime import timedelta

        dist.init_process_group(
            backend=backend,
            device_id=device_id,
            timeout=timedelta(minutes=15),
        )
        rank = dist.get_rank()
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            torch.cuda.set_per_process_memory_fraction(0.90)  # type: ignore
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        # Keep model initialisation deterministic (same weights on every rank);
        # the DistributedSampler gives each rank different data.
        if options.manual_seed is not None:
            torch.manual_seed(options.manual_seed)  # type: ignore
            # Seed the per-device CUDA RNG so noise tensors generated on
            # each rank are reproducible (though intentionally different
            # across ranks for diversity).
            if torch.cuda.is_available():
                torch.cuda.manual_seed(options.manual_seed + rank)
    else:
        device = (
            torch.device("cpu")
            if getattr(options, "use_cpu", False) or not torch.cuda.is_available()
            else torch.device(f"cuda:{options.gpu_device}")
        )

    is_main = rank == 0

    if getattr(options, "output_fullpath", None):
        options.output_path = options.output_fullpath
    else:
        timestamp = datetime.now(tz.tzlocal()).strftime("%Y_%m_%d_%H_%M_%S")
        options.output_path = os.path.join(options.output_path, timestamp)
    options.start_scale = (
        options.start_scale
        if hasattr(options, "start_scale") and options.start_scale
        else 0
    )

    if is_main:
        os.makedirs(options.output_path, exist_ok=True)

        # Save the input parameters options
        with open(os.path.join(options.output_path, OPT_FILE), "w") as file:
            json.dump(vars(options), file, indent=4)  # type: ignore

        init_output_logging(os.path.join(options.output_path, "log.txt"))

    # Synchronise so non-zero ranks wait for rank 0 to create output dir
    if distributed:
        dist.barrier()  # type: ignore[arg-type]
    if not is_main:
        os.makedirs(options.output_path, exist_ok=True)

    if is_main:
        print("\n" + "=" * 60)
        print("PARALLEL FACIESGAN TRAINING")
        print("=" * 60)
        print(f"Device: {device}")
        if distributed:
            world_size = dist.get_world_size()
            print(f"DDP training: {world_size} processes")
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Training scales: {options.start_scale} to {options.stop_scale}")
        print(f"Parallel scales: {options.num_parallel_scales}")
        print(f"Iterations per scale: {options.num_iter}")
        print(f"Output path: {options.output_path}")
        print("=" * 60 + "\n")

    # Performance tuning: enable cuDNN autotuner and TF32 where available
    try:
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            # Allow TF32 for faster matmuls on compatible NVIDIA GPUs
            try:
                torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore
            except Exception:
                pass
            try:
                torch.backends.cudnn.allow_tf32 = True  # type: ignore
            except Exception:
                pass
            # Use TF32 precision globally for matmuls (Ampere+ GPUs)
            try:
                torch.set_float32_matmul_precision("high")  # type: ignore
            except Exception:
                pass

            # Silence harmless torch.compile symbolic-shape warnings.

            # The pow_by_natural warnings come from torch.utils._sympy.interp
            # logger at WARNING level during shape guard compilation.
            import logging
            import warnings

            logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
            logging.getLogger("torch._functorch").setLevel(logging.ERROR)
            logging.getLogger("torch._inductor").setLevel(logging.ERROR)
            logging.getLogger("torch.utils._sympy.interp").setLevel(logging.ERROR)
            warnings.filterwarnings("ignore", message=".*pow_by_natural.*")
            warnings.filterwarnings("ignore", category=UserWarning, module="torch")
            # Only call set_logs when TORCH_LOGS env var is not already set,
            # otherwise PyTorch ignores the call and emits a warning.
            if "TORCH_LOGS" not in os.environ:
                try:
                    torch._logging.set_logs(dynamo=logging.ERROR)  # type: ignore[attr-defined]
                except Exception:
                    pass
    except Exception:
        # If backend tuning isn't supported on this build, continue without failing
        pass

    # Reasonable default for intra-op threads to avoid oversubscription
    try:
        cpu_threads = min(4, max(1, (os.cpu_count() or 1) // 2))
        torch.set_num_threads(cpu_threads)
    except Exception:
        pass

    trainer = Trainer(options, device=device, distributed=distributed)
    if is_main:
        print("Using Torch backend for training.")

    # Resume from checkpoint when starting from a non-zero scale
    if options.start_scale > 0:
        trainer.load(options.output_path, until_scale=options.start_scale - 1)
        if is_main:
            print(f"Resumed from checkpoint: loaded scales 0-{options.start_scale - 1}")

    # Register cleanup so child processes are killed on any exit path
    # (unhandled exception, KeyboardInterrupt, normal exit, etc.)
    atexit.register(_kill_child_processes)

    # Optionally run the trainer under the PyTorch profiler and export traces.
    # Optionally run the trainer under the PyTorch profiler and export traces.
    try:
        if getattr(options, "use_profiler", False):
            # Use standard torch.profiler for CPU/CUDA
            try:
                from torch.profiler import ProfilerActivity  # type: ignore
                from torch.profiler import profile
            except Exception:
                print("Warning: PyTorch profiler not available in this environment.")
                trainer.train()
            else:
                trace_file = os.path.join(options.output_path, "profiler_trace.json")
                activities = [ProfilerActivity.CPU]
                if torch.cuda.is_available():
                    activities.append(ProfilerActivity.CUDA)

                with profile(
                    activities=activities, record_shapes=True, profile_memory=True
                ) as prof:
                    trainer.train()

                try:
                    prof.export_chrome_trace(trace_file)
                    print(f"Profiler trace saved to: {trace_file}")
                    print("View in Chrome at: chrome://tracing")
                except Exception as e:
                    print(f"Could not export profiler trace: {e}")
        else:
            trainer.train()
    except Exception as exc:
        import sys
        import traceback

        # Print the traceback on EVERY rank so the actual error from the
        # failing worker is never silently swallowed.  Previously only
        # rank 0 printed, which hid OOM/NCCL errors from other ranks.
        rank_label = f"[rank {rank}] " if distributed else ""
        print(
            f"\n{rank_label}{'=' * 60}\n"
            f"{rank_label}TRAINING FAILED — cleaning up\n"
            f"{rank_label}{'=' * 60}",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)

        # Surface CUDA memory stats when the failure looks like OOM.
        if torch.cuda.is_available():
            try:
                dev = torch.device(f"cuda:{local_rank}" if distributed else device)
                alloc = torch.cuda.memory_allocated(dev) / (1024**3)
                reserved = torch.cuda.memory_reserved(dev) / (1024**3)
                peak = torch.cuda.max_memory_allocated(dev) / (1024**3)
                total = torch.cuda.get_device_properties(dev).total_mem / (1024**3)  # type: ignore
                print(
                    f"{rank_label}CUDA memory: "
                    f"alloc={alloc:.2f}G  reserved={reserved:.2f}G  "
                    f"peak={peak:.2f}G  total={total:.2f}G",
                    file=sys.stderr,
                )
            except Exception:
                pass

        sys.stderr.flush()
        sys.stdout.flush()

        # Shut down background workers before hard exit
        try:
            from background_workers import BackgroundWorker

            BackgroundWorker().shutdown(wait=False)
        except Exception:
            pass

        # Under DDP, a crash on one rank leaves the other hanging in NCCL
        # collectives.  dist.destroy_process_group() itself can block when
        # the peer is already dead.  Drain pending GPU work before hard-
        # exiting so the driver is not left in a dirty state ("GPU has
        # fallen off the bus").
        if distributed:
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception as sync_err:
                    print(
                        f"Warning: CUDA sync before exit failed: {sync_err}",
                        file=sys.stderr,
                    )
            os._exit(1)

        raise exc
    finally:
        import sys

        try:
            from background_workers import BackgroundWorker

            bw = BackgroundWorker()
            # Wait for pending plot jobs with a bounded timeout so we
            # don't block DDP teardown indefinitely.  Then shut down the
            # pool without waiting (workers are killed on process exit).
            bw.wait_pending(timeout=120.0)
            bw.shutdown(wait=False)
        except Exception as e:
            print(f"Warning: BackgroundWorker shutdown failed: {e}", file=sys.stderr)
        if distributed:
            try:
                if (
                    os.environ.get("FG_PROFILE_ALLREDUCE", "0") == "1"
                    and "trainer" in locals()
                    and hasattr(trainer, "model")
                    and hasattr(trainer.model, "report_allreduce_profile")
                ):
                    report_allreduce = getattr(
                        trainer.model, "report_allreduce_profile"
                    )
                    report_allreduce()
            except Exception as profile_err:
                print(
                    f"Warning: allreduce profile reporting failed: {profile_err}",
                    file=sys.stderr,
                )

            # Explicitly delete trainer/model BEFORE destroying the NCCL
            # process group.  If these objects survive until Python's
            # final GC sweep, their C++ destructors try to free
            # NCCL-related buffers after the communicator is gone,
            # causing std::bad_alloc.
            if "trainer" in locals():
                try:
                    del trainer  # noqa: F821
                except Exception:
                    pass
            import gc

            gc.collect()

            # Drain all pending GPU kernels before tearing down NCCL.
            # Destroying the process group while CUDA ops are in-flight
            # can corrupt driver state ("GPU has fallen off the bus").
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                except Exception as sync_err:
                    print(
                        f"Warning: CUDA sync before destroy_process_group "
                        f"failed: {sync_err}",
                        file=sys.stderr,
                    )
            # Barrier so both ranks reach destroy_process_group together.
            # Without this, one rank may be stuck in BackgroundWorker
            # shutdown while the other already entered destroy_process_group
            # (a collective), causing a hang.
            try:
                if dist.is_initialized():
                    dist.barrier()  # type: ignore[arg-type]
            except Exception:
                pass
            try:
                dist.destroy_process_group()
            except Exception as e:
                print(
                    f"Warning: destroy_process_group failed: {e}",
                    file=sys.stderr,
                )

            # Pre-empt the Inductor atexit handler that waits 300s for
            # compile-worker subprocesses to exit — after NCCL teardown
            # they can hang indefinitely.  Unregister it, then kill all
            # remaining child processes (compile workers included).
            try:
                from torch._inductor.async_compile import shutdown_compile_workers

                atexit.unregister(shutdown_compile_workers)
            except Exception:
                pass
            _kill_child_processes()

    if is_main:
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    import lovely_tensors as lt  # type: ignore

    lt.monkey_patch()

    main()
