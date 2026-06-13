"""Main entry point for parallel multi-scale FaciesGAN training.

This script provides the command-line interface and initialization for
training a FaciesGAN model with parallel scale processing. Multiple pyramid
scales can be trained simultaneously for faster overall training.
"""

import atexit
import json
import logging
import os
from pathlib import Path
import signal
import warnings
from argparse import ArgumentParser
from datetime import datetime

import psutil
import torch
import torch.distributed as dist
from dateutil import tz  # type: ignore[import-untyped]

import utils
import cli_shared
from config import CheckpointFilenames
from device import device_manager
from log import init_output_logging
from options import TrainingOptions
from training import Trainer

logger = logging.getLogger(__name__)


def _setup_environment() -> None:
    """Configure environment variables and silence noisy loggers."""
    # Suppress torch.compile symbolic-shape C++ warnings
    os.environ.setdefault("TORCH_LOGS", "-dynamo,-inductor")
    os.environ.setdefault("TORCHDYNAMO_VERBOSE", "0")
    os.environ.setdefault("TRITON_VERBOSE", "0")
    os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "0")
    os.environ.setdefault("TORCH_COMPILE_DEBUG", "0")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "16")
    os.environ.setdefault("TORCHINDUCTOR_AUTOTUNE_NUM_CHOICES_DISPLAYED", "0")
    os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_REPORT_CHOICES_STATS", "0")
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128"
    )

    # Silence known non-actionable compile warnings.
    warnings.filterwarnings("ignore", message=".*pow_by_natural.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    warnings.filterwarnings(
        "ignore", message=".*save_cache_artifacts.*", category=Warning
    )

    # Suppress torch.compile and Triton verbose output at the logger level
    for name in [
        "torch._dynamo",
        "torch._functorch",
        "torch._inductor",
        "torch._inductor.autotune_process",
        "torch.utils._sympy.interp",
        "triton",
        "torch.cuda",
    ]:
        logging.getLogger(name).setLevel(logging.ERROR)
    logging.getLogger("torch._inductor.select_algorithm").setLevel(logging.CRITICAL)

    # Only call set_logs when TORCH_LOGS env var is not already set
    if "TORCH_LOGS" not in os.environ:
        try:
            torch._logging.set_logs(dynamo=logging.ERROR, inductor=logging.ERROR)  # type: ignore[attr-defined]
        except Exception:
            logger.debug("Unable to configure torch logging", exc_info=True)


def _silence_resource_tracker() -> None:
    """Silence harmless resource_tracker KeyError tracebacks at exit."""
    try:
        from multiprocessing.resource_tracker import (
            _resource_tracker,  # type: ignore[attr-defined]
        )

        if _resource_tracker._pid is not None:  # type: ignore[union-attr]
            os.kill(_resource_tracker._pid, signal.SIGKILL)  # type: ignore[arg-type]
            os.waitpid(_resource_tracker._pid, 0)  # type: ignore[arg-type]
            _resource_tracker._pid = None  # type: ignore[assignment]
        if _resource_tracker._fd is not None:  # type: ignore[union-attr]
            os.close(_resource_tracker._fd)  # type: ignore[arg-type]
            _resource_tracker._fd = None  # type: ignore[assignment]
    except Exception:
        logger.debug("Failed silencing resource tracker", exc_info=True)


atexit.register(_silence_resource_tracker)
_setup_environment()


# Handled by cli_shared.py


def get_arguments() -> ArgumentParser:
    """Parse command-line arguments for parallel FaciesGAN training.

    Returns
    -------
    ArgumentParser
        Configured argument parser with all training options.
    """
    parser = ArgumentParser()

    # Workspace and hardware:
    cli_shared.add_device_args(parser)

    # Grouped arguments from shared logic
    cli_shared.add_io_args(parser)
    cli_shared.add_network_args(parser)
    cli_shared.add_optimization_args(parser)
    cli_shared.add_pyramid_args(parser)
    cli_shared.add_physics_args(parser)

    # Scale 0 specific overrides:
    cli_shared.add_scale0_args(parser)

    # GAN loss and regularization:
    cli_shared.add_gan_loss_args(parser)

    # Visualization, logging, and advanced compute:
    cli_shared.add_runtime_args(parser)

    # Advanced compute:
    parser.add_argument(
        "--use-profiler",
        action="store_true",
        help="Enable PyTorch profiler and export a chrome trace",
    )

    return parser


def _kill_child_processes() -> None:
    """Kill all descendant processes of the current PID.

    Uses ``psutil`` to find and terminate children gracefully (SIGTERM)
    followed by a SIGKILL if they persist. This prevents leaving the GPU
    driver in a bad state.
    """
    try:
        parent = psutil.Process()
        children = parent.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return

    if not children:
        return

    # Phase 1: SIGTERM — let children flush GPU work
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass

    # Brief grace period for GPU kernels to drain
    _, alive = psutil.wait_procs(children, timeout=0.5)

    # Phase 2: SIGKILL any survivors
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass


def _post_process_args(options: TrainingOptions) -> None:
    """Apply manual overrides and handle incompatible flags in options."""
    # Handle --gradient-checkpoint (incompatible with torch.compile)
    if getattr(options, "gradient_checkpointing", False):
        # torch.compile reorders saved tensors in a way that breaks
        # checkpoint recomputation metadata — disable it when
        # checkpointing is active.
        options.compile_backend = False


def _setup_output_dir(options: TrainingOptions) -> None:
    """Configure the output directory and initialize logging."""
    if getattr(options, "output_fullpath", None):
        options.output_path = options.output_fullpath
    else:
        timestamp = datetime.now(tz.tzlocal()).strftime("%Y_%m_%d_%H_%M_%S")
        options.output_path = str(Path(options.output_path) / timestamp)

    if device_manager.is_main_process:
        utils.create_dirs(options.output_path)

        # Save the input parameters options
        output_path = Path(options.output_path)
        options_file = output_path / CheckpointFilenames.OPTIONS
        with options_file.open("w") as file:
            json.dump(vars(options), file, indent=4)  # type: ignore

        if getattr(options, "enable_logging", False):
            init_output_logging(str(output_path / "log.txt"))

    # Synchronise so non-zero ranks wait for rank 0 to create output dir
    if device_manager.is_distributed:
        dist.barrier()  # type: ignore[arg-type]
    if not device_manager.is_main_process:
        utils.create_dirs(options.output_path)


def _tune_performance() -> None:
    """Enable cuDNN autotuner and other backend performance optimizations."""
    if not device_manager.is_cuda:
        return

    torch.backends.cudnn.benchmark = True

    # Allow TF32 for faster matmuls on compatible NVIDIA GPUs
    for attr in [
        "torch.backends.cuda.matmul.allow_tf32",
        "torch.backends.cudnn.allow_tf32",
    ]:
        try:
            parts = attr.split(".")
            obj = torch
            for p in parts[1:-1]:
                obj = getattr(obj, p)
            setattr(obj, parts[-1], True)
        except (AttributeError, Exception):
            logger.debug("Failed to tune backend attribute %s", attr, exc_info=True)

    # Use TF32 precision globally for matmuls (Ampere+ GPUs)
    try:
        torch.set_float32_matmul_precision("high")  # type: ignore
    except (AttributeError, Exception):
        logger.debug("Failed to set float32 matmul precision", exc_info=True)

    # Reasonable default for intra-op threads to avoid oversubscription
    try:
        cpu_threads = min(4, max(1, (os.cpu_count() or 1) // 2))
        torch.set_num_threads(cpu_threads)
    except Exception:
        logger.debug("Failed to set CPU thread count", exc_info=True)


def _report_failure(exc: Exception) -> None:
    """Log training failure details including tracebacks and GPU memory stats."""
    import traceback

    rank_label = (
        f"[rank {device_manager.rank}] " if device_manager.is_distributed else ""
    )
    # Always print to stderr directly — logging may be partially disabled.
    print(
        f"\n{rank_label}{'=' * 60}\n"
        f"{rank_label}TRAINING FAILED - cleaning up\n"
        f"{rank_label}{'=' * 60}",
        flush=True,
    )
    traceback.print_exc()
    logger.error(
        "\n%s%s\n%sTRAINING FAILED - cleaning up\n%s%s",
        rank_label,
        "=" * 60,
        rank_label,
        rank_label,
        "=" * 60,
    )
    logger.exception("Unhandled exception during training")

    # Surface CUDA memory stats when the failure looks like OOM.
    if torch.cuda.is_available():
        try:
            dev = device_manager.device
            alloc = float(torch.cuda.memory_allocated(dev) / (1024**3))
            reserved = float(torch.cuda.memory_reserved(dev) / (1024**3))
            peak = float(torch.cuda.max_memory_allocated(dev) / (1024**3))
            total = float(torch.cuda.get_device_properties(dev).total_memory / (1024**3))  # type: ignore
            msg = (
                f"{rank_label}CUDA memory: alloc={alloc:.2f}G  "
                f"reserved={reserved:.2f}G  peak={peak:.2f}G  total={total:.2f}G"
            )
            print(msg, flush=True)
            logger.error(
                "%sCUDA memory: alloc=%.2fG  reserved=%.2fG  peak=%.2fG  total=%.2fG",
                rank_label,
                alloc,
                reserved,
                peak,
                total,
            )
        except Exception:
            logger.debug("Failed collecting CUDA failure stats", exc_info=True)


def main() -> None:
    """Run parallel FaciesGAN training.

    Orchestrates argument parsing, environment setup, performance tuning,
    and the training loop.
    """
    # 1. Argument parsing and post-processing
    parser = get_arguments()
    options = parser.parse_args(namespace=TrainingOptions())
    options.post_process()
    _post_process_args(options)

    # When file logging is disabled, suppress INFO/DEBUG noise but keep
    # WARNING+ visible so _report_failure (which uses logger.error/exception)
    # always surfaces tracebacks on the console.
    if not getattr(options, "enable_logging", False):
        logging.disable(logging.INFO)

    # 2. Global signals and device initialization
    signal.signal(signal.SIGTERM, lambda *_: os._exit(1))  # type: ignore
    device_manager.initialize(
        gpu_id=options.gpu_device,
        use_cpu=getattr(options, "use_cpu", False),
        manual_seed=options.manual_seed,
    )

    # Ensure logger prints INFO+ to the console by default so user-facing
    # informational messages (tables, progress, etc.) remain visible.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
    )

    # 3. Output directory and logging
    _setup_output_dir(options)

    if device_manager.is_main_process:
        print("\n" + "=" * 60)
        print("PARALLEL FACIESGAN TRAINING")
        print("=" * 60)
        print(f"Device: {device_manager.device}")
        if device_manager.is_distributed:
            world_size = device_manager.world_size
            print(f"DDP training: {world_size} processes")
        print(f"Training scales: {options.start_scale} to {options.stop_scale}")
        print(f"Output path: {options.output_path}")
        print("=" * 60 + "\n")

    # 4. Performance tuning
    _tune_performance()

    # 5. Training loop with error handling and cleanup
    trainer: Trainer | None = None
    try:
        trainer = Trainer(options)

        # Load Inductor cache artifacts
        if getattr(options, "compile_backend", False):
            try:
                cache_path = Path(options.output_path) / "inductor_cache.bin"
                if cache_path.is_file():
                    torch.compiler.load_cache_artifacts(cache_path)  # type: ignore
            except Exception:
                logger.debug("Failed to load inductor cache artifacts", exc_info=True)

        # Resume from checkpoint
        if options.start_scale > 0:
            trainer.load(options.output_path, until_scale=options.start_scale - 1)

        # Register cleanup for child processes
        atexit.register(_kill_child_processes)

        # Run trainer (optionally with profiler)
        if getattr(options, "use_profiler", False):
            from torch.profiler import ProfilerActivity, profile

            trace_file = str(Path(options.output_path) / "profiler_trace.json")
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
            except Exception:
                logger.debug("Failed exporting profiler trace", exc_info=True)
        else:
            trainer.train()

    except Exception as e:
        _report_failure(e)

        # Distributed cleanup
        if device_manager.is_distributed:
            device_manager.synchronize()
            os._exit(1)
        raise

    finally:
        # Background worker and DDP teardown
        try:
            from background_workers import BackgroundWorker

            bw = BackgroundWorker()
            bw.wait_pending(timeout=120.0)
            bw.shutdown(wait=False)
        except Exception:
            logger.debug("Background worker shutdown failed", exc_info=True)

        if device_manager.is_distributed:
            try:
                # Cleanup trainer and model to avoid NCCL destruction issues
                if trainer is not None:
                    del trainer  # type: ignore[undefined-variable]
                import gc

                gc.collect()

                device_manager.synchronize()
                device_manager.release_accelerator_memory()

                if dist.is_initialized():
                    dist.barrier()  # type: ignore
                dist.destroy_process_group()
            except Exception:
                logger.debug("Distributed cleanup failed", exc_info=True)

            # Final process cleanup
            try:
                from torch._inductor.async_compile import shutdown_compile_workers

                atexit.unregister(shutdown_compile_workers)
            except Exception:
                logger.debug("Failed to unregister compile workers", exc_info=True)
            _kill_child_processes()

    if device_manager.is_main_process:
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 60 + "\n")

    # Persist Inductor cache artifacts
    if device_manager.is_main_process and getattr(options, "compile_backend", False):
        try:
            torch.compiler.save_cache_artifacts()
        except Exception:
            logger.debug("Failed saving inductor cache artifacts", exc_info=True)


if __name__ == "__main__":
    import lovely_tensors as lt  # type: ignore

    lt.monkey_patch()
    main()
