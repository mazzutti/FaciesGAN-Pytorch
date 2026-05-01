"""Background worker helpers to offload CPU-bound visualization and quantization.

This module exposes a small :class:`concurrent.futures.ProcessPoolExecutor`
and helper ``submit_*`` functions that run heavy image processing and
saving in separate processes so the main training loop stays responsive.

Design notes:
    (e.g. moved to CPU with ``tensor.detach().cpu()``) so the worker process
    can safely serialize and operate on them. Avoid passing CUDA tensors
    directly into the process pool to prevent pickling errors.
- Uses ``multiprocessing.get_context('spawn')`` to be compatible across
    platforms (macOS).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import threading
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)


def _configure_python_executable() -> None:
    """Ensure multiprocessing spawn uses the local venv python when available."""
    try:
        import os
        import sys

        # Prefer local .venv if present
        repo_root = os.path.dirname(os.path.abspath(__file__))
        venv_py = os.path.join(repo_root, ".venv", "bin", "python")
        py = None
        if os.path.exists(venv_py):
            py = venv_py
        else:
            venv = os.environ.get("VIRTUAL_ENV", "")
            if venv:
                candidate = os.path.join(venv, "bin", "python")
                if os.path.exists(candidate):
                    py = candidate

        if py:
            sys.executable = py
            os.environ["PYTHONEXECUTABLE"] = py
            try:
                mp.set_executable(py)
            except Exception:
                pass
    except Exception:
        pass


def _save_plot_task(
    fake_list: Any,
    real_arr: Any,
    stage: int,
    index: int,
    out_dir: str,
    masks_arr: Any = None,
    batch_id: int | None = None,
    plot_title: str = "Facies",
    cmap: str = "viridis",
) -> bool:
    """
    Internal task to save a plot in a background process.

    Supports 3D, 4D, and 5D tensors/arrays for fake_list, real_arr, and masks_arr:
      - fake_list can be a sequence of tensors/arrays or a single tensor/array
      - (C, H, W), (B, C, H, W), (B, T, C, H, W) for torch/mx/np
      - (B, H, W, C), (B, T, H, W, C) for np arrays (after conversion)
    """
    import os

    os.environ["FG_NO_TORCH_IMPORT"] = "1"

    # Import locally so the worker process has its own module imports
    from utils import plot_generated_outputs

    # The plotting helper will accept the tensors and perform any
    # conversions internally as needed.
    plot_generated_outputs(
        fake_list,
        real_arr,
        stage,
        index,
        masks_arr,
        out_dir,
        save=True,
        batch_id=batch_id,
        plot_title=plot_title,
        cmap=cmap,
    )
    return True


def _warmup_worker() -> bool:
    """Pre-import heavy dependencies in the worker process.

    Called once at pool creation time so that the first real plotting task
    does not pay the ~3-4 second matplotlib/numpy import cost.
    """
    import os

    os.environ["FG_NO_TORCH_IMPORT"] = "1"
    # Import the plotting module which pulls in matplotlib, numpy, etc.
    try:
        from utils import plot_generated_outputs  # type: ignore[import]
    except Exception:
        pass
    return True


def _save_plot_task_from_npy(
    fake_path: str,
    real_path: str,
    stage: int,
    index: int,
    out_dir: str,
    masks_path: str | None = None,
    batch_id: int | None = None,
) -> bool:
    """
    Internal task to load .npy files and save a plot in a background process.

    This keeps numpy loading and plotting off the main training process.
    """
    import os

    import numpy as np

    os.environ["FG_NO_TORCH_IMPORT"] = "1"

    from utils import plot_generated_outputs

    fake_arr = np.load(fake_path)
    real_arr = np.load(real_path)
    masks_arr = np.load(masks_path) if masks_path else None

    plot_generated_outputs(
        fake_arr,
        real_arr,
        stage,
        index,
        masks_arr,
        out_dir,
        save=True,
        batch_id=batch_id,
    )
    return True


class BackgroundWorker:
    """Singleton manager for a process pool that offloads CPU-bound tasks.

    Features:
    - Singleton instance (calling BackgroundWorker() returns same instance).
    - Bounded pending-job queue to avoid unbounded memory growth.
    - Tracks pending futures and exposes wait/shutdown helpers.
    - Logs exceptions raised by background tasks.
    """

    _instance: BackgroundWorker | None = None
    _instance_lock = threading.Lock()

    def __new__(cls, max_workers: int = 2, max_pending: int = 32) -> "BackgroundWorker":
        """Return the singleton BackgroundWorker instance.

        Implements double-checked locking to ensure thread-safe lazy
        initialization of the singleton instance.
        """
        # Double-checked locking to safely initialize singleton
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, max_workers: int = 2, max_pending: int = 32) -> None:
        """Initialize the process pool and pending-job tracking.

        Parameters
        ----------
        max_workers : int
            Maximum number of worker processes in the pool.
        max_pending : int
            Maximum number of pending futures before callers are blocked.
        """
        # Initialize only once
        if getattr(self, "_initialized", False):
            return

        _configure_python_executable()

        ctx = mp.get_context("spawn")
        self._executor: ProcessPoolExecutor = ProcessPoolExecutor(
            max_workers=max_workers, mp_context=ctx
        )

        # Pending futures tracking and coordination
        self._pending: set[Future[bool]] = set()
        self._pending_cond = threading.Condition()
        self._max_pending = int(max_pending)
        self._initialized = True

        # Pre-warm: spawn worker processes now so they are ready when the
        # first real task is submitted.  Without this, the first submit()
        # triggers a fork+reimport that takes several seconds (importing
        # matplotlib, numpy, etc.).  The warmup task also imports the
        # plotting dependencies so subsequent tasks start instantly.
        # Fire-and-forget: don't block init; the worker starts in parallel
        # with training so it's ready by the time plots are needed.
        try:
            self._warmup_future: Future[bool] | None = self._executor.submit(
                _warmup_worker
            )
        except Exception:
            self._warmup_future = None

    def _restart_pool(self) -> None:
        """Recreate the process pool after a BrokenProcessPool failure.

        Discards all pending futures (they are already lost) and creates a
        fresh ProcessPoolExecutor so subsequent submissions can succeed.
        """
        logger.warning("BackgroundWorker: process pool is broken — restarting pool")
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass
        # Discard all stale pending futures — the outputs are unrecoverable.
        with self._pending_cond:
            self._pending.clear()
            self._pending_cond.notify_all()
        ctx = mp.get_context("spawn")
        self._executor = ProcessPoolExecutor(max_workers=2, mp_context=ctx)

    def _submit_with_retry(self, fn: Any, *args: Any) -> "Future[bool]":
        """Submit *fn* to the pool, restarting it once on BrokenProcessPool."""
        from concurrent.futures.process import BrokenProcessPool

        try:
            return self._executor.submit(fn, *args)
        except BrokenProcessPool:
            self._restart_pool()
            return self._executor.submit(fn, *args)

    def _on_done(self, fut: "Future[bool]") -> None:
        # Callback executed in the main process thread when a Future completes
        try:
            exc = fut.exception()
            if exc is not None:
                logger.exception("Background task failed", exc_info=exc)
        except Exception:
            logger.exception("Error checking future result")
        finally:
            with self._pending_cond:
                self._pending.discard(fut)
                self._pending_cond.notify_all()

    def submit_plot_generated_outputs(
        self,
        fake: Any,
        real: Any,
        stage: int,
        index: int,
        out_dir: str,
        masks: Any = None,
        wait_if_full: bool = True,
        timeout: float | None = None,
        batch_id: int | None = None,
        plot_title: str = "Facies",
        cmap: str = "viridis",
    ) -> Future[bool]:
        """
        Submit a plot job to the process pool (non-blocking by default).

        Supports 3D, 4D, and 5D tensors/arrays for fake_list, real, and masks:
          - fake_list can be a sequence of tensors/arrays or a single tensor/array
          - (C, H, W), (B, C, H, W), (B, T, C, H, W) for torch/mx/np
          - (B, H, W, C), (B, T, H, W, C) for np arrays (after conversion)

        If the number of pending jobs reaches ``max_pending``, the call will
        block until space is available when ``wait_if_full=True``. If
        ``wait_if_full=False`` a completed ``Future`` with a ``False`` result
        is returned immediately.

        Important
        ---------
        (``tensor.detach().cpu()``) before calling this method rather than
        passing CUDA tensors directly into the executor to avoid
        pickling/serialization issues with GPU-backed storage.
        """
        # Wait for available slot if configured
        with self._pending_cond:
            if self._max_pending > 0:
                if not wait_if_full and len(self._pending) >= self._max_pending:
                    fut: Future[bool] = Future()
                    fut.set_result(False)
                    return fut

                if timeout is None:
                    # Wait indefinitely until there's space
                    while len(self._pending) >= self._max_pending:
                        self._pending_cond.wait()
                else:
                    import time

                    # Compute deadline once and wait with the remaining time
                    end = time.time() + timeout
                    while len(self._pending) >= self._max_pending:
                        remaining = end - time.time()
                        if remaining <= 0:
                            fut: Future[bool] = Future()
                            fut.set_result(False)
                            return fut
                        self._pending_cond.wait(timeout=remaining)
            fut = self._submit_with_retry(
                _save_plot_task,
                fake,
                real,
                int(stage),
                int(index),
                str(out_dir),
                masks,
                int(batch_id) if batch_id is not None else None,
                str(plot_title),
                str(cmap),
            )
            # Track and attach callback
            self._pending.add(fut)
            fut.add_done_callback(self._on_done)  # type: ignore
            return fut

    def submit_plot_generated_outputs_from_npy(
        self,
        fake_path: str,
        real_path: str,
        stage: int,
        index: int,
        out_dir: str,
        masks_path: str | None = None,
        wait_if_full: bool = True,
        timeout: float | None = None,
    ) -> Future[bool]:
        """
        Submit a plot job by passing .npy paths to the process pool.

        This avoids loading numpy arrays in the main training process.
        """
        with self._pending_cond:
            if self._max_pending > 0:
                if not wait_if_full and len(self._pending) >= self._max_pending:
                    fut: Future[bool] = Future()
                    fut.set_result(False)
                    return fut

                if timeout is None:
                    while len(self._pending) >= self._max_pending:
                        self._pending_cond.wait()
                else:
                    import time

                    end = time.time() + timeout
                    while len(self._pending) >= self._max_pending:
                        remaining = end - time.time()
                        if remaining <= 0:
                            fut: Future[bool] = Future()
                            fut.set_result(False)
                            return fut
                        self._pending_cond.wait(timeout=remaining)
            fut = self._submit_with_retry(
                _save_plot_task_from_npy,
                str(fake_path),
                str(real_path),
                int(stage),
                int(index),
                str(out_dir),
                str(masks_path) if masks_path else None,
            )
            self._pending.add(fut)
            fut.add_done_callback(self._on_done)  # type: ignore
            return fut

    def pending_count(self) -> int:
        """Return the current number of pending background jobs."""
        with self._pending_cond:
            return len(self._pending)

    def wait_pending(self, timeout: float | None = None) -> None:
        """Block until all pending tasks complete or timeout elapses."""
        with self._pending_cond:
            if timeout is None:
                while self._pending:
                    self._pending_cond.wait()
            else:
                import time

                end = time.time() + timeout
                while self._pending and time.time() < end:
                    self._pending_cond.wait(timeout=end - time.time())

    def shutdown(self, wait: bool = False) -> None:
        """Shutdown the process pool and optionally wait for pending jobs.

        Parameters
        ----------
        wait : bool
            If True, block until all pending tasks complete before shutting
            down the pool.
        """
        try:
            # Optionally wait for pending jobs to complete
            if wait:
                self.wait_pending()
            self._executor.shutdown(wait=wait)
        except Exception:
            logger.exception("Error shutting down BackgroundWorker")


def submit_plot_generated_outputs(
    fake: Any,
    real: Any,
    stage: int,
    index: int,
    out_dir: str,
    masks: Any = None,
    batch_id: int | None = None,
    plot_title: str = "Facies",
    cmap: str = "viridis",
) -> Future[bool]:
    """
    Submit a plot job using the module-level BackgroundWorker.

        Supports 3D, 4D, and 5D tensors/arrays for fake_list, real, and masks:
            - fake_list can be a sequence of tensors/arrays or a single tensor/array
            - (C, H, W), (B, C, H, W), (B, T, C, H, W) for torch/mx/np
            - (B, H, W, C), (B, T, H, W, C) for np arrays (after conversion)

    This wrapper provides backward compatibility for callers that expect a
    module-level function rather than instantiating the singleton class.

    A 30-second timeout is enforced so that a full worker queue cannot
    block the main training loop indefinitely.  Under DDP this prevents
    rank 0 from stalling while other ranks proceed to the next NCCL
    collective — a common source of silent deadlocks.
    """
    # BackgroundWorker is a singleton — calling the constructor returns the
    # shared instance. Use it to submit the job.
    worker = BackgroundWorker()
    return worker.submit_plot_generated_outputs(
        fake,
        real,
        stage,
        index,
        out_dir,
        masks,
        batch_id=batch_id,
        timeout=30.0,
        plot_title=plot_title,
        cmap=cmap,
    )
