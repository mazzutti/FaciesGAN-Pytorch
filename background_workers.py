"""Background worker helpers to offload CPU-bound visualization and I/O.

This module exposes a :class:`concurrent.futures.ThreadPoolExecutor`-backed
singleton and helper ``submit_*`` functions that run heavy image processing
and saving in background threads so the main training loop stays responsive.

Design notes:
- Arrays passed to worker functions are accessed directly via shared memory
  (no pickling / IPC serialization), which eliminates the ~100 ms per-call
  overhead that a ProcessPoolExecutor with spawn would impose.
- PIL and numpy release the GIL for most operations, so plotting threads run
  concurrently with GPU computation without blocking the training loop.
- CUDA tensors must still be moved to CPU (using ``device_manager.to_cpu()``)
  before submitting to avoid accidental GPU-to-CPU copies inside the worker.
"""

from __future__ import annotations

import atexit
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Self, cast

logger = logging.getLogger(__name__)


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
    normalization_range: tuple[float, float] = (0.0, 1.0),
) -> bool:
    """
    Internal task to save a plot in a background thread.

    Supports 3D, 4D, and 5D tensors/arrays for fake_list, real_arr, and masks_arr:
      - fake_list can be a sequence of tensors/arrays or a single tensor/array
      - (C, H, W), (B, C, H, W), (B, T, C, H, W) for torch/mx/np
      - (B, H, W, C), (B, T, H, W, C) for np arrays (after conversion)
    """
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
        normalization_range=normalization_range,
    )
    return True


def _save_plot_task_from_npy(
    fake_path: str,
    real_path: str,
    stage: int,
    index: int,
    out_dir: str,
    masks_path: str | None = None,
    batch_id: int | None = None,
    normalization_range: tuple[float, float] = (0.0, 1.0),
) -> bool:
    """
    Internal task to load .npy files and save a plot in a background process.

    This keeps numpy loading and I/O off the main training thread.
    """
    import numpy as np

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
        normalization_range=normalization_range,
    )
    return True


# noinspection PyBroadException
def _save_image_task(img_np: Any, out_path: str) -> bool:
    """Internal task to save a single image in a background thread."""
    from PIL import Image

    try:
        # Convert to uint8 for saving if not already
        if img_np.dtype != "uint8":
            img_np = (img_np * 255).astype("uint8")
        Image.fromarray(img_np).save(out_path)
        return True
    except Exception:
        return False


# noinspection PyBroadException
class BackgroundWorker:
    """Singleton manager for a process pool that offloads CPU-bound tasks.

    Features:
    - Singleton instance (calling BackgroundWorker() returns same instance).
    - Bounded pending-job queue to avoid unbounded memory growth.
    - Tracks pending futures and exposes wait/shutdown helpers.
    - Logs exceptions raised by background tasks.
    """

    _instance: "BackgroundWorker | None" = None
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
        return cast(Self, cls._instance)

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

        self._max_workers = int(max_workers)
        # ThreadPoolExecutor: zero IPC serialization cost — threads share
        # memory directly, so large numpy arrays are never pickled.  PIL and
        # numpy both release the GIL, so plotting tasks run concurrently with
        # the training loop without blocking GPU computation.
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=max_workers)

        # Pending futures tracking and coordination
        self._pending: set[Future[bool]] = set()
        self._pending_cond = threading.Condition()
        self._max_pending = int(max_pending)
        self._initialized = True
        atexit.register(self.shutdown, wait=True)

    def _restart_pool(self) -> None:
        """Recreate the thread pool after an unexpected failure.

        Discards all pending futures (they are already lost) and creates a
        fresh ThreadPoolExecutor so subsequent submissions can succeed.
        """
        logger.warning("BackgroundWorker: thread pool is broken — restarting pool")
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass
        # Discard all stale pending futures — the outputs are unrecoverable.
        with self._pending_cond:
            self._pending.clear()
            self._pending_cond.notify_all()
        self._executor = ThreadPoolExecutor(max_workers=self._max_workers)

    def _submit_with_retry(self, fn: Any, *args: Any) -> "Future[bool]":
        """Submit *fn* to the pool, restarting it once on RuntimeError."""
        try:
            return self._executor.submit(fn, *args)
        except RuntimeError:
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

    def _wait_for_slot(
        self, wait_if_full: bool, timeout: float | None
    ) -> "Future[bool] | None":
        """Wait until a slot is free in the pending queue (must hold _pending_cond).

        Returns a completed ``Future(False)`` when the queue is full and the
        caller should bail immediately, or ``None`` when a slot is available.
        """
        if self._max_pending <= 0:
            return None
        if not wait_if_full and len(self._pending) >= self._max_pending:
            bail: Future[bool] = Future()
            bail.set_result(False)
            return bail
        if timeout is None:
            while len(self._pending) >= self._max_pending:
                self._pending_cond.wait()
        else:
            end = time.time() + timeout
            while len(self._pending) >= self._max_pending:
                remaining = end - time.time()
                if remaining <= 0:
                    bail = Future()
                    bail.set_result(False)
                    return bail
                self._pending_cond.wait(timeout=remaining)
        return None

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
        normalization_range: tuple[float, float] = (0.0, 1.0),
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
        Move all tensors to CPU (using ``device_manager.to_cpu()``) before
        calling this method rather than passing CUDA tensors directly into
        the executor to avoid pickling/serialization issues with GPU-backed
        storage.
        """
        with self._pending_cond:
            bail = self._wait_for_slot(wait_if_full, timeout)
            if bail is not None:
                return bail
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
                normalization_range,
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
        normalization_range: tuple[float, float] = (0.0, 1.0),
    ) -> Future[bool]:
        """
        Submit a plot job by passing .npy paths to the process pool.

        This avoids loading numpy arrays in the main training process.
        """
        with self._pending_cond:
            bail = self._wait_for_slot(wait_if_full, timeout)
            if bail is not None:
                return bail
            fut = self._submit_with_retry(
                _save_plot_task_from_npy,
                str(fake_path),
                str(real_path),
                int(stage),
                int(index),
                str(out_dir),
                str(masks_path) if masks_path else None,
                normalization_range,
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

    def submit_save_image(self, img_np: Any, out_path: str) -> "Future[bool]":
        """Submit a simple image-save job to the background worker.

        Returns immediately if the queue is full (non-blocking) rather than
        stalling the training loop.
        """
        with self._pending_cond:
            # Don't block the training loop if the queue is full — skip the
            # save instead.  This prevents DDP rank 0 from timing out while
            # other ranks proceed to the next collective.
            bail = self._wait_for_slot(wait_if_full=False, timeout=None)
            if bail is not None:
                return bail
            fut = self._submit_with_retry(_save_image_task, img_np, str(out_path))
            self._pending.add(fut)
            fut.add_done_callback(self._on_done)  # type: ignore
            return fut


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
    normalization_range: tuple[float, float] = (0.0, 1.0),
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
        normalization_range=normalization_range,
    )


def submit_save_image(img_np: Any, out_path: str) -> Future[bool]:
    """Submit a simple image-save job using the module-level BackgroundWorker."""
    return BackgroundWorker().submit_save_image(img_np, out_path)
