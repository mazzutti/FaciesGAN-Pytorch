"""Global device management for FaciesGAN.

This module provides a singleton ``DeviceManager`` to centralize device
selection (CPU, CUDA) and distributed training rank tracking across
the entire codebase.
"""

import os
import threading
from typing import Any, cast

import torch
import torch.distributed as dist
import numpy as np

from enums import DeviceType


class _ThreadLocalState(threading.local):
    device: torch.device | None


class DeviceManager:
    """Singleton to manage global device state and distributed settings."""

    _instance: "DeviceManager | None" = None
    _thread_local: _ThreadLocalState

    def __new__(cls) -> "DeviceManager":
        if cls._instance is None:
            cls._instance = super(DeviceManager, cls).__new__(cls)
            cls._instance._initialized = False
            cls._instance._thread_local = _ThreadLocalState()
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._thread_local.device = None
        # Distributed environment variables (set by torchrun)
        self._local_rank = int(os.environ.get("LOCAL_RANK", -1))
        self._world_size = int(os.environ.get("WORLD_SIZE", 1))
        self._rank = int(os.environ.get("RANK", 0))
        self._is_distributed = self._local_rank != -1
        self._initialized = True

    def initialize(
        self, gpu_id: int = 0, use_cpu: bool = False, manual_seed: int | None = None
    ) -> torch.device:
        """Initialize the global device with priority: CUDA > CPU."""
        if manual_seed is not None:
            import utils

            utils.set_seed(manual_seed)

        # Set OMP_NUM_THREADS before torchrun can default it to 1.
        if "OMP_NUM_THREADS" not in os.environ:
            omp_threads = max(
                1, (os.cpu_count() or 1) // max(1, self.accelerator_count)
            )
            os.environ["OMP_NUM_THREADS"] = str(omp_threads)

        if use_cpu:
            self._thread_local.device = torch.device(DeviceType.CPU)
        elif self._is_distributed:
            # Distributed always implies CUDA in this codebase
            self._thread_local.device = torch.device(
                f"{DeviceType.CUDA}:{self._local_rank}"
            )
            torch.cuda.set_device(self._local_rank)

            if not dist.is_initialized():
                # Tell NCCL to surface errors asynchronously
                os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "Distributed training requires CUDA. No CUDA devices found."
                    )

                # Use a 15-minute timeout so a DDP desync surfaces as an error
                from datetime import timedelta

                from enums import DdpBackend

                dist.init_process_group(
                    backend=DdpBackend.NCCL,
                    timeout=timedelta(minutes=15),
                    device_id=self._thread_local.device,
                )

                torch.cuda.set_per_process_memory_fraction(0.90)  # type: ignore

                # Seed the RNGs
                if manual_seed is not None:
                    torch.manual_seed(manual_seed)  # type: ignore
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(manual_seed + self.rank)
        elif torch.cuda.is_available():
            self._thread_local.device = torch.device(f"{DeviceType.CUDA}:{gpu_id}")
            torch.cuda.set_device(gpu_id)
        else:
            self._thread_local.device = torch.device(DeviceType.CPU)

        return self._thread_local.device

    def get_or_initialize(self, gpu_id: int | None = None) -> torch.device:
        """Returns the current initialized device or initializes one if a GPU is explicitly requested."""
        if gpu_id is not None:
            return self.initialize(gpu_id=gpu_id)
        return self.device

    @property
    def device(self) -> torch.device:
        """Get the global (thread-local) device instance."""
        if (
            not hasattr(self._thread_local, "device")
            or self._thread_local.device is None
        ):
            # Fallback for uninitialized access
            return self.initialize()
        return self._thread_local.device

    @property
    def is_cuda(self) -> bool:
        """Return True if the current device is CUDA."""
        return self.device.type == DeviceType.CUDA

    @property
    def is_main_process(self) -> bool:
        """Return True if this is the main process (rank 0)."""
        return self.rank == 0

    @property
    def local_rank(self) -> int:
        """Return the local rank of the current process."""
        return self._local_rank

    def synchronize(self) -> None:
        """Synchronize the current device.

        For CUDA devices, this waits for all kernels and async transfers
        on the current stream to complete. For CPU, this is a no-op.
        """
        if self.is_cuda:
            torch.cuda.current_stream().synchronize()

    @property
    def rank(self) -> int:
        """Return the global rank of the current process."""
        if self._is_distributed and dist.is_initialized():
            return dist.get_rank()
        return self._rank

    @property
    def world_size(self) -> int:
        """Return the total number of processes in the distributed group."""
        if self._is_distributed and dist.is_initialized():
            return dist.get_world_size()
        return self._world_size

    @property
    def is_distributed(self) -> bool:
        """Return True if running in distributed mode."""
        return self._is_distributed

    @property
    def accelerator_count(self) -> int:
        """Get the number of available accelerators (GPUs)."""
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        return 0

    def warmup_all_accelerators(self) -> None:
        """Pre-initialize CUDA contexts on every GPU from the main thread.
        This prevents worker threads from hitting 'operation not permitted' errors.
        """
        if not self.is_cuda:
            return

        num_gpus = self.accelerator_count
        for gpu_id in range(num_gpus):
            torch.cuda.init()
            self.initialize(gpu_id=gpu_id)
            torch.zeros(1, device=self.device)

    def to_cpu(self, data: Any, non_blocking: bool = False) -> Any:
        """Recursively move tensors in a collection to CPU and detach them.

        Args:
            data: A tensor, list, tuple, or dict containing tensors.
            non_blocking: If True, use non-blocking transfer where supported.

        Returns:
            The same collection with all tensors moved to CPU and detached.
        """
        if isinstance(data, torch.Tensor):
            return data.detach().to(DeviceType.CPU, non_blocking=non_blocking)
        if isinstance(data, dict):
            data_dict = cast(dict[Any, Any], data)
            return {
                key: self.to_cpu(value, non_blocking=non_blocking)
                for key, value in data_dict.items()
            }
        if isinstance(data, list):
            return [
                self.to_cpu(value, non_blocking=non_blocking)
                for value in cast(list[Any], data)
            ]
        if isinstance(data, tuple):
            return tuple(
                self.to_cpu(value, non_blocking=non_blocking)
                for value in cast(tuple[Any, ...], data)
            )
        return data

    def to_numpy(self, data: Any) -> np.ndarray | Any:
        """Recursively convert tensors in a collection to NumPy arrays.

        Tensors are detached and moved to CPU before conversion.

        Args:
            data: A tensor, list, tuple, or dict containing tensors.

        Returns:
            The same collection with all tensors replaced by NumPy arrays.
        """
        if isinstance(data, torch.Tensor):
            return self.to_cpu(data).numpy()
        if isinstance(data, dict):
            data_dict = cast(dict[Any, Any], data)
            return {key: self.to_numpy(value) for key, value in data_dict.items()}
        if isinstance(data, list):
            return [self.to_numpy(value) for value in cast(list[Any], data)]
        if isinstance(data, tuple):
            return tuple(self.to_numpy(value) for value in cast(tuple[Any, ...], data))
        return data

    def to_device(
        self, data: Any, channels_last: bool = False, non_blocking: bool = False
    ) -> Any:
        """Recursively move tensors in a collection to the managed device.

        Args:
            data: A tensor, list, tuple, or dict containing tensors.
            channels_last: If True, convert tensors to channels_last memory format (CUDA only).
            non_blocking: If True, use non-blocking transfer where supported.

        Returns:
            The same collection with all tensors moved to the managed device.
        """
        if isinstance(data, torch.Tensor):
            if self.is_cuda:
                if channels_last:
                    return data.to(self.device, non_blocking=non_blocking).contiguous(
                        memory_format=torch.channels_last
                    )
                return data.to(self.device, non_blocking=non_blocking).contiguous()
            return data.to(self.device)

        if isinstance(data, dict):
            data_dict = cast(dict[Any, Any], data)
            return {
                key: self.to_device(value, channels_last, non_blocking)
                for key, value in data_dict.items()
            }
        if isinstance(data, list):
            return [
                self.to_device(value, channels_last, non_blocking)
                for value in cast(list[Any], data)
            ]
        if isinstance(data, tuple):
            return tuple(
                self.to_device(value, channels_last, non_blocking)
                for value in cast(tuple[Any, ...], data)
            )
        return data

    def to_gpu(
        self, data: Any, channels_last: bool = False, non_blocking: bool = False
    ) -> Any:
        """Recursively move tensors in a collection to the managed device (GPU).

        Alias for to_device.
        """
        return self.to_device(
            data, channels_last=channels_last, non_blocking=non_blocking
        )

    def release_accelerator_memory(self) -> None:
        """Release unused accelerator (GPU) memory back to the OS.

        Calls the caching allocator to release unused blocks.
        ``empty_cache()`` already triggers an implicit device sync, so
        an explicit ``synchronize()`` is unnecessary. GC is skipped
        because this is only called between scale groups and the
        allocator handles freed tensors without a Python GC pass.
        """
        if self.is_cuda:
            torch.cuda.empty_cache()

    def create_stream(self) -> Any:
        """Create a new CUDA stream on the managed device if available.

        Returns:
            A new torch.cuda.Stream or None if not using CUDA.
        """
        if self.is_cuda:
            return torch.cuda.Stream(device=self.device)
        return None

    def get_rng_state_dict(self) -> dict[str, Any]:
        """Collect RNG states for Python, PyTorch, and CUDA if available.

        Returns:
            Dictionary with RNG states for the respective ecosystems.
        """
        import random

        states: dict[str, Any] = {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
        }
        if self.is_cuda:
            states[DeviceType.CUDA] = torch.cuda.get_rng_state_all()
        return states

    def set_rng_state_dict(self, states: dict[str, Any]) -> None:
        """Restore RNG states for Python, PyTorch, and CUDA.

        Args:
            states: Dictionary containing the RNG states.
        """
        import random

        if "python" in states:
            try:
                random.setstate(states["python"])
            except Exception:
                pass

        if "torch" in states:
            try:
                torch.set_rng_state(cast(torch.Tensor, states["torch"]))
            except Exception:
                pass

        if DeviceType.CUDA in states and self.is_cuda:
            try:
                torch.cuda.set_rng_state_all(
                    cast(list[torch.Tensor], states[DeviceType.CUDA])
                )
            except Exception:
                pass


# Global singleton instance
device_manager: DeviceManager = DeviceManager()
