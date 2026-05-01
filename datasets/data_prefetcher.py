"""Data prefetching utility for asynchronous GPU batch preparation.

This module provides the `TorchDataPrefetcher` which wraps a PyTorch `DataLoader`
to overlap CPU data loading with GPU computation using a dedicated CUDA stream.
"""

from __future__ import annotations

from typing import Iterator, cast

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

import utils

from .pyramids_batch import Batch, PyramidsBatch


class TorchDataPrefetcher:
    """Wraps a :class:`torch.utils.data.DataLoader` and preloads the next
    batch while the current one is being processed.

    Parameters
    ----------
    loader: DataLoader[tuple[int, Batch] | Batch]
        A PyTorch ``DataLoader`` instance to iterate over.
    scale_indices : tuple[int, ...]
        Sequence of scale indices (0-based) to prepare data for.
    device : torch.device, optional
        Device to move tensors to, by default CPU.
    """

    def __init__(
        self,
        loader: DataLoader[tuple[int, Batch] | Batch],
        scale_indices: tuple[int, ...],
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.loader = iter(loader)
        self.scale_indices = scale_indices
        self.device = device

        self._stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda"
            else None
        )
        self.next_batch = None
        self.next_prepared = None
        # List of dataset sample indices contained in the most recent batch
        # (one index per batch element). Set to None when unavailable.
        self.last_seen_indices: torch.Tensor | None = None
        # All sample indices observed so far in the current iteration.
        self.seen_indices: list[torch.Tensor] = []
        self.preload()

    @property
    def stream(self) -> torch.cuda.Stream | None:
        """Return the CUDA stream used for prefetching."""
        return self._stream

    def preload(self) -> None:
        """Preload the next batch and queue preparation on the stream."""
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            self.next_prepared = None
            return

        if self.next_batch:
            if self._stream:
                with torch.cuda.stream(self._stream):
                    self.next_prepared = self.prepare_batch_async(self.next_batch)
            else:
                self.next_prepared = self.prepare_batch_async(self.next_batch)
        else:
            self.next_prepared = None

    def _to_pyramid(
        self, component: tuple[torch.Tensor, ...]
    ) -> dict[int, torch.Tensor]:
        """Move a tuple of per-scale tensors to the target device."""
        if not component:
            return {}
        return {
            idx: utils.to_device(
                component[idx], self.device, channels_last=True, non_blocking=True
            )
            for idx in self.scale_indices
            if idx < len(component)
        }

    def _coerce_indices(self, values: object) -> torch.Tensor:
        if isinstance(values, torch.Tensor):
            return values.detach().to(self.device)
        return torch.as_tensor(values, device=self.device, dtype=torch.long)

    def prepare_batch_async(self, batch: Batch) -> PyramidsBatch:
        """Perform batch preparation logic asynchronously.

        Moves facies, wells, and seismic data to the target device and
        computes masks from well data if not already provided in the batch.

        Parameters
        ----------
        batch : Batch
            The raw batch from the DataLoader.

        Returns
        -------
        PyramidsBatch
            A tuple of (facies, wells, masks, seismic) dictionaries.
        """
        # Expect either:
        #  - (indices_tensor, facies, wells, masks, seismic)
        #  - (facies, wells, masks, seismic)
        if len(batch) >= 5:
            indexed_batch = cast(tuple[object, object, object, object, object], batch)
            first = batch[0]
            if (
                isinstance(first, torch.Tensor)
                and first.dim() == 1
                and first.dtype in (torch.int64, torch.int32)
            ):
                self.last_seen_indices = self._coerce_indices(first)
                self.seen_indices.append(self.last_seen_indices)
                facies, wells, masks, seismic = cast(
                    tuple[tuple[torch.Tensor, ...], ...], indexed_batch[1:]
                )
            else:
                self.last_seen_indices = None
                facies, wells, masks, seismic = batch[:4]
        elif len(batch) == 2:
            # Case: (indices, Batch_as_named_tuple)
            self.last_seen_indices = self._coerce_indices(batch[0])
            self.seen_indices.append(self.last_seen_indices)
            facies, wells, masks, seismic = cast(
                tuple[tuple[torch.Tensor, ...], ...], batch[1]
            )
        else:
            self.last_seen_indices = None
            facies, wells, masks, seismic = batch

        # Move primary components to device
        facies_pyramid = self._to_pyramid(facies)
        wells_pyramid = self._to_pyramid(wells)
        seismic_pyramid = self._to_pyramid(seismic)

        # Handle masks: use provided masks if available, otherwise compute from wells
        if masks and len(masks) > 0 and masks[0].numel() > 0:
            masks_pyramid = self._to_pyramid(masks)
        elif wells_pyramid:
            # Masks are computed from the device-resident well tensors.
            # In One-Hot Tanh, background is -1.0. We check for any channel > -0.5.
            masks_pyramid = {
                idx: (w[:, 1:, ...].max(dim=1, keepdim=True).values > -0.5).float()
                for idx, w in wells_pyramid.items()
            }
        else:
            masks_pyramid = {}

        return (
            (
                self.last_seen_indices
                if self.last_seen_indices is not None
                else torch.arange(
                    facies[0].shape[0] if facies else 0,
                    device=self.device,
                    dtype=torch.long,
                )
            ),
            facies_pyramid,
            wells_pyramid,
            masks_pyramid,
            seismic_pyramid,
        )

    def _fetch_next(self) -> PyramidsBatch | None:
        """Return the next batch and trigger loading of the subsequent one."""
        if self._stream:
            torch.cuda.current_stream().wait_stream(self._stream)  # type: ignore

        batch = self.next_batch
        prepared = self.next_prepared

        if batch is not None:
            self.preload()

        return prepared

    def __iter__(self) -> Iterator[PyramidsBatch]:
        """Yield prepared batches from the prefetcher.

        Yields
        ------
        PyramidsBatch
            The next prepared batch of multi-scale tensors.
        """
        prepared = self._fetch_next()
        while prepared is not None:
            yield prepared
            prepared = self._fetch_next()

    def __repr__(self) -> str:
        return (
            f"TorchDataPrefetcher(device='{self.device}', "
            f"scale_indices={self.scale_indices})"
        )


def gather_seen_indices(prefetcher: TorchDataPrefetcher) -> list[int]:
    """Return dataset indices seen by all ranks during the current iteration.

    In DDP, each rank processes different data. We gather all indices
    to all ranks so that visualization can access any sample from the
    full dataset that was seen in this epoch.
    """
    local_indices: list[torch.Tensor] = prefetcher.seen_indices
    if not local_indices:
        return []

    if not dist.is_available() or not dist.is_initialized():
        # Move back to CPU only at the end of epoch
        return torch.cat(local_indices).cpu().flatten().tolist()  # type: ignore

    # Move indices to a single contiguous GPU tensor for gathering
    local_tensor = torch.cat(local_indices).to(prefetcher.device)

    world_size = dist.get_world_size()
    gathered_tensors: list[torch.Tensor] = [
        torch.zeros_like(local_tensor) for _ in range(world_size)
    ]

    # Collective call: all ranks must participate.
    dist.all_gather(gathered_tensors, local_tensor)  # type: ignore

    # Single GPU->CPU transfer for all gathered indices
    return torch.cat(gathered_tensors).cpu().flatten().tolist()  # type: ignore
