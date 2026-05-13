"""Data prefetching utility for asynchronous GPU batch preparation.

This module provides the `TorchDataPrefetcher` which wraps a PyTorch `DataLoader`
to overlap CPU data loading with GPU computation using a dedicated CUDA stream.
"""

from __future__ import annotations

from typing import Iterator, cast

import torch
import torch.distributed as dist

import utils

from .pyramids_batch import Batch, IDataLoader, PyramidsBatch


class DataPrefetcher:
    """Wraps a :class:`torch.utils.data.DataLoader` and preloads the next
    batch while the current one is being processed.

    Parameters
    ----------
    loader: IDataLoader
        A PyTorch ``DataLoader`` instance to iterate over.
    scale_indices : tuple[int, ...]
        Sequence of scale indices (0-based) to prepare data for.
    device : torch.device, optional
        Device to move tensors to, by default CPU.
    """

    def __init__(
        self,
        loader: IDataLoader,
        scale_indices: tuple[int, ...],
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.loader: Iterator[tuple[int, Batch] | Batch] = iter(loader)
        self.scale_indices = scale_indices
        self.device = device

        self._stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda"
            else None
        )
        self.next_batch: tuple[int, Batch] | Batch | None = None
        self.next_prepared: PyramidsBatch | None = None
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
        """Convert indices to a tensor on the target device.

        Parameters
        ----------
        values : object
            Either a torch.Tensor or a value that can be converted to a tensor.

        Returns
        -------
        torch.Tensor
            Indices tensor of dtype int64 on the target device.
        """
        if isinstance(values, torch.Tensor):
            return values.detach().to(self.device)
        return torch.as_tensor(values, device=self.device, dtype=torch.long)

    def prepare_batch_async(self, batch: tuple[int, Batch] | Batch) -> PyramidsBatch:
        """Perform batch preparation logic asynchronously.

        Moves facies, wells, and seismic data to the target device and
        computes masks from well data if not already provided in the batch.

        Parameters
        ----------
        batch : tuple[int, Batch] | Batch
            The raw batch from the DataLoader. It may be a ``Batch`` named
            tuple or a ``(indices, Batch)`` pair depending on the collate
            function.

        Returns
        -------
        PyramidsBatch
            A tuple of ``(indices, facies, wells, masks, seismic)`` where
            the per-scale components are dictionaries keyed by scale index.
        """
        # Expect either:
        #  - (indices_tensor, facies, wells, masks, seismic)
        #  - (facies, wells, masks, seismic)
        batch_tuple: tuple[object, ...] = cast(tuple[object, ...], batch)

        if len(batch_tuple) >= 5:
            first = batch_tuple[0]
            if (
                isinstance(first, torch.Tensor)
                and first.dim() == 1
                and first.dtype in (torch.int64, torch.int32)
            ):
                self.last_seen_indices = self._coerce_indices(first)
                self.seen_indices.append(self.last_seen_indices)
                facies = cast(tuple[torch.Tensor, ...], batch_tuple[1])
                wells = cast(tuple[torch.Tensor, ...], batch_tuple[2])
                masks = cast(tuple[torch.Tensor, ...], batch_tuple[3])
                seismic = cast(tuple[torch.Tensor, ...], batch_tuple[4])
            else:
                self.last_seen_indices = None
                facies = cast(tuple[torch.Tensor, ...], batch_tuple[0])
                wells = cast(tuple[torch.Tensor, ...], batch_tuple[1])
                masks = cast(tuple[torch.Tensor, ...], batch_tuple[2])
                seismic = cast(tuple[torch.Tensor, ...], batch_tuple[3])
        elif len(batch_tuple) == 2:
            # Case: (indices, Batch_as_named_tuple)
            self.last_seen_indices = self._coerce_indices(batch_tuple[0])
            self.seen_indices.append(self.last_seen_indices)
            facies_batch = cast(Batch, batch_tuple[1])
            facies, wells, masks, seismic = (
                facies_batch.facies,
                facies_batch.wells,
                facies_batch.masks,
                facies_batch.seismic,
            )
        else:
            self.last_seen_indices = None
            facies_batch = cast(Batch, batch)
            facies, wells, masks, seismic = (
                facies_batch.facies,
                facies_batch.wells,
                facies_batch.masks,
                facies_batch.seismic,
            )

        # Move primary components to device
        facies_pyramid = self._to_pyramid(facies)
        wells_pyramid = self._to_pyramid(wells)
        seismic_pyramid = self._to_pyramid(seismic)

        # Handle masks: use provided masks if available, otherwise compute from wells
        if masks and len(masks) > 0 and masks[0].numel() > 0:
            masks_pyramid = self._to_pyramid(masks)
        elif wells_pyramid:
            # Masks are computed from the device-resident well tensors.
            # In one-hot [0,1], background is 0.0. We check for any channel > 0.5.
            masks_pyramid = {
                idx: (w[:, 1:, ...].max(dim=1, keepdim=True).values > 0.5).float()
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
            The next prepared batch of multiscale tensors, represented as
            ``(indices, facies, wells, masks, seismic)``.
        """
        prepared = self._fetch_next()
        while prepared is not None:
            yield prepared
            prepared = self._fetch_next()

    def __repr__(self) -> str:
        return (
            f"TorchDataPrefetcher(device='{self.device.type}', "
            f"scale_indices={self.scale_indices})"
        )


def gather_seen_indices(prefetcher: DataPrefetcher) -> list[int]:
    """Return dataset indices seen by all ranks during the current iteration.

    In DDP, each rank processes different data. We gather all indices
    to all ranks so that visualization can access any sample from the
    full dataset that was seen in this epoch.

    Returns
    -------
    list[int]
        A flat list of sample indices gathered across all ranks.
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
