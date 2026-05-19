"""Data prefetching utility for asynchronous GPU batch preparation.

This module provides the `DataPrefetcher` which wraps a PyTorch `DataLoader`
to overlap CPU data loading with GPU computation using a dedicated CUDA stream.
"""

from __future__ import annotations

from typing import Iterator, List, Tuple

import numpy as np
import torch
import torch.distributed as dist

from device import device_manager
from typedefs import Batch, IDataLoader, PyramidsBatch, RawBatch


class DataPrefetcher:
    """Wraps a :class:`torch.utils.data.DataLoader` and preloads the next
    batch while the current one is being processed.

    Parameters
    ----------
    loader: IDataLoader
        A PyTorch ``DataLoader`` instance to iterate over.
    scale_indices : tuple[int, ...]
        Sequence of scale indices (0-based) to prepare data for.
    """

    def __init__(
        self,
        loader: IDataLoader,
        scale_indices: tuple[int, ...],
    ) -> None:
        self.loader: Iterator[RawBatch] = iter(loader)
        self.scale_indices = scale_indices

        self._stream = device_manager.create_stream()
        self.next_batch: RawBatch | None = None
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
        """Fetch the next raw batch from the DataLoader and queue GPU preparation.

        This method triggers `prepare_batch_async` on the prefetcher's CUDA stream
        to overlap GPU data movement with CPU computation.
        """
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            self.next_prepared = None
            return

        if self.next_batch:
            if self._stream:
                # Run preparation on the dedicated prefetch stream
                with torch.cuda.stream(self._stream):
                    self.next_prepared = self.prepare_batch_async(self.next_batch)
            else:
                self.next_prepared = self.prepare_batch_async(self.next_batch)
        else:
            self.next_prepared = None

    def _to_pyramid(
        self, component: tuple[torch.Tensor, ...]
    ) -> dict[int, torch.Tensor]:
        """Move a tuple of per-scale tensors to the target device and return a dict."""
        if not component:
            return {}
        return {
            idx: device_manager.to_device(
                component[idx],
                channels_last=True,
                non_blocking=True,
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
        # Use as_tensor to handle both tensors and sequences while enforcing
        # the correct device and dtype (int64) for indices.
        return torch.as_tensor(
            values, device=device_manager.device, dtype=torch.long
        ).detach()

    def prepare_batch_async(self, batch: RawBatch) -> PyramidsBatch:
        """Perform batch preparation logic asynchronously.

        Moves facies, wells, and seismic data to the target device and
        computes masks from well data if not already provided in the batch.

        Parameters
        ----------
        batch : RawBatch
            The raw batch from the DataLoader. It handles multiple formats:
            - `Batch` NamedTuple
            - `(indices, Batch)` pair

        Returns
        -------
        PyramidsBatch
            A tuple of ``(indices, facies, wells, masks, seismic)`` where
            the per-scale components are dictionaries keyed by scale index.
        """
        # Distinguish between `Batch` (NamedTuple) and `tuple[Tensor, Batch]`
        if isinstance(batch, Batch):
            self.last_seen_indices = None
            facies_batch = batch
        else:
            # It's a tuple[Tensor, Batch]
            self.last_seen_indices = self._coerce_indices(batch[0])
            self.seen_indices.append(self.last_seen_indices.detach().clone())
            facies_batch = batch[1]

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

        # Extract relative index for training conditioning (the first column of our [rel, orig] pairs)
        training_indices = self.last_seen_indices
        if training_indices is not None and training_indices.dim() == 2:
            training_indices = training_indices[:, 0]

        return (
            (
                training_indices
                if training_indices is not None
                else torch.arange(
                    facies[0].shape[0] if facies else 0,
                    device=device_manager.device,
                    dtype=torch.long,
                )
            ),
            facies_pyramid,
            wells_pyramid,
            masks_pyramid,
            seismic_pyramid,
        )

    def _fetch_next(self) -> PyramidsBatch | None:
        """Synchronize with the prefetch stream and return the prepared batch.

        This method waits for the asynchronous preparation to complete on the
        target device before returning the `PyramidsBatch` and triggering the
        next `preload` call.
        """
        if self._stream:
            # Wait for the async preparation on the prefetch stream to complete
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
            f"DataPrefetcher(device='{device_manager.device.type}', "
            f"scale_indices={self.scale_indices})"
        )


def gather_seen_indices(prefetcher: DataPrefetcher) -> List[Tuple[int, ...]]:
    """Return dataset indices seen by all ranks during the current iteration.

    In DDP, each rank processes different data. We gather all indices
    to all ranks so that visualization can access any sample from the
    full dataset that was seen in this epoch.

    Returns
    -------
    List[Union[int, Tuple[int, ...]]]
        A flat list of sample indices gathered across all ranks.
    """
    local_indices: List[torch.Tensor] = prefetcher.seen_indices
    if not local_indices:
        return []
        
    # Clear the list so we don't accumulate forever across epochs!
    prefetcher.seen_indices = []

    # Concatenate all local batches into one [N, 2] or [N] tensor/array
    local_data = device_manager.to_numpy(torch.cat(local_indices))

    if not dist.is_available() or not dist.is_initialized():
        if local_data.ndim == 2:
            return [tuple(row) for row in local_data]
        return local_data.tolist()

    # In DDP, use all_gather_object to handle potentially varying number of samples per rank.
    # This is much safer than fixed-size tensor gathering.
    world_size = dist.get_world_size()
    gathered_objects: List[np.ndarray] = [np.array([]) for _ in range(world_size)]
    dist.all_gather_object(gathered_objects, local_data)  # type: ignore

    # Combine results from all ranks
    combined = np.concatenate(gathered_objects, axis=0)
    if combined.ndim == 2:
        return [tuple(row) for row in combined]
    return combined.tolist()
