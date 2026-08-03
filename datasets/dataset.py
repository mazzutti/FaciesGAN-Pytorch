"""Dataset of multiscale pyramids for facies, wells and seismic data.

This module provides :class:`PyramidsDataset` which loads precomputed
multi-resolution tensors (pyramids) for facies images and optional
conditioning channels (wells, seismic). Each dataset item is a :class:`Batch`
named tuple containing per-scale tensors used by the training pipeline. When
``include_index=True``, items are returned as ``(idx, Batch)`` pairs for use
with data prefetching systems.
"""

from itertools import repeat
from typing import Optional

import torch
from torch.utils.data import Dataset

from config import DomainConfig
from options import TrainingOptions
from typedefs import Batch, RawBatch

from . import utils


class PyramidsDataset(Dataset[RawBatch]):
    """PyTorch dataset for multiscale facies with optional conditioning.

    Loads precomputed multi-resolution pyramids for facies images and optional
    conditioning channels (wells, seismic, masks, and rock-physics properties).
    Each dataset item is a :class:`Batch` object containing four tuple attributes,
    each with one tensor per scale:

    - ``facies``: Main target channels.
      - 4 channels: One-hot encoded facies [0:4].
      - 7 channels: [0:4] Facies + [4:7] Rock Physics (Ip, Is, Vp/Vs)
        when ``use_rock_physics`` is enabled.
    - ``wells``: Conditioning well locations.
      - 4 channels: One-hot encoded facies present at well locations.
    - ``seismic``: Conditioning seismic amplitude.
      - 1 channel: Normalized seismic volume slice.
    - ``masks``: Binary masks indicating well locations for sparse conditioning.

    Parameters
    ----------
    options : TrainingOptions
        Configuration providing ``input_path``, ``use_wells``, ``use_seismic``
        ``use_rock_physics``, and other scale-generation parameters.
    shuffle : bool, optional
        Shuffle samples after generation (default: ``False``).
    regenerate : bool, optional
        If ``True``, force recomputation of pyramid caches (default: ``False``).
    channels_last : bool, optional
        If ``True``, tensors use channels-last ordering (default: ``False``).
    include_index : bool, optional
        If ``True``, ``__getitem__`` returns ``(global_idx, Batch)`` tuples for
        use with data prefetchers (default: ``False``).
    """

    def __init__(
        self,
        options: TrainingOptions,
        shuffle: bool = False,
        regenerate: bool = False,
        channels_last: bool = False,
        include_index: bool = False,
    ) -> None:
        self.data_dir = options.input_path
        self.options = options
        self.channels_last = channels_last

        self.batches: list[Batch] = []
        self.scales = self.generate_scales(options, channels_last)

        # Internal cache for get_scale_data (mapping scale_idx -> (f, w, s) tensors)
        self._scale_data_cache: dict[int, tuple[torch.Tensor, ...]] = {}

        if regenerate:
            self.clean_cache()

        fp, wp, mp, sp = self.generate_pyramids()

        n_samples: int = fp[0].shape[0] if fp and fp[0].numel() > 0 else 0
        self.indices = torch.arange(n_samples, dtype=torch.long)

        if n_samples > 0:
            has_wells = bool(wp and wp[0].numel() > 0)
            has_masks = bool(mp and mp[0].numel() > 0)
            has_seismic = bool(sp and sp[0].numel() > 0)

            # Iterators for each component (repeat empty if missing)
            facies_iter = zip(*fp)
            wells_iter = zip(*wp) if has_wells else repeat(())
            masks_iter = zip(*mp) if has_masks else repeat(())
            seismic_iter = zip(*sp) if has_seismic else repeat(())

            for f, w, m, s in zip(facies_iter, wells_iter, masks_iter, seismic_iter):
                self.batches.append(
                    Batch(
                        facies=f,
                        wells=w if has_wells else (),
                        masks=m if has_masks else (),
                        seismic=s if has_seismic else (),
                    )
                )

        if shuffle:
            self.shuffle()

        # When True, __getitem__ will return (global_index, item)
        # so DataLoader collate will produce a 1-D Tensor of indices
        # as the first batch element. This is compatible with
        # `TorchDataPrefetcher` which detects indices in the first
        # position of the batch.
        self.include_index = include_index

    def generate_pyramids(
        self,
    ) -> tuple[
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
    ]:
        """Generate pyramid tensors for facies, wells, masks and seismic.

        This method orchestrates calls to the :mod:`datasets.utils` functions
        to load and normalize precomputed pyramid files from disk. When
        ``use_rock_physics`` is enabled, rock-physics properties (Ip, Is, Vp/Vs)
        are concatenated to the facies channels. When ``use_wells`` is enabled,
        masks are generated from well locations.

        Returns
        -------
        tuple
            A 4-tuple of (facies, wells, masks, seismic) pyramids. Each
            component is a tuple of tensors (one per scale).
        """
        normalization_range = (
            float(self.options.normalization_range[0]),
            float(self.options.normalization_range[1]),
        )

        facies_pyramids = utils.to_facies_pyramids(
            self.scales,
            data_dir=self.data_dir,
            channels_last=self.channels_last,
            num_classes=DomainConfig.NUM_FACIES,
            normalization_range=normalization_range,
        )

        if self.options.use_rock_physics:
            # We call the public wrappers directly to ensure Joblib populates
            # the specific cache folders (to_ip_pyramids, to_is_pyramids, etc.)
            rock_physics_attrs: list[tuple[torch.Tensor, ...]] = []
            if getattr(self.options, "use_ip", True):
                rock_physics_attrs.append(
                    utils.to_ip_pyramids(
                        self.scales,
                        self.data_dir,
                        self.channels_last,
                        normalization_range=normalization_range,
                    )
                )
            if getattr(self.options, "use_is", True):
                rock_physics_attrs.append(
                    utils.to_is_pyramids(
                        self.scales,
                        self.data_dir,
                        self.channels_last,
                        normalization_range=normalization_range,
                    )
                )
            if getattr(self.options, "use_vpvs", True):
                rock_physics_attrs.append(
                    utils.to_vp_vs_pyramids(
                        self.scales,
                        self.data_dir,
                        self.channels_last,
                        normalization_range=normalization_range,
                        use_robust_range=bool(
                            getattr(self.options, "vp_vs_robust_range", False)
                        ),
                        robust_percentiles=(
                            float(
                                getattr(
                                    self.options,
                                    "vp_vs_robust_percentiles",
                                    (1.0, 99.0),
                                )[0]
                            ),
                            float(
                                getattr(
                                    self.options,
                                    "vp_vs_robust_percentiles",
                                    (1.0, 99.0),
                                )[1]
                            ),
                        ),
                    )
                )

            combined: list[torch.Tensor] = []
            dim = 3 if self.channels_last else 1

            for i, f in enumerate(facies_pyramids):
                # Concatenate [Facies | Ip | Is | VP_VS]
                to_cat = [f]
                for p in rock_physics_attrs:
                    if i < len(p) and p[i].numel() > 0:
                        to_cat.append(p[i])
                    else:
                        # Append zero placeholder to maintain channel count
                        if self.channels_last:
                            placeholder = torch.zeros(
                                (f.shape[0], *f.shape[1:3], 1), device=f.device
                            )
                        else:
                            placeholder = torch.zeros(
                                (f.shape[0], 1, *f.shape[2:]), device=f.device
                            )
                        to_cat.append(placeholder)

                combined.append(torch.cat(to_cat, dim=dim))
            facies_pyramids = tuple(combined)

        # Wells
        if self.options.use_wells:
            wells_pyramids = utils.to_wells_pyramids(
                self.scales,
                data_dir=self.data_dir,
                channels_last=self.channels_last,
                num_classes=DomainConfig.NUM_FACIES,
                normalization_range=normalization_range,
            )
            masks_pyramids = utils.to_masks_pyramids(
                self.scales,
                data_dir=self.data_dir,
                channels_last=self.channels_last,
                num_classes=DomainConfig.NUM_FACIES,
                normalization_range=normalization_range,
            )

        else:
            wells_pyramids = tuple()
            masks_pyramids = tuple()

        # Seismic
        if (
            self.options.use_seismic
            or getattr(self.options, "seismic_loss_penalty", 0.0) > 0.0
            or getattr(self.options, "use_residual_coupling", False)
        ):
            seismic_pyramids = utils.to_seismic_pyramids(
                self.scales,
                data_dir=self.data_dir,
                channels_last=self.channels_last,
                normalization_range=normalization_range,
            )
        else:
            seismic_pyramids = tuple()

        return facies_pyramids, wells_pyramids, masks_pyramids, seismic_pyramids

    @staticmethod
    def generate_scales(
        options: TrainingOptions, channels_last: bool = False
    ) -> tuple[tuple[int, ...], ...]:
        """Return the scale descriptor sequence used by the dataset.

        Parameters
        ----------
        options : TrainingOptions
            Training configuration.
        channels_last : bool, optional
            If True, use NHWC layout (default: False).

        Returns
        -------
        tuple
            A tuple of shape tuples, one for each pyramid scale.
        """
        return utils.generate_scales(options, channels_last)

    def shuffle(self, seed: Optional[int] = None) -> None:
        """Shuffle ``self.batches`` in-place.

        Parameters
        ----------
        seed : int, optional
            If provided, uses this seed for reproducible shuffling.
        """
        if seed is not None:
            g = torch.Generator()
            g.manual_seed(seed)
            indexes = torch.randperm(len(self.batches), generator=g)
        else:
            indexes = torch.randperm(len(self.batches))

        self.batches = [self.batches[i] for i in indexes]
        # Keep indices as a tensor to support advanced indexing (list/tensor indexing)
        self.indices = self.indices[indexes]
        # Invalidate the scale data cache since order has changed
        self._scale_data_cache.clear()

    def select_equally_spaced(self, n: int) -> list[int]:
        """Select n equally spaced indices, subset/shuffle in-place, and return absolute indices.

        Parameters
        ----------
        n : int
            Number of training pyramids to select.

        Returns
        -------
        list[int]
            Sorted list of the selected absolute/original sample indices.
        """
        import numpy as np
        total = len(self)
        if n >= total:
            return sorted(self.indices.tolist())

        # Generate n equally spaced float indices between 0 and total - 1
        indices = np.round(np.linspace(0, total - 1, n)).astype(int)

        # Ensure they are unique and exactly n elements
        unique_indices = []
        seen = set()
        for idx in indices:
            val = int(idx)
            if val not in seen:
                unique_indices.append(val)
                seen.add(val)

        if len(unique_indices) < n:
            # Pad with remaining indices to reach n
            all_set = set(range(total))
            remaining = sorted(list(all_set - seen))
            while len(unique_indices) < n and remaining:
                val = remaining.pop(0)
                unique_indices.append(val)
                seen.add(val)

        selected = sorted(unique_indices)

        # Subset the dataset batches and indices
        selected_batches = [self.batches[i] for i in selected]
        selected_indices = self.indices[torch.as_tensor(selected, dtype=torch.long)]

        # Shuffle the selected subset via torch.randperm
        # Use a seed (falling back to 42 if manual_seed is None) to guarantee DDP ranks shuffle identically
        seed = self.options.manual_seed if getattr(self.options, "manual_seed", None) is not None else 42
        gen = torch.Generator().manual_seed(seed)
        g = torch.randperm(len(selected_batches), generator=gen)
        self.batches = [selected_batches[i] for i in g]
        self.indices = selected_indices[g]

        # Invalidate cache
        self._scale_data_cache.clear()

        return sorted(selected_indices.tolist())

    def clean_cache(self) -> None:
        """Clear the joblib pyramid cache and internal scale-data cache.

        Useful when regenerating pyramids or freeing memory after dataset use.
        """
        utils.memory.clear(warn=False)
        self._scale_data_cache.clear()

    def __repr__(self) -> str:
        return (
            f"PyramidsDataset(n_samples={len(self)}, "
            f"n_scales={len(self.scales)}, "
            f"wells={getattr(self.options, 'use_wells', False)}, "
            f"seismic={getattr(self.options, 'use_seismic', False)}, "
            f"rock_physics={getattr(self.options, 'use_rock_physics', False)})"
        )

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, idx: int) -> RawBatch:
        """Return a dataset item at the given index.

        Parameters
        ----------
        idx : int
            The sample index.

        Returns
        -------
        Batch or tuple[int, Batch]
            If ``include_index=False`` (default), returns the ``Batch`` directly.
            If ``include_index=True``, returns ``(idx, Batch)`` for use with
            data prefetchers that expect global sample indices.
        """
        item = self.batches[idx]
        if self.include_index:
            # Return tensor: [relative_idx, original_idx]
            # This allows tracking both prefetch position and absolute dataset ID
            return torch.tensor([idx, self.indices[idx]], dtype=torch.long), item
        return item

    def get_scale_data(self, scale: int | None = None) -> tuple[torch.Tensor, ...]:
        """Return per-scale tensors for the requested scale index.

        The results are cached internally to avoid redundant stacking operations
        across multiple calls for the same scale.

        Parameters
        ----------
        scale : int, optional
            0-based index of the scale to return (coarse → fine). If ``None``,
            returns the finest scale.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            ``(facies, wells, masks, seismic)`` tensors for the specified scale.
        """
        if not self.batches:
            return (torch.empty(0),) * 4

        if scale is None:
            scale = len(self.scales) - 1

        if scale in self._scale_data_cache:
            return self._scale_data_cache[scale]

        # Aggregate per-scale tensors across all batches
        def _stack_component(name: str) -> torch.Tensor:
            component_pyramids = [getattr(batch, name) for batch in self.batches]
            if not component_pyramids[0]:
                return torch.empty((0,), dtype=torch.float32)
            return torch.stack([p[scale] for p in component_pyramids], dim=0)

        result = (
            _stack_component("facies"),
            _stack_component("wells"),
            _stack_component("masks"),
            _stack_component("seismic"),
        )
        self._scale_data_cache[scale] = result
        return result
