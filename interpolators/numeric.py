"""Unified numeric interpolator for categorical and continuous data.

This module provides the :class:`NumericInterpolator`, which handles all
dataset components. It selects the appropriate mathematical
strategy based on the configuration:

* **Categorical (Facies)**: Uses nearest-neighbour upscaling and mode-filter
  (majority voting) downscaling to preserve discrete class labels.
* **Continuous (Rock Physics)**: Uses Lanczos upscaling and Backus averaging
  (harmonic mean) downscaling. This follows standard geophysical practice
  for effective medium properties like Acoustic Impedance.

All outputs are returned as multi-scale PyTorch pyramids.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.ndimage import zoom  # type: ignore[import-untyped]

from enums import InterpolationStrategy
from interpolators.base import BaseInterpolator
from interpolators.config import InterpolatorConfig

logger = logging.getLogger(__name__)


def _mode_filter_reduction(block: NDArray[np.float32]) -> float:
    """Compute the majority-vote (mode) of a block."""
    channel_block = block.ravel()
    int_block = np.rint(channel_block).astype(np.int32)
    counts = np.bincount(int_block)
    return float(np.argmax(counts))


def _backus_average_reduction(block: NDArray[np.float32]) -> float:
    """Compute the Backus average (harmonic mean) of a block."""
    vals = block.ravel()
    if np.any(vals <= 0):
        return float(np.mean(vals))
    return float(len(vals) / np.sum(1.0 / vals))


class NumericInterpolator(BaseInterpolator):
    """Unified multi-scale interpolator for numeric array data."""

    def __init__(self, config: InterpolatorConfig) -> None:
        super().__init__(config)

    def _nearest_resize(
        self,
        data: NDArray[np.float32],
        target_h: int,
        target_w: int,
        use_mode_filter: bool = True,
    ) -> NDArray[np.float32]:
        """Resize using nearest-neighbour interpolation or mode filter."""
        src_h, src_w = data.shape[:2]

        if (
            use_mode_filter
            and target_h <= src_h
            and target_w <= src_w
            and (target_h < src_h or target_w < src_w)
        ):
            return self._block_reduce(data, target_h, target_w, _mode_filter_reduction)

        has_channels = data.ndim == 3
        zoom_h = target_h / src_h
        zoom_w = target_w / src_w

        factors = (zoom_h, zoom_w, 1.0) if has_channels else (zoom_h, zoom_w)
        result: NDArray[np.float32] = np.asarray(zoom(data, factors, order=0), dtype=np.float32)
        return result

    def interpolate_array(
        self,
        data: NDArray[np.float32],
        resolutions: tuple[tuple[int, ...], ...],
    ) -> list[torch.Tensor]:
        """Create a multi-scale pyramid from a raw NumPy array."""
        if data.ndim == 2:
            data = data[:, :, None]
        data = np.asarray(data, dtype=np.float32)

        if self.config.strategy == InterpolationStrategy.CATEGORICAL:
            return self._interpolate_categorical(data, resolutions)
        else:
            return self._interpolate_continuous(data, resolutions)

    def _interpolate_categorical(
        self, data: NDArray[np.float32], resolutions: tuple[tuple[int, ...], ...]
    ) -> list[torch.Tensor]:
        """Interpolate discrete labels and one-hot encode."""
        global_max_class = int(data.max())
        actual_num_classes = max(global_max_class + 1, self.config.num_classes)
        pyramid: list[torch.Tensor] = []

        for resolution in resolutions:
            if self.config.channels_last:
                _, target_h, target_w, _ = resolution
            else:
                _, _, target_h, target_w = resolution

            resized = self._nearest_resize(
                data, target_h, target_w, use_mode_filter=self.config.use_mode_filter
            )
            resized_indices = resized[:, :, 0] if resized.ndim == 3 else resized

            indices_tensor = torch.as_tensor(resized_indices, dtype=torch.long)
            one_hot = self._one_hot_encode(indices_tensor, actual_num_classes)
            pyramid.append(one_hot)

        return pyramid

    def _interpolate_continuous(
        self, data: NDArray[np.float32], resolutions: tuple[tuple[int, ...], ...]
    ) -> list[torch.Tensor]:
        """Interpolate continuous properties with Backus and normalization."""
        data = self._normalize_data(data)
        src_h, src_w = data.shape[:2]
        pyramid: list[torch.Tensor] = []

        for resolution in resolutions:
            if self.config.channels_last:
                _, target_h, target_w, _ = resolution
            else:
                _, _, target_h, target_w = resolution

            if (
                target_h <= src_h
                and target_w <= src_w
                and (target_h < src_h or target_w < src_w)
            ):
                resized = self._block_reduce(
                    data, target_h, target_w, _backus_average_reduction
                )
            else:
                resized = self._lanczos_resize(data, target_h, target_w)

            clipped = np.clip(resized, 0.0, 1.0)
            resized = np.asarray(clipped, dtype=np.float32)
            if resized.ndim == 2:
                resized = resized[:, :, None]
            pyramid.append(torch.as_tensor(resized, dtype=torch.float32))

        return pyramid
