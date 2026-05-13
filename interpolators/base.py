"""Base classes and shared utilities for interpolator.

This module defines :class:`BaseInterpolator`, a light-weight abstract
base class that provides common helpers and a consistent API for
interpolator implementations. Subclasses should implement the
:meth:`interpolate_array` method for numeric data.
"""

from __future__ import annotations

import logging
from typing import Callable, cast

import numpy as np
import torch
from numpy.typing import NDArray

from interpolators.config import InterpolatorConfig

logger = logging.getLogger(__name__)

# Constants for Lanczos resampling
LANCZOS_RADIUS = 3


class BaseInterpolator:
    """Base class providing common functionality for all interpolator."""

    config: InterpolatorConfig

    def __init__(self, config: InterpolatorConfig) -> None:
        """Store the provided configuration on the instance."""
        self.config = config

    def _normalize_data(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        """Normalize numeric array to ``config.normalization_range``."""
        data_min = (
            self.config.data_min
            if self.config.data_min is not None
            else float(data.min())
        )
        data_max = (
            self.config.data_max
            if self.config.data_max is not None
            else float(data.max())
        )
        norm_lo = float(min(self.config.normalization_range))
        norm_hi = float(max(self.config.normalization_range))
        if data_max > data_min:
            unit = (data - data_min) / (data_max - data_min)
            mapped = norm_lo + unit * (norm_hi - norm_lo)
            return np.clip(mapped, norm_lo, norm_hi).astype(np.float32, copy=False)
        return np.full_like(data, norm_lo, dtype=np.float32)

    @staticmethod
    def _block_reduce(
        data: NDArray[np.float32],
        target_h: int,
        target_w: int,
        reduction_fn: Callable[[NDArray[np.float32]], float],
    ) -> NDArray[np.float32]:
        """Apply a reduction function to spatial blocks of an array.

        This helper centralizes the spatial partitioning logic used for
        categorical (mode-filter) and continuous (Backus averaging) downsampling.

        Parameters
        ----------
        data : NDArray[np.float32]
            Input array of shape (H, W) or (H, W, C).
        target_h, target_w : int
            Target output dimensions.
        reduction_fn : callable
            Function that takes a 2D channel block and returns a scalar.

        Returns
        -------
        NDArray[np.float32]
            Reduced array of shape (target_h, target_w) or (target_h, target_w, C).
        """
        src_h, src_w = data.shape[:2]
        has_channels = data.ndim == 3
        num_channels = data.shape[2] if has_channels else 1

        if not has_channels:
            data = data[:, :, None]

        result = np.empty((target_h, target_w, num_channels), dtype=np.float32)
        row_edges: NDArray[np.int64] = np.linspace(
            0, src_h, target_h + 1, dtype=np.int64
        )
        col_edges: NDArray[np.int64] = np.linspace(
            0, src_w, target_w + 1, dtype=np.int64
        )

        for i in range(target_h):
            for j in range(target_w):
                block = data[
                    row_edges[i] : row_edges[i + 1],
                    col_edges[j] : col_edges[j + 1],
                    :,
                ]
                for c in range(num_channels):
                    result[i, j, c] = reduction_fn(block[:, :, c])

        if not has_channels or num_channels == 1:
            return cast(NDArray[np.float32], result)[:, :, 0]
        return cast(NDArray[np.float32], result)

    def _one_hot_encode(
        self, indices_tensor: torch.Tensor, actual_num_classes: int
    ) -> torch.Tensor:
        """Apply one-hot encoding and handle background channel dropping."""
        one_hot = torch.nn.functional.one_hot(
            indices_tensor, num_classes=actual_num_classes
        ).float()

        if actual_num_classes > self.config.num_classes:
            one_hot = one_hot[..., -self.config.num_classes :]

        return one_hot

    # --- Shared Mathematical Kernels and Resampling Logic ---

    @staticmethod
    def _gaussian_kernel(sigma: float) -> NDArray[np.float32]:
        """Build a normalized 1D Gaussian kernel."""
        sigma = max(float(sigma), 0.5)
        radius = max(1, int(np.ceil(3.0 * sigma)))
        offsets = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-(offsets**2) / (2.0 * sigma * sigma))
        kernel_sum = float(kernel.sum())
        if kernel_sum > 0.0:
            kernel /= kernel_sum
        return kernel.astype(np.float32, copy=False)

    @staticmethod
    def _convolve_axis(
        img: NDArray[np.float32], kernel: NDArray[np.float32], axis: int
    ) -> NDArray[np.float32]:
        """Apply a 1D convolution along one axis with reflect padding."""
        if kernel.size == 1:
            return img

        work = np.moveaxis(img, axis, -1)
        pad = kernel.size // 2
        padded = np.pad(work, [(0, 0)] * (work.ndim - 1) + [(pad, pad)], mode="reflect")
        windows = np.lib.stride_tricks.sliding_window_view(
            padded, window_shape=kernel.size, axis=-1
        )
        filtered = np.tensordot(windows, kernel, axes=([-1], [0]))
        return np.moveaxis(filtered, -1, axis)

    @staticmethod
    def _gaussian_blur(img: NDArray[np.float32], sigma: float) -> NDArray[np.float32]:
        """Blur a volume with a separable Gaussian kernel."""
        kernel = BaseInterpolator._gaussian_kernel(sigma)
        blurred = BaseInterpolator._convolve_axis(img, kernel, axis=0)
        blurred = BaseInterpolator._convolve_axis(blurred, kernel, axis=1)
        return blurred.astype(np.float32, copy=False)

    @staticmethod
    def _lanczos_kernel(
        x: NDArray[np.float32], radius: int = LANCZOS_RADIUS
    ) -> NDArray[np.float32]:
        """Evaluate the Lanczos kernel for a vector of offsets."""
        ax = np.abs(x)
        out = np.sinc(x) * np.sinc(x / radius)
        out = np.where(ax < radius, out, 0.0)
        return out.astype(np.float32, copy=False)

    @staticmethod
    def _resample_axis(
        img: NDArray[np.float32],
        target_len: int,
        axis: int,
        radius: int = LANCZOS_RADIUS,
    ) -> NDArray[np.float32]:
        """Resample a single axis with separable Lanczos interpolation."""
        if img.shape[axis] == target_len:
            return img

        work = np.moveaxis(img, axis, -1)
        src_len = work.shape[-1]
        scale = src_len / max(target_len, 1)
        positions = (np.arange(target_len, dtype=np.float32) + 0.5) * scale - 0.5
        base = np.floor(positions).astype(np.int64)
        window = np.arange(-radius + 1, radius + 1, dtype=np.int64)
        indices = base[:, None] + window[None, :]
        distances = positions[:, None] - indices.astype(np.float32)
        weights = BaseInterpolator._lanczos_kernel(distances, radius=radius)
        valid = (indices >= 0) & (indices < src_len)
        weights = np.where(valid, weights, 0.0)
        weight_sum = weights.sum(axis=1, keepdims=True)
        weights = np.divide(
            weights, weight_sum, out=np.zeros_like(weights), where=weight_sum != 0
        )

        indices = np.clip(indices, 0, src_len - 1)
        gathered = np.take(work, indices, axis=-1)
        resampled = np.sum(gathered * weights[None, ...], axis=-1)
        return np.moveaxis(resampled, -1, axis)

    @staticmethod
    def _lanczos_resize(
        img: NDArray[np.float32], target_h: int, target_w: int
    ) -> NDArray[np.float32]:
        """Resize a volume with separable Lanczos interpolation."""
        resized_w = BaseInterpolator._resample_axis(img, target_w, axis=1)
        resized_hw = BaseInterpolator._resample_axis(resized_w, target_h, axis=0)
        return resized_hw

    def interpolate_array(
        self,
        data: NDArray[np.float32],
        resolutions: tuple[tuple[int, ...], ...],
    ) -> list[torch.Tensor]:
        """Create a multiscale pyramid from a raw NumPy array.

        This abstract method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement interpolate_array")
