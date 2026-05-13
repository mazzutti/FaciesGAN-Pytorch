"""Seismic interpolator module.

Provides a Lanczos-based seismic interpolator for array slices.
The finest scale preserves detail and lower resolutions are progressively
low-pass filtered to mimic noisier, bandwidth-limited seismic conditioning.
"""

# pyright: reportIncompatibleMethodOverride=false
from __future__ import annotations

import logging

import numpy as np
import torch
from numpy.typing import NDArray

from interpolators.base import BaseInterpolator
from interpolators.config import InterpolatorConfig

logger = logging.getLogger(__name__)


class SeismicInterpolator(BaseInterpolator):
    """Multiscale seismic interpolator with Lanczos upsampling and blur.

    The finest scale uses a Lanczos resize to preserve coherent reflectors.
    Coarser scales are additionally low-pass filtered before resizing to
    simulate reduced bandwidth and a dirtier seismic appearance.
    """

    def __init__(self, config: InterpolatorConfig) -> None:
        super().__init__(config)

    def _get_seismic_stats(self) -> tuple[float, float]:
        """Return the seismic min/max values used for normalization.

        Returns
        -------
        tuple[float, float]
            A tuple ``(min, max)`` used to normalize seismic amplitudes.
        """
        if self.config.data_min is not None and self.config.data_max is not None:
            return float(self.config.data_min), float(self.config.data_max)

        from constants import DEFAULT_DATA_DIR
        from datasets.utils import get_global_stats
        from enums import StatKey

        stats = get_global_stats(DEFAULT_DATA_DIR).get("SEISMIC")
        if stats is not None:
            return float(stats[StatKey.MIN]), float(stats[StatKey.MAX])
        return (
            float(min(self.config.normalization_range)),
            float(max(self.config.normalization_range)),
        )

    def interpolate_array(
        self,
        seismic_data: NDArray[np.float32],
        resolutions: tuple[tuple[int, ...], ...],
    ) -> list[torch.Tensor]:
        """Create a seismic pyramid from a raw NumPy array.

        Parameters
        ----------
        seismic_data : NDArray[np.float32]
            Input seismic slice. May be 2-D (H, W) or 3-D (H, W, C).
        resolutions : tuple[tuple[int, ...], ...]
            Sequence of resolution tuples used to generate the pyramid. Each
            resolution is a shape tuple; only the spatial H/W components are
            used by the interpolator.

        Returns
        -------
        list[torch.Tensor]
            A list of float32 PyTorch tensors (one per resolution) containing
            normalized seismic data in the configured ``normalization_range``.
        """
        if seismic_data.ndim == 2:
            seismic_data = seismic_data[:, :, None]
        seismic_data = seismic_data.astype(np.float32)
        seismic_norm = self._normalize_data(seismic_data)

        smooth_seismic: list[torch.Tensor] = []

        logger.info("Rendering with Lanczos seismic interpolation and blur...")

        for _, resolution in enumerate(resolutions):
            if self.config.channels_last:
                _, target_h, target_w, _ = resolution
            else:
                _, _, target_h, target_w = resolution

            resized = self._resize_with_band_limiting(
                seismic_norm,
                target_h,
                target_w,
            )
            smooth_seismic.append(torch.as_tensor(resized, dtype=torch.float32))

        return smooth_seismic

    def _resize_with_band_limiting(
        self, img: NDArray[np.float32], target_h: int, target_w: int
    ) -> NDArray[np.float32]:
        """Resize seismic data with Lanczos resampling and scale-dependent blur.

        Parameters
        ----------
        img : NDArray[np.float32]
            Input image/volume (H, W) or (H, W, C) already clipped to the
            configured normalization range.
        target_h : int
            Target height in pixels.
        target_w : int
            Target width in pixels.

        Returns
        -------
        NDArray[np.float32]
            Resampled image clipped to the normalization range and converted to
            ``np.float32``.
        """
        src_h, src_w = img.shape[:2]

        # The coarser the target scale, the more aggressive the low-pass.
        downscale = max(src_h / max(target_h, 1), src_w / max(target_w, 1))
        blur_radius = 0.0
        if downscale > 1.0:
            blur_radius = min(3.0, 0.45 * (downscale - 1.0) ** 1.2)

        norm_lo = float(min(self.config.normalization_range))
        norm_hi = float(max(self.config.normalization_range))

        work_img = img.clip(norm_lo, norm_hi)
        if blur_radius > 0.0:
            # Apply a gentle, scale-aware low-pass before shrinking
            work_img = self._gaussian_blur(work_img, blur_radius)

        resized = self._lanczos_resize(work_img, target_h, target_w)
        return resized.clip(norm_lo, norm_hi).astype(np.float32)
