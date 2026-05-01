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
    """Multi-scale seismic interpolator with Lanczos upsampling and blur.

    The finest scale uses a Lanczos resize to preserve coherent reflectors.
    Coarser scales are additionally low-pass filtered before resizing to
    simulate reduced bandwidth and a dirtier seismic appearance.
    """

    def __init__(self, config: InterpolatorConfig) -> None:
        super().__init__(config)

    def _get_seismic_stats(self) -> tuple[float, float]:
        """Return the seismic min/max values used for normalization."""
        if self.config.data_min is not None and self.config.data_max is not None:
            return float(self.config.data_min), float(self.config.data_max)

        from datasets.data_files import DEFAULT_DATA_DIR
        from datasets.utils import get_global_stats

        stats = get_global_stats(DEFAULT_DATA_DIR).get("SEISMIC", {})
        return float(stats.get("min", 0.0)), float(stats.get("max", 1.0))

    def interpolate_array(
        self,
        seismic_data: NDArray[np.float32],
        resolutions: tuple[tuple[int, ...], ...],
    ) -> list[torch.Tensor]:
        """Create a seismic pyramid from a raw NumPy array."""
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
            smooth_seismic.append(torch.from_numpy(resized))  # type: ignore[arg-type]

        return smooth_seismic

    def _resize_with_band_limiting(
        self, img: NDArray[np.float32], target_h: int, target_w: int
    ) -> NDArray[np.float32]:
        """Resize seismic data with Lanczos resampling and scale-dependent blur."""
        src_h, src_w = img.shape[:2]

        # The coarser the target scale, the more aggressive the low-pass.
        downscale = max(src_h / max(target_h, 1), src_w / max(target_w, 1))
        blur_radius = 0.0
        if downscale > 1.0:
            blur_radius = min(3.0, 0.45 * (downscale - 1.0) ** 1.2)

        work_img = img.clip(0.0, 1.0)
        if blur_radius > 0.0:
            # Apply a gentle, scale-aware low-pass before shrinking
            work_img = self._gaussian_blur(work_img, blur_radius)

        resized = self._lanczos_resize(work_img, target_h, target_w)
        return resized.clip(0.0, 1.0).astype(np.float32)
