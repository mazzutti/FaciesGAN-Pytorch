"""Mask interpolation for sparse conditioning.

This module implements :class:`MaskInterpolator`, which generates
multiscale binary mask pyramids. Masks are used to indicate which
regions contain valid data for conditioning (e.g., well locations,
seismic coverage areas).
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray

from interpolators.config import InterpolatorConfig
from interpolators.well import WellInterpolator

logger = logging.getLogger(__name__)


class MaskInterpolator(WellInterpolator):
    """Binary mask interpolator for sparse conditioning regions.

    Creates multiscale binary mask pyramids by downsampling using a voting
    scheme: a location is included in the downsampled mask if the
    corresponding source block contains any non-zero values.
    """

    def __init__(self, config: InterpolatorConfig) -> None:
        super().__init__(config)

    def interpolate_array(
        self,
        data_array: np.ndarray,
        resolutions: tuple[tuple[int, ...], ...],
    ) -> list[torch.Tensor]:
        """Create multiscale binary mask representations from a raw NumPy array."""
        well_pyramid: list[torch.Tensor] = super().interpolate_array(
            data_array, resolutions
        )
        pyramid: list[torch.Tensor] = []

        for well_tensor in well_pyramid:
            # well_tensor is (H, W, C) from WellInterpolator._one_hot_encode
            # Filter classes 1: and exclude background class 0
            mask = well_tensor[..., 1:].any(dim=-1).int()  # (H, W)

            if self.config.channels_last:
                # NHWC: (H, W, 1)
                mask = mask.unsqueeze(-1)
            else:
                # NCHW: (1, H, W)
                mask = mask.unsqueeze(0)

            pyramid.append(mask.float())

        return pyramid

    @staticmethod
    def _downsample_mask(
        mask: NDArray[np.float32],
        target_h: int,
        target_w: int,
    ) -> NDArray[np.float32]:
        """Downsample binary mask using majority voting."""
        src_h, src_w = mask.shape
        result = np.zeros((target_h, target_w), dtype=np.float32)

        row_edges: NDArray[np.int32] = np.linspace(
            0, src_h, target_h + 1, dtype=np.int32
        )

        col_edges: NDArray[np.int32] = np.linspace(
            0, src_w, target_w + 1, dtype=np.int32
        )

        for i in range(target_h):
            for j in range(target_w):
                # Extract the block from the source
                block = mask[
                    row_edges[i] : row_edges[i + 1],
                    col_edges[j] : col_edges[j + 1],
                ]
                # Pixel is 1 if any values in the block are non-zero
                result[i, j] = 1.0 if block.any() else 0.0

        return cast(NDArray[np.float32], result)
