"""Well-based interpolation utilities.

This module implements :class:`WellInterpolator`, which generates
multi-scale images by extracting vertical well traces from numeric arrays.
The output is suitable for tasks that require preserving vertical trace
consistency (e.g. well logs).
"""

# pyright: reportIncompatibleMethodOverride=false

from __future__ import annotations

import logging

import numpy as np
import torch

from interpolators.base import BaseInterpolator
from interpolators.config import InterpolatorConfig

logger = logging.getLogger(__name__)


class WellInterpolator(BaseInterpolator):
    """Well-based interpolator that extracts and scales vertical well traces.

    Extracts vertical well trace(s) from input categorical arrays and
    positions them at proportionally scaled column locations across
    resolutions.
    """

    def __init__(self, config: InterpolatorConfig) -> None:
        super().__init__(config)

    def interpolate_array(
        self,
        data_array: np.ndarray,
        resolutions: tuple[tuple[int, ...], ...],
    ) -> list[torch.Tensor]:
        """Create multi-scale well representations from a raw NumPy array."""
        if data_array.ndim == 3:
            data_array = data_array[:, :, 0]
        data_array = data_array.astype(np.float32, copy=False)

        height, width = data_array.shape
        nonzero_cols = np.where(data_array.any(axis=0))[0]

        global_max_class = int(data_array.max())
        actual_num_classes = max(global_max_class + 1, self.config.num_classes)
        pyramid: list[torch.Tensor] = []

        for resolution in resolutions:
            if self.config.channels_last:
                _, new_h, new_w, _ = resolution
            else:
                _, _, new_h, new_w = resolution

            output: np.ndarray = np.zeros((new_h, new_w), dtype=np.float32)

            for well_col in nonzero_cols:
                # Scale the column position proportionally
                scaled_col = int(well_col * new_w / width)
                scaled_col = min(new_w - 1, max(0, scaled_col))

                # Downsample the vertical trace using nearest neighbor
                step = height / new_h
                indices = np.minimum((np.arange(new_h) * step).astype(int), height - 1)
                output[:, scaled_col] = data_array[indices, well_col]

            indices_tensor = torch.as_tensor(output, dtype=torch.long)
            one_hot = self._one_hot_encode(indices_tensor, actual_num_classes)
            pyramid.append(one_hot)

        return pyramid
