"""Well-based interpolation utilities.

This module implements :class:`WellInterpolator`, which generates
multiscale images by extracting vertical well traces from numeric arrays.
The output is suitable for tasks that require preserving vertical trace
consistency (e.g. well logs).
"""

# pyright: reportIncompatibleMethodOverride=false

from __future__ import annotations

import logging

import numpy as np
import torch
from device import device_manager

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
        """Create multiscale well representations from a raw NumPy array.

        Parameters
        ----------
        data_array : np.ndarray
            Input array containing the categorical well trace data. May be
            2-D (H, W) or 3-D (H, W, C), in which case the first channel is
            used.
        resolutions : tuple[tuple[int, ...], ...]
            Sequence of target resolution tuples. Only the spatial H/W
            dimensions are used by this interpolator.

        Returns
        -------
        list[torch.Tensor]
            A list of one-hot encoded PyTorch tensors, one per requested
            resolution, representing the well trace pyramid.
        """
        import torch.nn.functional as F

        if data_array.ndim == 3:
            data_array = data_array[:, :, 0]
        data_array = data_array.astype(np.float32, copy=False)

        _, width = data_array.shape
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

                # Extract the vertical trace (H,)
                trace = data_array[:, well_col]

                # --- Majority Vote (Mode Pooling) Downsampling ---
                # Step 1: Convert to PyTorch long tensor
                trace_tensor: torch.Tensor = torch.from_numpy(  # type: ignore
                    trace.astype(np.int64)
                ).long()

                # Step 2: One-hot encode the 1D trace
                trace_one_hot = F.one_hot(
                    trace_tensor, num_classes=actual_num_classes
                ).float()

                # Step 3: Permute for 1D pooling: (Length, Channels) -> (1, Channels, Length)
                trace_one_hot = trace_one_hot.permute(1, 0).unsqueeze(0)

                # Step 4: Adaptive Average Pool to get the volumetric proportion of each facies
                pooled_proportions = F.adaptive_avg_pool1d(
                    trace_one_hot, output_size=new_h
                )

                # Step 5: Argmax to select the dominant facies (Majority Vote)
                pooled_labels = torch.argmax(pooled_proportions.squeeze(0), dim=0)

                # Insert the pooled trace into the output array
                output[:, scaled_col] = device_manager.to_cpu(
                    pooled_labels, non_blocking=True
                ).numpy()

            # Return the label indices directly (LongTensor)
            # Downstream consumers (e.g. datasets/utils.py) will map these to RGB.
            indices_tensor = torch.as_tensor(output, dtype=torch.long)
            pyramid.append(indices_tensor)

        return pyramid
