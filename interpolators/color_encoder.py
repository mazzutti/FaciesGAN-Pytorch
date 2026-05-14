"""Color encoder for converting RGB images to label indices and vice versa.

This module provides the ColorEncoder class for managing color palettes
and conversions between RGB and categorical label representations.
"""

import logging
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from config import DomainConfig
from device import device_manager

logger = logging.getLogger(__name__)


class ColorEncoder:
    """Manage colors palette and conversions between RGB and categorical labels.

    This class extracts unique colors from input images to build a palette,
    then provides methods to convert between RGB pixel values and categorical
    label indices. Supports CUDA devices with proper float32 handling.

    Parameters
    ----------
    img_array : NDArray[Any]
        Input RGB image array of shape (H, W, 3) with pixel values.

    Attributes
    ----------
    palette : NDArray[np.float32]
        Array of unique RGB colors found in the image, shape (N, 3).
    num_classes : int
        Number of unique facies classes (colors) detected.
    device : torch.device
        Device used for tensor operations.
    palette_tensor : torch.Tensor
        Palette as a float32 tensor on the specified device.
    """

    def __init__(self, img_array: NDArray[Any]) -> None:
        """Create a ColorEncoder from an example RGB image.

        Parameters
        ----------
        img_array : ndarray
            RGB image array shaped (H, W, 3) used to build the palette.
        """
        # Ensure we work with float32 NumPy arrays to avoid creating
        # torch.float64 tensors.
        pixels = img_array.reshape(-1, 3).astype(np.float32, copy=False)
        self.palette = np.unique(pixels, axis=0).astype(np.float32, copy=False)
        self.num_classes = len(self.palette)
        self.device = device_manager.device
        # Use torch.from_numpy to preserve dtype (float32).
        try:
            self.palette_tensor = torch.from_numpy(  # pyright: ignore
                self.palette,
            ).to(self.device)
        except (TypeError, ValueError, RuntimeError):
            # Fall back to generic constructor but force float32
            self.palette_tensor = torch.tensor(self.palette, dtype=torch.float32).to(
                self.device
            )
        logger.info(f"Detected {self.num_classes} unique facies classes.")

    def rgb_to_labels(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Convert an RGB tensor to label indices using the encoder palette.

        Accepts tensors with final dim == 3. Returns a tensor of indices with
        the same leading shape as input (i.e. replaces last dim with class idx).
        """

        # Ensure float32 on the same device as palette for cdist
        img = img_tensor.to(self.device).float()
        if img.shape[-1] != 3:
            raise ValueError("img_tensor must have last dimension size 3 (RGB)")

        orig_shape = img.shape[:-1]
        flat = img.reshape(-1, 3)

        # Compute distances to palette and pick nearest color index
        dists = torch.cdist(flat, self.palette_tensor)
        labels = torch.argmin(dists, dim=1)
        return labels.reshape(orig_shape).to(self.device)

    def labels_to_rgb(self, label_tensor: torch.Tensor) -> torch.Tensor:
        """Map label indices back to RGB values from the palette.

        Returns a float32 tensor on the encoder device with last dim == 3.
        """

        labels = label_tensor.long().to(self.device)
        orig_shape = labels.shape
        flat = labels.reshape(-1)
        colors = self.palette_tensor[flat]
        return colors.reshape(*orig_shape, 3)

    def compute_class_weights(self, labels: torch.Tensor) -> torch.Tensor:
        """Calculate inverse frequency class weights for imbalanced datasets.

        Weight formula: weight[i] = N / (num_classes * count[i])
        Normalized to mean 1.0. Returns weights on encoder device.
        """

        # Move to CPU for bincount and ensure integer dtype
        labels_cpu = device_manager.to_cpu(labels).long().reshape(-1)
        counts = torch.bincount(labels_cpu, minlength=self.num_classes).float()
        total = labels_cpu.numel()

        weights = total / (
            self.num_classes * (counts + max(DomainConfig.EPSILON, 1e-5))
        )
        weights = weights / weights.mean()

        # Log rounded weights for user information
        try:
            rounded = device_manager.to_numpy(weights).round(2).astype(float).tolist()
        except (RuntimeError, TypeError, ValueError):
            rounded = [float(x) for x in device_manager.to_cpu(weights).reshape(-1)]
        logger.info(f"Auto-calculated Class Weights: {rounded}")

        return weights.to(self.device).float()
