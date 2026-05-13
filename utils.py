"""Utility helpers for facies visualization, masking and array/tensor helpers.

This module contains small, performance-minded helpers used by the training
and visualization code: cached color extraction, mask preprocessing, and
plotting helpers that convert tensors to PIL images. Docstrings in this
module follow NumPy-style conventions and describe input/output shapes where helpful.
"""

from __future__ import annotations

import hashlib
import os
import random
from collections import OrderedDict
from typing import Any, Self, TypeVar, cast

import numpy as np
import scipy.stats as st  # type: ignore
import torch
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Normalization Helpers
# ---------------------------------------------------------------------------


def norm(
    x: torch.Tensor,
    normalization_range: tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """Clamp tensor values to the configured normalization range.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    normalization_range : tuple[float, float], optional
        Inclusive output range ``(min, max)``. Defaults to ``(0.0, 1.0)``.

    Returns
    -------
    torch.Tensor
        Tensor with values clamped to ``normalization_range``.
    """
    norm_min = float(normalization_range[0])
    norm_max = float(normalization_range[1])
    lo = min(norm_min, norm_max)
    hi = max(norm_min, norm_max)
    return x.clamp(lo, hi)


def denorm(
    tensor: torch.Tensor | NDArray[np.float32],
    ceiling: bool = False,
    normalization_range: tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor | NDArray[np.float32]:
    """Clamp tensor/array values to the configured normalization range.

    Parameters
    ----------
    tensor : torch.Tensor | NDArray[np.float32]
        Input tensor or array.
    ceiling : bool, optional
        Whether to binarize values after clamping. When ``True``, values
        greater than the lower bound map to the upper bound; others map to
        the lower bound. Defaults to False.
    normalization_range : tuple[float, float], optional
        Inclusive output range ``(min, max)``. Defaults to ``(0.0, 1.0)``.

    Returns
    -------
    torch.Tensor | NDArray[np.float32]
        Tensor or array with values clamped to ``normalization_range``.
    """
    norm_min = float(normalization_range[0])
    norm_max = float(normalization_range[1])
    lo = min(norm_min, norm_max)
    hi = max(norm_min, norm_max)

    if isinstance(tensor, torch.Tensor):
        tensor = tensor.clamp(lo, hi)
    else:
        tensor = np.clip(tensor, lo, hi).astype(np.float32, copy=False)

    if ceiling:
        if isinstance(tensor, torch.Tensor):
            tensor = torch.where(
                tensor > lo,
                torch.as_tensor(hi, dtype=tensor.dtype, device=tensor.device),
                torch.as_tensor(lo, dtype=tensor.dtype, device=tensor.device),
            )
        else:
            tensor = np.where(tensor > lo, hi, lo).astype(np.float32, copy=False)

    return tensor


def get_padding_value(
    normalization_range: tuple[float, ...],
) -> float:
    """Compute the zero-padding fallback value (midpoint) from the normalization range.

    Parameters
    ----------
    normalization_range : tuple[float, float]
        Inclusive output range ``(min, max)``.

    Returns
    -------
    float
        The midpoint of the normalization range.
    """
    norm_min = float(normalization_range[0])
    norm_max = float(normalization_range[1])
    val = (norm_min + norm_max) / 2.0
    return val


from models.palette import PALETTE_RGB

# ---------------------------------------------------------------------------
# Framework Shims & Standard Imports
# ---------------------------------------------------------------------------


# Generic type variable used for helpers that return the same type as the input
T = TypeVar("T")

from config import DomainConfig
from config import DirectoryConfig
from enums import DeviceType, FaciesClass

# ---------------------------------------------------------------------------
# Color & Palette Helpers
# ---------------------------------------------------------------------------


class ExtractUniqueColors:
    """Callable class to extract unique colors from facies tensors with caching.

    Usage:
        extract_unique_colors = ExtractUniqueColors()
        palette = extract_unique_colors(facies_tensor, tolerance=0.01)

    The cache is keyed by `(tensor.shape, tolerance)`. Use
    `extract_unique_colors.clear_cache()` to invalidate.
    """

    # Singleton instance holder
    _instance: Self | None = None

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        max_cache_size: int = 128,
        device: torch.device = torch.device(DeviceType.CPU),
    ) -> None:
        # initialize cache once - safe even if __init__ called multiple times
        if getattr(self, "_cache", None) is None:
            # use OrderedDict for simple LRU eviction
            self._cache: OrderedDict[tuple[Any, ...], np.ndarray] = OrderedDict()
        self.max_cache_size = max_cache_size
        # Optional device for torch operations (e.g. torch.device('cuda'))
        self.device = device

    def __call__(
        self, facies_tensor: torch.Tensor, tolerance: float = 0.01
    ) -> np.ndarray:
        # Compute a small thumbnail fingerprint to include in the cache key.
        # This avoids returning stale outputs when the image content changes
        # but shapes remain the same.
        try:
            # If a torch tensor on CUDA, move a small downsampled tensor to CPU
            if facies_tensor.dim() >= 3:
                t0 = facies_tensor
                if t0.dim() == 4:
                    t0 = t0[0]
                # t0 shape: (C,H,W)
                # convert to HWC numpy quickly by permuting and taking a small thumbnail
                # Optionally move tensor to configured device for interpolation
                if (
                    getattr(self, "device", None) is not None
                    and t0.device != self.device
                ):
                    try:
                        t0 = t0.to(self.device)
                    except Exception:
                        pass

                small = torch.nn.functional.interpolate(
                    t0.unsqueeze(0), size=(16, 16), mode="bilinear", align_corners=False
                )[0]
                # Ensure thumbnail is on CPU before converting to numpy
                try:
                    small_cpu = small.detach().cpu()
                except Exception:
                    small_cpu = small.detach()
                thumb_np = np.clip(torch2np(small_cpu), 0.0, 1.0)
            else:
                # Fallback: convert via torch2np
                facies_np_full = torch2np(
                    facies_tensor[0] if facies_tensor.dim() == 4 else facies_tensor
                )
                img = Image.fromarray(
                    (np.clip(facies_np_full, 0.0, 1.0) * 255).astype("uint8")
                )
                thumb = img.resize((16, 16), Image.Resampling.BILINEAR)
                thumb_np = np.array(thumb).astype(np.float32) / 255.0

            try:
                thumb_hash = hashlib.sha1(thumb_np.tobytes()).hexdigest()[:12]
            except Exception:
                thumb_hash = "nohash"
        except Exception:
            thumb_hash = "nohash"

        key = (tuple(facies_tensor.shape), thumb_hash, float(tolerance))
        if key in self._cache:
            # move to end to mark recent use (LRU)
            val = self._cache.pop(key)
            self._cache[key] = val
            return val

        # Convert to numpy and reshape to (N, 3)
        if facies_tensor.dim() == 4:
            # Batch dimension present, use first sample
            facies_np = torch2np(facies_tensor[0])
        else:
            facies_np = torch2np(facies_tensor)

        # Reshape to (H*W, C) where C is number of channels
        num_ch = facies_np.shape[-1]
        pixels = facies_np.reshape(-1, num_ch)

        # Vectorized unique-color extraction using tolerance
        if float(tolerance) <= 0.0:
            unique_colors_array = np.unique(pixels, axis=0).astype(np.float32)
        else:
            # Scale and round to cluster colors within tolerance
            tol = float(tolerance)
            scaled = np.round(pixels / tol).astype(np.int64)
            uniq_scaled = np.unique(scaled, axis=0)
            unique_colors_array = (uniq_scaled.astype(np.float32)) * tol

        if unique_colors_array.size == 0:
            unique_colors_array = np.zeros((0, num_ch), dtype=np.float32)
        else:
            brightness = unique_colors_array.sum(axis=1)
            sorted_indices = np.argsort(brightness)
            unique_colors_array = unique_colors_array[sorted_indices]

        # Cache with simple LRU eviction
        self._cache[key] = unique_colors_array
        if len(self._cache) > self.max_cache_size:
            self._cache.popitem(last=False)
        return unique_colors_array

    def clear_cache(self) -> None:
        """Clear the internal cache."""
        self._cache.clear()

    @property
    def cache(self) -> dict[tuple[Any, ...], np.ndarray]:
        """Expose the internal cache dictionary for compatibility."""
        return self._cache


# ---------------------------------------------------------------------------
# Well Masking & Preprocessing
# ---------------------------------------------------------------------------


class PreprocessWellMask:
    """Callable class that preprocesses well masks and caches outputs.

    Usage:
        preprocess_well_mask = PreprocessWellMask()
        mask_2d, mask_bool, well_columns = preprocess_well_mask(mask, target_shape)

    The cache is keyed by (mask.shape, target_shape, sha1(mask.tobytes())).
    Call `preprocess_well_mask.clear_cache()` to invalidate.
    """

    def __init__(
        self,
        max_cache_size: int = 128,
        device: torch.device = torch.device(DeviceType.CPU),
    ) -> None:
        # OrderedDict for LRU eviction
        self._cache: OrderedDict[
            tuple[tuple[int, ...], tuple[int, int], str],
            tuple[np.ndarray, np.ndarray, np.ndarray],
        ] = OrderedDict()
        self.max_cache_size = max_cache_size
        self.device = device

    # Singleton instance holder
    _instance: Any | None = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "PreprocessWellMask":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __call__(
        self, mask: np.ndarray, target_shape: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mask_2d = np.squeeze(mask)

        # Compute a stable content hash for the mask using the full mask bytes.
        # Using a thumbnail could cause collisions for small/resampled masks
        # (different scales appearing identical once downsampled). Hashing the
        # full mask bytes guarantees uniqueness across mask content and is
        # acceptably fast for mask arrays of typical size.
        try:
            content_hash = hashlib.sha1(mask_2d.tobytes()).hexdigest()[:12]
        except Exception:
            content_hash = "nohash"

        cache_key = (mask_2d.shape, target_shape, content_hash)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if mask_2d.size > 0:
            # Resize mask to match target dimensions if needed (use PIL for speed)
            if mask_2d.shape != target_shape:
                from PIL import Image

                mask_img = Image.fromarray((mask_2d > 0.5).astype("uint8") * 255)
                # PIL resize expects (width, height)
                resized = mask_img.resize(
                    (target_shape[1], target_shape[0]), Image.Resampling.NEAREST
                )
                mask_2d = np.array(resized) > 127

            # Find well column locations by summing mask vertically
            mask_bool = mask_2d > 0.5
            well_columns = np.where(np.sum(mask_bool, axis=0) > 0)[0]

            result = (mask_2d, mask_bool, well_columns)
            self._cache[cache_key] = result
            if len(self._cache) > self.max_cache_size:
                self._cache.popitem(last=False)
            return result

        result = (mask_2d, np.zeros_like(mask_2d, dtype=bool), np.array([], dtype=int))
        self._cache[cache_key] = result
        return result

    def clear_cache(self) -> None:
        self._cache.clear()


# ---------------------------------------------------------------------------
# Plotting & Visualization
# ---------------------------------------------------------------------------


def facies_to_rgb(
    facies: torch.Tensor | np.ndarray,
    palette: list[list[float]] | None = None,
) -> np.ndarray:
    """Convert one-hot encoded or class-indexed facies to an RGB image.

    This helper maps categorical facies data (either as one-hot channels
    or as a single channel of class indices) to a high-contrast RGB
    representation suitable for TensorBoard or Matplotlib.

    Parameters
    ----------
    facies : torch.Tensor | np.ndarray
        Facies data in (C, H, W), (H, W, C), or (H, W) format.
        Multi-channel floating inputs are treated as class probabilities and
        converted to RGB by palette blending.
    palette : list[list[float]], optional
        List of RGB triples in [0, 1] range. If None, a standard
        4-class palette (Yellow, Green, Gray, Blue) is used.

    Returns
    -------
    np.ndarray
        RGB image of shape (3, H, W) with values in [0.0, 1.0].
    """
    if palette is None:
        palette = PALETTE_RGB
    num_colors = len(palette)

    # Convert to numpy and handle dimensions
    if isinstance(facies, torch.Tensor):
        facies_np = facies.detach().cpu().numpy()
    else:
        facies_np = np.asarray(facies)

    if facies_np.ndim == 3:
        # Accept either CHW or HWC layouts.
        if facies_np.shape[0] > num_colors and facies_np.shape[-1] <= num_colors:
            facies_np = np.transpose(facies_np, (2, 0, 1))

        # Ignore non-facies channels (e.g. rock-physics) when present.
        if facies_np.shape[0] > num_colors:
            facies_np = facies_np[:num_colors, ...]

        # Probabilistic maps (float channels) are rendered by palette blending.
        # This avoids argmax tie artifacts that can appear as all-black images
        # when class probabilities are near-uniform at early training stages.
        if np.issubdtype(facies_np.dtype, np.floating):
            # If the input has exactly 3 channels and negative values, it's highly likely
            # to be a direct RGB image in [-1, 1] (e.g. Tanh output), not a probability map.
            if facies_np.shape[0] == 3:
                if np.min(facies_np) < -0.1:
                    rgb = (facies_np + 1.0) / 2.0
                    return np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)
                elif np.max(facies_np) <= 1.01:
                    # Likely already an RGB image in [0, 1] (e.g. from the dataset)
                    return np.clip(facies_np, 0.0, 1.0).astype(np.float32, copy=False)

            probs = np.clip(facies_np.astype(np.float32, copy=False), 0.0, 1.0)
            denom = np.maximum(probs.sum(axis=0, keepdims=True), DomainConfig.EPSILON)
            probs = probs / denom

            palette_np = np.asarray(palette, dtype=np.float32)
            # Ensure the palette size matches the number of channels to avoid tensordot crash
            if probs.shape[0] < palette_np.shape[0]:
                palette_np = palette_np[: probs.shape[0]]

            rgb = np.tensordot(
                np.transpose(probs, (1, 2, 0)), palette_np, axes=([2], [0])
            )
            rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)
            return np.transpose(rgb, (2, 0, 1))

        # One-hot or integer channels (C, H, W) -> (H, W) via argmax.
        indices = np.argmax(facies_np, axis=0)
    elif facies_np.ndim == 2:
        # Already index map (H, W)
        indices = facies_np.astype(np.int32)
    else:
        raise ValueError(
            f"Unsupported facies shape for RGB conversion: {facies_np.shape}"
        )
    indices = np.clip(indices, 0, max(num_colors - 1, 0))

    h, w = indices.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    for i, color in enumerate(palette):
        mask = indices == i
        rgb[mask] = color

    # Return as (3, H, W) for TensorBoard/PyTorch standard
    return np.transpose(rgb, (2, 0, 1))


def rgb_to_facies(
    rgb: torch.Tensor | np.ndarray,
    palette: list[list[float]] | None = None,
) -> np.ndarray:
    """Map a normalized RGB tensor back to discrete facies class indices.

    This helper assigns each pixel of a continuous RGB image (typically in
    the ``[-1, 1]`` range) to the closest color in the specified palette using
    Euclidean distance.

    Parameters
    ----------
    rgb : torch.Tensor | np.ndarray
        RGB data in ``(C, H, W)`` or ``(H, W, C)`` format. If ``C`` is the first
        dimension, it must be 3.
    palette : list[list[float]], optional
        List of RGB triples representing the class centers. If None, uses
        :data:`PALETTE_NORMALIZED` from :mod:`models.palette`.

    Returns
    -------
    np.ndarray
        2D integer array of shape ``(H, W)`` containing class indices.
    """
    if palette is None:
        from models.palette import PALETTE_NORMALIZED

        palette = PALETTE_NORMALIZED

    if isinstance(rgb, torch.Tensor):
        rgb_np = rgb.detach().cpu().numpy()
    else:
        rgb_np = np.asarray(rgb)

    if rgb_np.ndim != 3 or (rgb_np.shape[0] != 3 and rgb_np.shape[-1] != 3):
        raise ValueError(f"Expected 3D RGB array, got shape: {rgb_np.shape}")

    # Ensure channels last: (H, W, 3)
    if rgb_np.shape[0] == 3 and rgb_np.shape[-1] != 3:
        rgb_np = np.transpose(rgb_np, (1, 2, 0))

    pal_arr = np.array(palette, dtype=np.float32)  # (K, 3)

    # Compute squared Euclidean distance: ||x - c||^2
    # shape: (H, W, 1, 3) - (1, 1, K, 3) -> (H, W, K, 3) -> sum -> (H, W, K)
    diff = rgb_np[..., np.newaxis, :] - pal_arr[np.newaxis, np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)

    return np.argmin(dist_sq, axis=-1).astype(np.int32)


def set_seed(seed: int = DomainConfig.RANDOM_SEED) -> None:
    """Set seeds for torch, numpy and python.random at module level.

    Keeping a module-level `set_seed` makes it easy for other modules to
    call it without constructing a `NeuralSmoother` instance. The
    `NeuralSmoother.set_seed` method remains as a thin wrapper that
    delegates to this function to preserve backward compatibility.
    """
    seed_int = int(seed)
    torch.manual_seed(seed_int)  # pyright: ignore
    np.random.seed(seed_int)
    random.seed(seed_int)


# ---------------------------------------------------------------------------
# Device & Tensor Conversion
# ---------------------------------------------------------------------------


def resolve_device(gpu_device: int = 0) -> torch.device:
    """Return the preferred device: CUDA, or CPU.

    Exposed at module level so other modules can determine the best device
    without constructing a `NeuralSmoother` instance.
    """
    if torch.cuda.is_available():
        return torch.device(f"{DeviceType.CUDA}:{gpu_device}")
    return torch.device(DeviceType.CPU)


def apply_well_mask(
    gen_facies: np.ndarray,
    mask: np.ndarray,
    real_facies: np.ndarray,
    highlight_well: bool = True,
) -> np.ndarray:
    """Overlay well pixels and highlight the specific well column as white."""
    result = gen_facies.copy()
    mask = np.squeeze(mask)

    # Normalize mask to a boolean array for safe logical operations.
    # Accept numeric masks (e.g., uint8 or float) and threshold them,
    # otherwise cast to bool for object/other types.
    if mask.dtype == np.bool_:
        mask_bool = mask
    else:
        try:
            mask_bool = mask > 0.5
        except Exception:
            mask_bool = np.asarray(mask, dtype=bool)

    # Find the specific columns where the well is present using boolean mask
    well_col = np.where(mask_bool.any(axis=0))[0]
    if well_col.size == 0:
        return result

    # 1. Highlight columns in white (optional)
    if highlight_well and well_col.size < result.shape[1] * 0.3:
        # Only convert pixels that are effectively black to white so we
        # preserve existing non-black content in the well columns.
        if result.ndim == 3:
            slice_ = result[:, well_col, :]
            if np.issubdtype(result.dtype, np.integer):
                # uint8-style images: treat values <= 5 (~almost black) as black
                black_mask = np.all(slice_ <= 5, axis=2)
                slice_[black_mask] = 255
            else:
                # float images in [0,1]: treat values <= 0.05 as black
                black_mask = np.all(slice_ <= 0.05, axis=2)
                slice_[black_mask] = 1.0
            result[:, well_col, :] = slice_
        else:
            slice_ = result[:, well_col]
            if np.issubdtype(result.dtype, np.integer):
                black_mask = slice_ <= 5
                slice_[black_mask] = 255
            else:
                black_mask = slice_ <= 0.05
                slice_[black_mask] = 1.0
            result[:, well_col] = slice_

    # 2. Identify active, non-background well pixels
    # We exclude channel 0 (Floodplain) in One-Hot or dark pixels in RGB.
    if real_facies.ndim == 3 and real_facies.shape[2] > 3:  # One-Hot case
        is_not_bg = np.argmax(real_facies, axis=-1) != FaciesClass.FLOODPLAIN
    else:  # RGB/Grayscale case
        is_not_bg = np.max(np.abs(real_facies), axis=-1) > 0.3

    # Ensure overlay uses boolean mask
    overlay_mask = mask_bool & is_not_bg

    # 3. Apply the overlay with channel handling
    if (
        result.ndim == 3
        and real_facies.ndim == 3
        and result.shape[2] != real_facies.shape[2]
    ):
        if result.shape[2] == 3 and real_facies.shape[2] > 3:
            # One-Hot well data -> RGB for display
            well_rgb = np.transpose(
                facies_to_rgb(np.transpose(real_facies, (2, 0, 1))), (1, 2, 0)
            )
            result[overlay_mask] = well_rgb[overlay_mask]
        else:
            result[overlay_mask] = real_facies[overlay_mask, : result.shape[2]]
    else:
        result[overlay_mask] = real_facies[overlay_mask]

    return result


def create_dirs(path: str) -> None:
    """Create directory and all parent directories if they don't exist.

    Parameters
    ----------
    path : str
        Directory path to create.

    Raises
    ------
    RuntimeError
        If directory creation fails.
    """
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        msg = "Error creating directory:"
        raise RuntimeError(msg, path, e)


def np2torch(
    np_array: NDArray[np.float32],
    normalize: bool = False,
    normalization_range: tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor | NDArray[np.float32]:
    """Convert NumPy array to PyTorch tensor with optional normalization.

    Parameters
    ----------
    np_array : NDArray[np.float32]
        Input NumPy array to convert.
    normalize : bool, optional
        Whether to clamp the tensor to the configured normalization range.
        Defaults to False.
    normalization_range : tuple[float, float], optional
        Inclusive output range ``(min, max)`` used when ``normalize`` is True.
        Defaults to ``(0.0, 1.0)``.

    Returns
    -------
    torch.Tensor
        Converted tensor, clamped to ``normalization_range`` if
        ``normalize`` is True.

    """
    # Support input layouts: (B, T, H, W, C), (B, H, W, C), (H, W, C), or (H, W) grayscale.
    arr = np_array
    if arr.ndim == 5:
        # (B, T, H, W, C) -> (B, T, C, H, W)
        arr = np.transpose(arr, (0, 1, 4, 2, 3))
    elif arr.ndim == 4:
        # (B, H, W, C) -> (B, C, H, W)
        arr = np.transpose(arr, (0, 3, 1, 2))
    elif arr.ndim == 3:
        # (H, W, C) -> (C, H, W)
        arr = np.transpose(arr, (2, 0, 1))
    # For 2D arrays leave as-is (H, W) -> treat as single-channel
    if not hasattr(torch, "from_numpy"):
        return arr
    tensor = torch.from_numpy(arr).float()  # type: ignore
    if normalize:
        tensor = norm(tensor, normalization_range=normalization_range)
    return tensor


def torch2np(
    tensor: torch.Tensor,
    denormalize: bool = False,
    ceiling: bool = False,
    normalization_range: tuple[float, float] = (0.0, 1.0),
) -> NDArray[np.float32]:
    """
    Convert PyTorch tensor to NumPy array with optional denormalization.

    Supports 3D, 4D, and 5D tensors:
      - (C, H, W)   -> (H, W, C)
      - (B, C, H, W) -> (B, H, W, C)
      - (B, T, C, H, W) -> (B, T, H, W, C)
    Optionally applies compatibility denormalization to ``normalization_range``.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor of shape (C, H, W), (B, C, H, W), or (B, T, C, H, W).
    denormalize : bool, optional
        If True, apply compatibility denormalization to
        ``normalization_range``. Defaults to False.
    ceiling : bool, optional
        If True, set positive values to 1 during denormalization (currently
        not implemented in denorm). Defaults to False.
    normalization_range : tuple[float, float], optional
        Inclusive output range ``(min, max)``. Defaults to ``(0.0, 1.0)``.

    Returns
    -------
    NDArray[np.float32]
        NumPy array with shape (H, W, C), (B, H, W, C), or
        (B, T, H, W, C) and values clipped to ``normalization_range``.
    """
    norm_min = float(normalization_range[0])
    norm_max = float(normalization_range[1])
    lo = min(norm_min, norm_max)
    hi = max(norm_min, norm_max)

    if denormalize:
        tensor = cast(
            torch.Tensor,
            denorm(tensor, ceiling, normalization_range=normalization_range),
        )
    np_array = tensor.detach().cpu().numpy()
    # Support 5D, 4D, and 3D tensors
    if np_array.ndim == 5:
        # (B, T, C, H, W) -> (B, T, H, W, C)
        np_array = np.transpose(np_array, (0, 1, 3, 4, 2))
    elif np_array.ndim == 4:
        # (B, C, H, W) -> (B, H, W, C)
        np_array = np.transpose(np_array, (0, 2, 3, 1))
    elif np_array.ndim == 3:
        # (C, H, W) -> (H, W, C)
        np_array = np.transpose(np_array, (1, 2, 0))
    else:
        raise ValueError(f"Unsupported tensor ndim for torch2np: {np_array.ndim}")

    np_array = np.clip(np_array, lo, hi)
    return np_array.astype(np.float32)


def tensor2np(
    tensor: torch.Tensor,
    denormalize: bool = False,
    ceiling: bool = False,
    normalization_range: tuple[float, float] = (0.0, 1.0),
) -> NDArray[np.float32]:
    """Convert PyTorch tensor  to NumPy array with optional denormalization.

    Transforms tensor/array to NumPy array, optionally applying
    compatibility denormalization to ``normalization_range``.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor in (B, C, H, W) format.
    denormalize : bool, optional
        If True, apply compatibility denormalization to
        ``normalization_range``. Defaults to False.
    ceiling : bool, optional
        If True, set positive values to 1 during denormalization. Defaults to False.
    normalization_range : tuple[float, float], optional
        Inclusive output range ``(min, max)``. Defaults to ``(0.0, 1.0)``.

    Returns
    -------
    NDArray[np.float32]
        NumPy array with shape (B, H, W, C) and values clipped to
        ``normalization_range``.
    """
    norm_min = float(normalization_range[0])
    norm_max = float(normalization_range[1])
    lo = min(norm_min, norm_max)
    hi = max(norm_min, norm_max)

    # Check if it's a torch tensor using hasattr instead of isinstance
    # (isinstance can fail due to import/module reloading issues)
    if (
        hasattr(tensor, DeviceType.CPU)
        and hasattr(tensor, "detach")
        and hasattr(tensor, "numpy")
    ):
        return torch2np(
            tensor,
            denormalize,
            ceiling,
            normalization_range=normalization_range,
        )
    else:
        arr: np.ndarray = np.asarray(tensor).copy()
        if denormalize:
            arr = np.asarray(
                denorm(arr, ceiling=ceiling, normalization_range=normalization_range),
                dtype=np.float32,
            )
        # Auto-transpose to channels-last if it looks like (B, C, H, W)
        if arr.ndim == 4 and arr.shape[1] in [1, 3, 4, 6, 7]:
            arr = np.transpose(arr, (0, 2, 3, 1))
        elif arr.ndim == 3 and arr.shape[0] in [1, 3, 4, 6, 7]:
            arr = np.transpose(arr, (1, 2, 0))
        return np.clip(arr, lo, hi).astype(np.float32)
    return np.array(tensor).astype(np.float32)


def to_device(
    tensor: torch.Tensor,
    device: torch.device,
    *,
    channels_last: bool = True,
    non_blocking: bool = True,
) -> torch.Tensor:
    """Move ``tensor`` to ``device`` with appropriate layout and contiguity.

    - On CUDA: optionally convert to `channels_last` memory format and use
      `non_blocking` transfer where supported.
    - On CPU: returns the original tensor (if already on CPU).

    This helper centralizes device-layout handling so callers can avoid
    duplicating `.to(...).contiguous(...)` branches.
    """
    if device.type == DeviceType.CUDA:
        if channels_last:
            return tensor.to(device, non_blocking=non_blocking).contiguous(
                memory_format=torch.channels_last
            )
        return tensor.to(device, non_blocking=non_blocking).contiguous()
    return tensor


def draw_well_arrows(
    mask: np.ndarray | torch.Tensor | tuple[np.ndarray, np.ndarray, np.ndarray],
    draw: ImageDraw.ImageDraw,
    x_offset: int,
    y_offset: int,
    cell_size: int,
) -> None:
    """Draw red arrow markers above the plot area to indicate well positions.

    Parameters
    ----------
    mask : np.ndarray
        Well mask array indicating well column positions.
    draw : ImageDraw.ImageDraw
        PIL ImageDraw object for drawing on the main canvas.
    x_offset : int
        X position of the subplot on the main canvas.
    y_offset : int
        Y position where arrows should be drawn (above the subplot).
    cell_size : int
        Width of the subplot in pixels.
    """
    # Sum vertically to find columns that contain well pixels
    # If caller provided a preprocessed mask tuple (mask_2d, mask_bool, well_columns)
    # use the `well_columns` directly since they are already resized to the
    # `cell_size` resolution; this yields exact pixel alignment.
    if isinstance(mask, tuple):
        _, _, well_columns = mask
        if well_columns.size == 0:
            return
        # well_columns are indices in [0, cell_size-1]; center the arrow on the
        # middle of the column by adding 0.5 before integer conversion.
        center_x = float(np.mean(well_columns)) + 0.5
        arrow_x = x_offset + int(round(center_x))

        arrow_y = y_offset + 3
        arrow_size = 10
        draw.polygon(
            [
                (arrow_x, arrow_y + arrow_size),
                (arrow_x - arrow_size // 2, arrow_y),
                (arrow_x + arrow_size // 2, arrow_y),
            ],
            fill=(255, 0, 0),
        )
        return

    # Fallback: accept a raw mask array and map its column indices into the
    # cell pixel coordinates (for backward compatibility with callers that
    # still pass the raw mask).
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask_sum = np.sum(np.squeeze(mask), axis=0)
    well_cols = np.where(mask_sum > 0)[0]
    if well_cols.size == 0:
        return

    center_col = float(np.mean(well_cols))
    num_columns = len(mask_sum)
    center_x = (center_col + 0.5) * (cell_size / num_columns)
    arrow_x = x_offset + int(round(center_x))

    arrow_y = y_offset + 3
    arrow_size = 10
    draw.polygon(
        [
            (arrow_x, arrow_y + arrow_size),
            (arrow_x - arrow_size // 2, arrow_y),
            (arrow_x + arrow_size // 2, arrow_y),
        ],
        fill=(255, 0, 0),
    )


# ---------------------------------------------------------------------------
# Plotting & Visualization
# ---------------------------------------------------------------------------


def _apply_colormap_1ch(arr2d: np.ndarray, cmap_name: str = "viridis") -> np.ndarray:
    """Map a 2-D float array in [0, 1] to an (H, W, 3) uint8-style float RGB image.

    Uses matplotlib's registered colormaps so that single-channel continuous
    data (e.g. rock-physics properties) is rendered with a perceptually uniform
    colormap instead of the discrete facies palette.

    Parameters
    ----------
    arr2d : np.ndarray
        Shape (H, W), values in [0, 1].
    cmap_name : str
        Any matplotlib colormap name (default ``'viridis'``).

    Returns
    -------
    np.ndarray
        Shape (H, W, 3), dtype float32, values in [0, 1].
    """
    import matplotlib  # type: ignore

    matplotlib.use("Agg")  # non-interactive backend, safe for worker processes
    import matplotlib.pyplot as plt  # type: ignore

    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(arr2d.astype(np.float32))  # (H, W, 4)  RGBA in [0, 1]
    return rgba[..., :3].astype(np.float32)  # drop alpha


def plot_generated_outputs(
    fake_facies: (
        torch.Tensor | NDArray[np.float32] | list[torch.Tensor | NDArray[np.float32]]
    ),
    real_facies: torch.Tensor | NDArray[np.float32],
    stage: int,
    index: int,
    masks: torch.Tensor | NDArray[np.float32] | None = None,
    out_dir: str = DirectoryConfig.OUTPUTS,
    save: bool = False,
    cell_size: int = 256,
    device: torch.device = torch.device(DeviceType.CPU),
    batch_id: int | None = None,
    plot_title: str = "Facies",
    cmap: str = "viridis",
    normalization_range: tuple[float, float] = (0.0, 1.0),
) -> None:
    """Plot and optionally save generated facies using PIL (50-100x faster than matplotlib).

    Creates a grid visualization comparing generated facies realizations with
    real facies and well constraint overlays.

    Parameters
    ----------
    fake_facies : torch.Tensor | list[torch.Tensor]
        Sequence of generated facies tensors/arrays, or a single batched
        tensor. Accepts PyTorch tensors or NumPy arrays (host-side) for
        worker process compatibility.
    real_facies : torch.Tensor | np.ndarray
        Real facies batch as a tensor/array.
    masks : torch.Tensor | np.ndarray
        Optional well-location masks corresponding to `real_facies`.
    stage : int
        Current training stage/scale for labeling.
    index : int
        Iteration index for filename when saving.
    out_dir : str, optional
        Directory to save the plot. Defaults to DirectoryConfig.OUTPUTS.
    save : bool, optional
        Whether to save the plot to disk. Defaults to False.
    cell_size : int
        Size of each cell in pixels (default 256).
    device : torch.device
        Optional device to run torch-based helpers on (e.g. `torch.device('cuda')`).
    """
    if not save:
        return

    norm_min = float(normalization_range[0])
    norm_max = float(normalization_range[1])
    norm_lo = min(norm_min, norm_max)
    norm_hi = max(norm_min, norm_max)

    def _to_numpy_image(
        value: torch.Tensor | NDArray[np.float32],
        *,
        denormalize: bool,
        ceiling: bool = False,
    ) -> NDArray[np.float32]:
        """Convert tensor/array to float32 numpy in channels-last format."""
        if isinstance(value, torch.Tensor):
            return tensor2np(
                value,
                denormalize=denormalize,
                ceiling=ceiling,
                normalization_range=normalization_range,
            )

        arr = np.asarray(value, dtype=np.float32)
        if denormalize:
            arr = np.asarray(
                denorm(
                    arr,
                    ceiling=ceiling,
                    normalization_range=normalization_range,
                ),
                dtype=np.float32,
            )

        if arr.ndim == 4 and arr.shape[1] in [1, 3, 4, 6, 7]:
            arr = np.transpose(arr, (0, 2, 3, 1))
        elif arr.ndim == 3 and arr.shape[0] in [1, 3, 4, 6, 7]:
            arr = np.transpose(arr, (1, 2, 0))

        return np.clip(arr, norm_lo, norm_hi).astype(np.float32, copy=False)

    fake_facies_arr: list[list[np.ndarray]]
    num_real_facies: int
    num_generated_per_real: int

    if isinstance(fake_facies, list):
        np_real_for_count = _to_numpy_image(real_facies, denormalize=False)
        num_real_facies = (
            int(np_real_for_count.shape[0]) if np_real_for_count.ndim >= 4 else 1
        )
        fake_facies_arr = [[] for _ in range(num_real_facies)]

        for ff in fake_facies:
            arr = _to_numpy_image(ff, denormalize=False)
            if arr.ndim == 4:
                limit = min(num_real_facies, int(arr.shape[0]))
                for j in range(limit):
                    fake_facies_arr[j].append(np.asarray(arr[j], dtype=np.float32))
            else:
                arr_sample = np.asarray(arr, dtype=np.float32)
                for j in range(num_real_facies):
                    fake_facies_arr[j].append(arr_sample)

        if not fake_facies_arr:
            return

        num_real_facies = min(num_real_facies, len(fake_facies_arr))
        num_generated_per_real = max((len(row) for row in fake_facies_arr), default=0)
        if num_generated_per_real <= 0:
            return
    else:
        arr = _to_numpy_image(fake_facies, denormalize=False)
        if arr.ndim == 5:
            num_real_facies = int(arr.shape[0])
            num_generated_per_real = int(arr.shape[1])
            fake_facies_arr = [
                [
                    np.asarray(arr[i, j], dtype=np.float32)
                    for j in range(num_generated_per_real)
                ]
                for i in range(num_real_facies)
            ]
        elif arr.ndim == 4:
            num_real_facies = int(arr.shape[0])
            num_generated_per_real = 1
            fake_facies_arr = [
                [np.asarray(arr[i], dtype=np.float32)] for i in range(num_real_facies)
            ]
        elif arr.ndim == 3:
            num_real_facies = 1
            num_generated_per_real = 1
            fake_facies_arr = [[np.asarray(arr, dtype=np.float32)]]
        else:
            raise ValueError(f"Unsupported fake_facies ndim for plotting: {arr.ndim}")

    np_real_facies = _to_numpy_image(real_facies, denormalize=False)

    np_masks: NDArray[np.float32] | None
    if masks is not None:
        np_masks = _to_numpy_image(masks, denormalize=False)
    else:
        np_masks = None

    real_count = int(np_real_facies.shape[0]) if np_real_facies.ndim >= 4 else 1
    mask_count = (
        int(np_masks.shape[0])
        if np_masks is not None and np_masks.ndim >= 3
        else real_count
    )
    num_real_facies = min(num_real_facies, real_count, mask_count)
    if num_real_facies <= 0:
        return

    # Calculate grid dimensions with spacing and margins
    spacing = 20  # pixels between subplots
    title_height = 20  # height for subplot titles
    main_title_height = 30  # height for main title at top
    arrow_height = 15  # height for arrow markers above plots
    margin = 20  # margin around the entire figure
    cols = num_generated_per_real + 1
    rows = num_real_facies
    grid_width = cols * (cell_size + spacing) - spacing + 2 * margin
    grid_height = (
        main_title_height
        + arrow_height
        + rows * (cell_size + title_height + spacing)
        - spacing
        + 2 * margin
    )

    # Create output image (RGB)
    output_img = Image.new("RGB", (grid_width, grid_height), color=(255, 255, 255))
    main_draw: ImageDraw.ImageDraw = ImageDraw.Draw(output_img)

    # Try to use a better font, fall back to default if unavailable
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        main_title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except (IOError, OSError):
        title_font = ImageFont.load_default()
        main_title_font = ImageFont.load_default()

    # Draw main title
    main_title = f"Stage {stage} - Well Log, Real vs Generated {plot_title}"
    # Get text bounding box for centering
    bbox = main_draw.textbbox((0, 0), main_title, font=main_title_font)
    text_width = bbox[2] - bbox[0]
    main_title_x = (grid_width - text_width) // 2
    cast(Any, main_draw).text(
        (main_title_x, margin), main_title, fill=(0, 0, 0), font=main_title_font
    )

    # Determine the palette (pure_colors) used for quantization and plotting.
    # For Facies, we use a fixed high-contrast palette matching plot_pyramids.py.
    # For other data (e.g. Rock Physics), we optionally extract from real data.
    if "facies" in plot_title.lower():
        pure_colors = np.array(PALETTE_RGB, dtype=np.float32)
    else:
        torch_available = os.getenv("FG_NO_TORCH_IMPORT") != "1" and bool(
            getattr(torch, "from_numpy", None)
        )
        if torch_available:
            extractor = ExtractUniqueColors(device=device)
            pure_colors = extractor(np2torch(np_real_facies), 0.01)  # type: ignore
            pure_colors = np.asarray(pure_colors)
        else:
            arr = np.asarray(np_real_facies, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[None, ...]
            if arr.ndim != 4:
                pure_colors = np.array(PALETTE_RGB, dtype=np.float32)
            else:
                if arr.shape[-1] == 1:
                    arr = np.repeat(arr, 3, axis=-1)
                elif arr.shape[-1] > 3:
                    arr = arr[..., :3]
                # Downsample for speed
                h, w = arr.shape[1], arr.shape[2]
                step_h = max(1, h // 64)
                step_w = max(1, w // 64)
                sampled = arr[:, ::step_h, ::step_w, :]
                colors = sampled.reshape(-1, 3)
                tol = 0.01
                if tol > 0:
                    colors = np.round(colors / tol) * tol
                pure_colors = np.unique(colors, axis=0)

    is_facies_plot = "facies" in plot_title.lower()

    for i in range(num_real_facies):
        # Real facies (first column) - use RGB directly without colormap
        real_arr = np.squeeze(np_real_facies[i])

        # 1. Format conversion (e.g. to RGB)
        if real_arr.ndim == 3:
            if real_arr.shape[-1] > 3:
                # One-hot facies (possibly with RP) -> RGB
                real_rgb_temp = facies_to_rgb(np.transpose(real_arr, (2, 0, 1)))
                real_arr = np.transpose(real_rgb_temp, (1, 2, 0))
            elif real_arr.shape[-1] == 1:
                # Single-channel continuous property (e.g. rock physics)
                if is_facies_plot:
                    idx_map = real_arr.squeeze(-1)
                    real_rgb_temp = facies_to_rgb(idx_map)
                    real_arr = np.transpose(real_rgb_temp, (1, 2, 0))
                else:
                    real_arr = _apply_colormap_1ch(real_arr.squeeze(-1), cmap)
            # else: shape[-1] == 3 -> already RGB, fall through
        elif real_arr.ndim == 2:
            # Grayscale / index map
            if is_facies_plot:
                real_rgb_temp = facies_to_rgb(real_arr)
                real_arr = np.transpose(real_rgb_temp, (1, 2, 0))
            else:
                real_arr = _apply_colormap_1ch(real_arr, cmap)

        # Apply well mask at native resolution
        if np_masks is not None:
            mask_i = cast(np.ndarray, np_masks[i])
            real_arr = apply_well_mask(
                real_arr,
                mask_i,
                real_arr,
                highlight_well=True,
            )

        # Save native-resolution version for generator-overlay loop
        real_native = real_arr.copy()

        # 2. Resizing
        if real_arr.shape[:2] != (cell_size, cell_size):
            img = Image.fromarray((real_arr * 255).astype("uint8"), mode="RGB")
            resized = img.resize((cell_size, cell_size), Image.Resampling.NEAREST)
            real_arr = np.array(resized).astype(np.float32) / 255.0

        real_rgb = real_arr

        # Convert to uint8 and ensure 3 channels for RGB plotting
        if real_rgb.ndim == 3 and real_rgb.shape[2] > 3:
            real_rgb = real_rgb[..., :3]
        real_rgb = (cast(np.ndarray, real_rgb) * 255).astype(np.uint8)

        h, w = real_rgb.shape[:2]
        real_img = Image.fromarray(real_rgb, mode="RGB")
        if h != cell_size or w != cell_size:
            real_img = real_img.resize((cell_size, cell_size), Image.Resampling.BICUBIC)

        # Paste into grid with spacing, margin, and title
        x_offset = margin
        y_offset = (
            main_title_height
            + arrow_height
            + margin
            + i * (cell_size + title_height + spacing)
        )

        # Draw subplot title
        subplot_title = f"Real Slice {i + 1}"
        bbox = main_draw.textbbox((0, 0), subplot_title, font=title_font)
        text_width = bbox[2] - bbox[0]
        title_x = x_offset + (cell_size - text_width) // 2
        # Shift subplot title 5 pixels up for improved spacing
        cast(Any, main_draw).text(
            (title_x, y_offset - 5), subplot_title, fill=(0, 0, 0), font=title_font
        )

        # Draw arrow above the plot
        arrow_y = y_offset + title_height - arrow_height
        if np_masks is not None:
            draw_well_arrows(np_masks[i], main_draw, x_offset, arrow_y, cell_size)

        output_img.paste(real_img, (x_offset, y_offset + title_height))

        # Generated facies (remaining columns) - use RGB directly without colormap
        row = fake_facies_arr[i]
        for j in range(min(num_generated_per_real, len(row))):
            gen_arr = row[j]
            # Ensure gen_arr is a numpy float array in [0, 1]
            gen_arr_np = np.asarray(gen_arr, dtype=np.float32)
            if gen_arr_np.size == 0:
                gen_arr_np = gen_arr_np.reshape((cell_size, cell_size))

            # 1. Format conversion (e.g. to RGB)
            if gen_arr_np.ndim == 3:
                if gen_arr_np.shape[-1] > 3:
                    # One-hot facies -> RGB
                    gen_rgb_temp = facies_to_rgb(np.transpose(gen_arr_np, (2, 0, 1)))
                    gen_arr_np = np.transpose(gen_rgb_temp, (1, 2, 0))
                elif gen_arr_np.shape[-1] == 1:
                    # Single-channel continuous property (e.g. rock physics)
                    if is_facies_plot:
                        idx_map = gen_arr_np.squeeze(-1)
                        gen_rgb_temp = facies_to_rgb(idx_map)
                        gen_arr_np = np.transpose(gen_rgb_temp, (1, 2, 0))
                    else:
                        gen_arr_np = _apply_colormap_1ch(gen_arr_np.squeeze(-1), cmap)
                # else: shape[-1] == 3 -> already RGB, fall through
            elif gen_arr_np.ndim == 2:
                if is_facies_plot:
                    gen_rgb_temp = facies_to_rgb(gen_arr_np)
                    gen_arr_np = np.transpose(gen_rgb_temp, (1, 2, 0))
                else:
                    gen_arr_np = _apply_colormap_1ch(gen_arr_np, cmap)

            # Apply well mask at native resolution
            if np_masks is not None:
                mask_np = np.asarray(
                    np_masks[i].detach().cpu().numpy()
                    if isinstance(np_masks[i], torch.Tensor)
                    else np_masks[i]
                )
                gen_arr_np = apply_well_mask(
                    gen_arr_np,
                    mask_np,
                    real_native,
                    highlight_well=True,
                )

            # 2. Resizing
            if gen_arr_np.shape[:2] != (cell_size, cell_size):
                img = Image.fromarray((gen_arr_np * 255).astype("uint8"), mode="RGB")
                resized = img.resize((cell_size, cell_size), Image.Resampling.NEAREST)
                gen_arr_np = np.array(resized).astype(np.float32) / 255.0

            gen_rgb = gen_arr_np

            # Normalize generated image to [0,1]
            max_val = float(gen_rgb.max()) if gen_rgb.size > 0 else 1.0
            if max_val > 1.0:
                gen_rgb = gen_rgb / max_val

            # Guard against arrays that may be 2D (h, w) or 3D (h, w, c).
            # Use -1 to safely index the channel dimension for type checkers.
            if gen_rgb.ndim == 3 and gen_rgb.shape[-1] > 3:
                gen_rgb = gen_rgb[..., :3]
            gen_rgb = (gen_rgb * 255).astype(np.uint8)

            h, w = gen_rgb.shape[:2]
            gen_img = Image.fromarray(gen_rgb, mode="RGB")
            if h != cell_size or w != cell_size:
                gen_img = gen_img.resize(
                    (cell_size, cell_size), Image.Resampling.BICUBIC
                )

            x_offset = margin + (j + 1) * (cell_size + spacing)

            # Draw subplot title for generated facies
            gen_title = f"Gen. Slice {j + 1}"
            bbox = main_draw.textbbox((0, 0), gen_title, font=title_font)
            text_width = bbox[2] - bbox[0]
            title_x = x_offset + (cell_size - text_width) // 2
            # Shift generated subplot title 5 pixels up to match real facies titles
            cast(Any, main_draw).text(
                (title_x, y_offset - 5), gen_title, fill=(0, 0, 0), font=title_font
            )
            # Draw arrow above the plot
            arrow_y = y_offset + title_height - arrow_height
            if np_masks is not None:
                draw_well_arrows(np_masks[i], main_draw, x_offset, arrow_y, cell_size)

            output_img.paste(gen_img, (x_offset, y_offset + title_height))

    # Save directly
    if batch_id is not None:
        output_img.save(f"{out_dir}/gen_{stage}_{batch_id}_{index}.png", optimize=True)
    else:
        output_img.save(f"{out_dir}/gen_{stage}_{index}.png", optimize=True)


def load_facies_for_plot(
    height: int,
    width: int,
    channels: int,
    num_real: int,
) -> np.ndarray:
    """Load a facies batch for plotting when real data is missing.

    Uses the same cached pyramid helpers as training to fetch a scale that
    matches the requested spatial resolution. Returns values in [0, 1] and
    channels-last layout (N, H, W, C).
    """
    from datasets.utils import to_facies_pyramids

    scale_list = ((1, height, width, channels),)
    facies = to_facies_pyramids(scale_list, channels_last=True)
    if len(facies) == 0:
        return np.zeros((num_real, height, width, channels), dtype=np.float32)

    arr = facies[0].numpy()
    if arr.ndim == 3:
        arr = arr[None, ...]
    if num_real > 0 and arr.shape[0] > num_real:
        arr = arr[:num_real]

    arr = np.clip(arr, 0.0, 1.0)
    return arr.astype(np.float32)


def get_best_distribution(data: np.ndarray) -> tuple[str, float, tuple[Any, ...]]:
    """Identify the best-fitting distribution for an array of observations.

    The function fits several candidate continuous distributions to the
    flattened ``data`` array and selects the distribution with the highest
    Kolmogorov-Smirnov p-value.

    Parameters
    ----------
    data : np.ndarray
        Input data array of observations. The array will be flattened before
        fitting.

    Returns
    -------
    tuple[str, float, tuple[Any, ...]]
        A tuple containing ``(best_dist_name, best_p_value, best_parameters)``
        where ``best_dist_name`` is the SciPy distribution name, ``best_p_value``
        is the KS-test p-value for that fit, and ``best_parameters`` are the
        parameters returned by ``scipy.stats.<dist>.fit`` for the chosen
        distribution.
    """
    dist_names = ["norm", "exponweib", "pareto", "genextreme"]
    # Explicitly type these containers so static analysis can infer types for methods like .append
    dist_results: list[tuple[str, float]] = []
    params: dict[str, tuple[Any, ...]] = {}

    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data.flatten())
        params[dist_name] = param

        # Applying the Kolmogorov-Smirnov test
        # Use flattened data for the KS test to match the fit input
        result = st.kstest(data.flatten(), dist_name, args=param)  # type: ignore
        # pvalue may be a scalar or array-like; coerce to float
        pval = float(np.sum(result.pvalue))  # type: ignore
        dist_results.append((dist_name, pval))

    # Select the best fitted distribution
    best_dist, best_p = max(dist_results, key=lambda item: item[1])

    print(f"Best fitting distribution: {best_dist}")
    print(f"Best p value: {best_p}")
    print(f"Parameters for the best fit: {params[best_dist]}")

    return best_dist, best_p, params[best_dist]
