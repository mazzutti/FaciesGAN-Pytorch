"""Utility functions for multi-scale pyramid generation and data loading.

This module provides helpers to load numeric data from .npz archives, generate
progressive resolutions, and orchestrate the creation of multi-scale tensors
for facies, wells, and rock-physics properties.
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from joblib import Memory  # type: ignore

from datasets.data_files import DEFAULT_DATA_DIR, DataFiles
from interpolators.config import InterpolatorConfig
from interpolators.mask import MaskInterpolator
from interpolators.numeric import NumericInterpolator
from interpolators.seismic import SeismicInterpolator
from interpolators.well import WellInterpolator
from options import TrainingOptions
from torch_utils import norm

logger = logging.getLogger(__name__)

# Create a cache directory for joblib memory
memory = Memory("./.cache", verbose=0)  # type: ignore


def generate_scales(
    options: TrainingOptions, channels_last: bool = False
) -> tuple[tuple[int, ...], ...]:
    """Generate multi-scale pyramid resolutions for progressive training.

    The resolutions are determined by a geometric progression from ``min_size``
    up to the target ``max_size`` (or ``crop_size``), ensuring even spatial
    dimensions for compatibility with standard convolutional blocks.

    Parameters
    ----------
    options : TrainingOptions
        Configuration defining ``min_size``, ``max_size``, ``crop_size``,
        and ``stop_scale``.
    channels_last : bool, optional
        If True, returns shapes in ``(B, H, W, C)`` layout. Otherwise, uses
        ``(B, C, H, W)``. By default False.

    Returns
    -------
    tuple[tuple[int, ...], ...]
        A sequence of shape tuples, one for each pyramid scale from coarse
        to fine.
    """
    shapes: list[tuple[int, ...]] = []
    if options.stop_scale <= 0:
        scale_factor = 1.0
    else:
        scale_factor = math.pow(
            options.min_size / (min(options.max_size, options.crop_size)),
            1.0 / options.stop_scale,
        )

    for i in range(options.stop_scale + 1):
        scale = math.pow(scale_factor, options.stop_scale - i)
        out_shape = cast(
            list[int],
            np.uint(
                np.round(
                    np.array(
                        [
                            min(options.max_size, options.crop_size),
                            min(options.max_size, options.crop_size),
                        ]
                    )
                    * scale
                )
            ).tolist(),
        )

        # Force even dimensions for consistency with strided convs
        if out_shape[0] % 2 != 0:
            out_shape = [int(shape + 1) for shape in out_shape]

        num_ch = options.num_facies_classes

        if channels_last:
            shapes.append((options.batch_size, *out_shape, num_ch))
        else:
            shapes.append((options.batch_size, num_ch, *out_shape))

    return tuple(shapes)


def load_samples(
    component: DataFiles, data_dir: str | None = None
) -> dict[str, np.ndarray]:
    """Load all numeric samples for a component from a .npz archive.

    Returns
    -------
    dict[str, np.ndarray]
        A dictionary mapping sample names (stems) to NumPy arrays.
    """
    base_dir = Path(data_dir) if data_dir else Path(DEFAULT_DATA_DIR)
    npz_path = base_dir / f"{component.name.lower()}.npz"

    if not npz_path.exists():
        logger.error(f"Required data file missing: {npz_path}")
        return {}

    try:
        with np.load(npz_path) as data:
            return {k: data[k] for k in sorted(data.files)}
    except Exception as e:
        logger.error(f"Failed to load {npz_path}: {e}")
        return {}


def get_global_stats(data_dir: str | None = None) -> dict[str, dict[str, float]]:
    """Retrieve or compute global min/max statistics for all data components."""
    base_dir = Path(data_dir if data_dir else DEFAULT_DATA_DIR)
    stats_path = base_dir / "stats.json"

    if stats_path.exists():
        try:
            with open(stats_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load stats from %s: %e", stats_path, e)

    # Compute and save if not exists
    logger.info("Computing global dataset statistics from .npz archives...")
    stats: dict[str, dict[str, float]] = {}

    # We care about continuous numeric components
    components = [
        DataFiles.SEISMIC,
        DataFiles.VP,
        DataFiles.VS,
        DataFiles.RHO,
        DataFiles.Ip,
        DataFiles.Is,
        DataFiles.VP_VS,
    ]

    for comp in components:
        npz_path = base_dir / f"{comp.name.lower()}.npz"

        if npz_path.exists():
            try:
                with np.load(npz_path) as data_dict:
                    all_values: list[np.ndarray] = []
                    for key in data_dict.files:
                        all_values.append(data_dict[key].flatten())

                    if all_values:
                        combined = np.concatenate(all_values)
                        stats[comp.name] = {
                            "min": float(combined.min()),
                            "max": float(combined.max()),
                            "mean": float(combined.mean()),
                        }
            except Exception as e:
                logger.warning(f"Failed to compute stats for {comp.name}: {e}")
        else:
            # For derived components, compute if base components exist
            if comp in [DataFiles.Ip, DataFiles.Is, DataFiles.VP_VS]:
                stats[comp.name] = _compute_derived_global_stats(base_dir, comp)

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)

    return stats


def _compute_derived_global_stats(
    data_dir: Path, component: DataFiles
) -> dict[str, float]:
    """Helper to compute global stats for derived attributes (Ip, Is, Vp/Vs)."""
    facies_samples = load_samples(DataFiles.FACIES, str(data_dir))
    vp_samples = load_samples(DataFiles.VP, str(data_dir))
    vs_samples = load_samples(DataFiles.VS, str(data_dir))
    rho_samples = load_samples(DataFiles.RHO, str(data_dir))

    c_min, c_max = float("inf"), float("-inf")
    c_sum, c_count = 0.0, 0

    for name in facies_samples.keys():
        derived = None
        if component == DataFiles.Ip:
            if name in vp_samples and name in rho_samples:
                derived = (vp_samples[name] * 1000.0) * rho_samples[name]
        elif component == DataFiles.Is:
            if name in vs_samples and name in rho_samples:
                derived = (vs_samples[name] * 1000.0) * rho_samples[name]
        elif component == DataFiles.VP_VS:
            if name in vp_samples and name in vs_samples:
                vp_data = vp_samples[name]
                vs_data = vs_samples[name]
                derived = np.divide(
                    vp_data, vs_data, out=np.zeros_like(vp_data), where=vs_data != 0
                )

        if derived is not None:
            c_min = min(c_min, float(derived.min()))
            c_max = max(c_max, float(derived.max()))
            c_sum += float(derived.sum())
            c_count += derived.size

    if c_min == float("inf"):
        return {"min": 0.0, "max": 1.0, "mean": 0.5}
    return {
        "min": c_min,
        "max": c_max,
        "mean": c_sum / c_count if c_count > 0 else (c_min + c_max) / 2.0,
    }


def _empty_pyramid_tensor(
    scale: tuple[int, ...], channels_last: bool = False
) -> torch.Tensor:
    """Return an empty placeholder tensor for a missing pyramid scale."""
    if channels_last:
        _, height, width, channels = scale
        return torch.empty((0, height, width, channels), dtype=torch.float32)
    else:
        _, channels, height, width = scale
        return torch.empty((0, channels, height, width), dtype=torch.float32)


def _stack_and_format(
    pyramids_list: list[list[torch.Tensor]],
    channels_last: bool = False,
    is_mask: bool = False,
) -> list[torch.Tensor]:
    """Stack lists of per-sample pyramids into multi-sample batch tensors."""
    if not pyramids_list or not pyramids_list[0]:
        return []

    if channels_last or is_mask:
        return [torch.stack(pyramid, dim=0) for pyramid in pyramids_list]
    return [
        torch.stack(pyramid, dim=0).permute(0, 3, 1, 2) for pyramid in pyramids_list
    ]


def _build_pyramid_batch(
    data_file: DataFiles,
    scale_list: tuple[tuple[int, ...], ...],
    interpolator: Any,
    data_dir: str | None = None,
    channels_last: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Centralized helper for loading .npz archives, interpolation, and batching."""
    is_mask = data_file == DataFiles.MASKS
    comp_name = DataFiles.WELLS.name.lower() if is_mask else data_file.name.lower()

    base_dir = Path(data_dir) if data_dir else Path(DEFAULT_DATA_DIR)
    npz_path = base_dir / f"{comp_name}.npz"

    if not npz_path.exists():
        logger.warning(f"Data file {npz_path} not found. Returning empty pyramids.")
        return tuple(_empty_pyramid_tensor(s, channels_last) for s in scale_list)

    pyramids_list: list[list[torch.Tensor]] = [[] for _ in range(len(scale_list))]

    logger.info(f"Loading {data_file.name} from {npz_path}")
    with np.load(npz_path) as data_dict:
        for key in sorted(data_dict.files):
            data = data_dict[key]
            pyramid = interpolator.interpolate_array(data, scale_list)
            for i in range(len(scale_list)):
                if is_mask:
                    pyramids_list[i].append(pyramid[i])
                else:
                    pyramids_list[i].append(norm(pyramid[i]))

    return tuple(_stack_and_format(pyramids_list, channels_last, is_mask))


def _to_generic_pyramid(
    data_file: DataFiles,
    scale_list: tuple[tuple[int, ...], ...],
    data_dir: str | None = None,
    channels_last: bool = False,
    global_stats: dict[str, dict[str, float]] | None = None,
) -> tuple[torch.Tensor, ...]:
    """Generic helper to load and interpolate numeric data into a pyramid."""
    stats = global_stats.get(data_file.name, {}) if global_stats else {}
    interpolator = NumericInterpolator(
        InterpolatorConfig(
            channels_last=channels_last,
            strategy="continuous",
            data_min=stats.get("min"),
            data_max=stats.get("max"),
        )
    )
    return _build_pyramid_batch(
        data_file, scale_list, interpolator, data_dir, channels_last
    )


@memory.cache  # type: ignore
def to_facies_pyramids(
    scale_list: tuple[tuple[int, ...], ...],
    data_dir: str | None = None,
    channels_last: bool = False,
    num_classes: int = 4,
) -> tuple[torch.Tensor, ...]:
    """Generate multi-scale pyramid tensors for facies data from .npz."""
    interpolator = NumericInterpolator(
        InterpolatorConfig(
            channels_last=channels_last, num_classes=num_classes, strategy="categorical"
        )
    )
    return _build_pyramid_batch(
        DataFiles.FACIES, scale_list, interpolator, data_dir, channels_last
    )


@memory.cache  # type: ignore
def to_ip_pyramids(
    scale_list: tuple[tuple[int, ...], ...],
    data_dir: str | None = None,
    channels_last: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Generate multi-scale pyramid tensors from Ip volume."""
    global_stats = get_global_stats(data_dir)
    return _to_derived_pyramid(
        DataFiles.Ip, scale_list, data_dir, channels_last, global_stats
    )


@memory.cache  # type: ignore
def to_is_pyramids(
    scale_list: tuple[tuple[int, ...], ...],
    data_dir: str | None = None,
    channels_last: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Generate multi-scale pyramid tensors from Is volume."""
    global_stats = get_global_stats(data_dir)
    return _to_derived_pyramid(
        DataFiles.Is, scale_list, data_dir, channels_last, global_stats
    )


@memory.cache  # type: ignore
def to_vp_vs_pyramids(
    scale_list: tuple[tuple[int, ...], ...],
    data_dir: str | None = None,
    channels_last: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Generate multi-scale pyramid tensors from Vp/Vs volume."""
    global_stats = get_global_stats(data_dir)
    return _to_derived_pyramid(
        DataFiles.VP_VS, scale_list, data_dir, channels_last, global_stats
    )


def _to_derived_pyramid(
    component: DataFiles,
    scale_list: tuple[tuple[int, ...], ...],
    data_dir: str | None = None,
    channels_last: bool = False,
    global_stats: dict[str, dict[str, float]] | None = None,
) -> tuple[torch.Tensor, ...]:
    """Generic helper to derive Ip, Is or Vp/Vs from Vp, Vs and Rho if missing."""
    base_dir = Path(data_dir) if data_dir else Path(DEFAULT_DATA_DIR)
    npz_path = base_dir / f"{component.name.lower()}.npz"

    if npz_path.exists():
        return _to_generic_pyramid(
            component, scale_list, data_dir, channels_last, global_stats
        )

    # Derive from base components (VP, VS, RHO)
    facies_samples = load_samples(DataFiles.FACIES, data_dir)
    if not facies_samples:
        return tuple(_empty_pyramid_tensor(s, channels_last) for s in scale_list)

    vp_samples = load_samples(DataFiles.VP, data_dir)
    vs_samples = load_samples(DataFiles.VS, data_dir)
    rho_samples = load_samples(DataFiles.RHO, data_dir)

    stats = global_stats.get(component.name, {}) if global_stats else {}
    interpolator = NumericInterpolator(
        InterpolatorConfig(
            channels_last=channels_last,
            strategy="continuous",
            data_min=stats.get("min"),
            data_max=stats.get("max"),
        )
    )

    pyramids_list: list[list[torch.Tensor]] = [[] for _ in range(len(scale_list))]
    logger.info(f"Deriving {component.name} from VP, VS, RHO...")

    for name, f_data in facies_samples.items():
        derived = None
        if component == DataFiles.Ip:
            if name in vp_samples and name in rho_samples:
                derived = (vp_samples[name] * 1000.0) * rho_samples[name]
        elif component == DataFiles.Is:
            if name in vs_samples and name in rho_samples:
                derived = (vs_samples[name] * 1000.0) * rho_samples[name]
        elif component == DataFiles.VP_VS:
            if name in vp_samples and name in vs_samples:
                vp_data = vp_samples[name]
                vs_data = vs_samples[name]
                derived = np.divide(
                    vp_data, vs_data, out=np.zeros_like(vp_data), where=vs_data != 0
                )

        if derived is None:
            derived = np.zeros_like(f_data, dtype=np.float32)

        pyramid = interpolator.interpolate_array(derived.astype(np.float32), scale_list)
        for i in range(len(scale_list)):
            pyramids_list[i].append(norm(pyramid[i]))

    return tuple(_stack_and_format(pyramids_list, channels_last))


@memory.cache  # type: ignore
def to_vp_pyramids(
    scale_list: tuple[tuple[int, ...], ...],
    data_dir: str | None = None,
    channels_last: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Generate multi-scale pyramid tensors from VP volume."""
    global_stats = get_global_stats(data_dir)
    return _to_generic_pyramid(
        DataFiles.VP, scale_list, data_dir, channels_last, global_stats
    )


@memory.cache  # type: ignore
def to_vs_pyramids(
    scale_list: tuple[tuple[int, ...], ...],
    data_dir: str | None = None,
    channels_last: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Generate multi-scale pyramid tensors from VS volume."""
    global_stats = get_global_stats(data_dir)
    return _to_generic_pyramid(
        DataFiles.VS, scale_list, data_dir, channels_last, global_stats
    )


@memory.cache  # type: ignore
def to_rho_pyramids(
    scale_list: tuple[tuple[int, ...], ...],
    data_dir: str | None = None,
    channels_last: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Generate multi-scale pyramid tensors from RHO volume."""
    global_stats = get_global_stats(data_dir)
    return _to_generic_pyramid(
        DataFiles.RHO, scale_list, data_dir, channels_last, global_stats
    )


@memory.cache  # type: ignore
def to_seismic_pyramids(
    scale_list: tuple[tuple[int, ...], ...],
    data_dir: str | None = None,
    channels_last: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Generate multi-scale pyramid tensors for seismic data."""
    global_stats = get_global_stats(data_dir)
    stats = global_stats.get(DataFiles.SEISMIC.name, {})
    interpolator = SeismicInterpolator(
        InterpolatorConfig(
            channels_last=channels_last,
            data_min=stats.get("min"),
            data_max=stats.get("max"),
        )
    )
    return _build_pyramid_batch(
        DataFiles.SEISMIC, scale_list, interpolator, data_dir, channels_last
    )


@memory.cache  # type: ignore
def to_wells_pyramids(
    scale_list: tuple[tuple[int, ...], ...],
    data_dir: str | None = None,
    channels_last: bool = False,
    num_classes: int = 4,
) -> tuple[torch.Tensor, ...]:
    """Generate multi-scale pyramid tensors for well location data."""
    interpolator = WellInterpolator(
        InterpolatorConfig(channels_last=channels_last, num_classes=num_classes)
    )
    return _build_pyramid_batch(
        DataFiles.WELLS, scale_list, interpolator, data_dir, channels_last
    )


@memory.cache  # type: ignore
def to_masks_pyramids(
    scale_list: tuple[tuple[int, ...], ...],
    data_dir: str | None = None,
    channels_last: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Generate multi-scale pyramid tensors for binary mask data."""
    interpolator = MaskInterpolator(InterpolatorConfig(channels_last=channels_last))
    return _build_pyramid_batch(
        DataFiles.MASKS, scale_list, interpolator, data_dir, channels_last
    )


def build_conditioning_pyramids(
    options: TrainingOptions,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Build wells and seismic conditioning pyramid dicts for inference."""
    scales = generate_scales(options)
    wells_pyramid: dict[int, torch.Tensor] = {}
    seismic_pyramid: dict[int, torch.Tensor] = {}

    if options.use_wells:
        wp = to_wells_pyramids(
            scales, data_dir=options.input_path, num_classes=options.num_facies_classes
        )
        for s, w in enumerate(wp):
            if w.numel() > 0:
                wells_pyramid[s] = w

    if options.use_seismic:
        sp = to_seismic_pyramids(scales, data_dir=options.input_path)
        for s, se in enumerate(sp):
            if se.numel() > 0:
                seismic_pyramid[s] = se

    return wells_pyramid, seismic_pyramid


