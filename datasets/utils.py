"""Utility functions for multiscale pyramid generation and data loading.

This module provides helpers to load numeric data from .npz archives, generate
progressive resolutions, and orchestrate the creation of multiscale tensors
for facies, wells, and rock-physics properties.
"""

import json
import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from joblib import Memory  # type: ignore
from numpy.typing import NDArray
from PIL import Image

from config import CheckpointFilenames, DirectoryConfig, DomainConfig, PhysicsConfig
from enums import DataFiles, DeviceType, StatKey
from options import NORMALIZATION_RANGE, TrainingOptions
from typedefs import FileLike


def generate_scales(
    options: TrainingOptions, channels_last: bool = False
) -> tuple[tuple[int, ...], ...]:
    """Generate multiscale pyramid resolutions for progressive training.

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
        ``(B, C, H, W)``. By default, False.

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
        raw = int(round(min(options.max_size, options.crop_size) * scale))
        if raw % 2 != 0:
            raw += 1
        out_shape = [raw, raw]

        num_ch = options.num_facies_channels

        if channels_last:
            shapes.append((options.batch_size, *out_shape, num_ch))
        else:
            shapes.append((options.batch_size, num_ch, *out_shape))

    return tuple(shapes)


logger = logging.getLogger(__name__)

# Create a cache directory for joblib memory
memory = Memory(DirectoryConfig.JOBLIB_CACHE, verbose=0)  # type: ignore


def load_samples(
    component: DataFiles, data_dir: str | None = None
) -> dict[str, NDArray[np.float32]]:
    """Load all numeric samples for a component from a .npz archive.

    Parameters
    ----------
    component : DataFiles
        The data component to load (e.g., FACIES, VP, VS).
    data_dir : str, optional
        Path to the data directory. If None, uses DirectoryConfig.DEFAULT_DATA.

    Returns
    -------
    dict[str, NDArray[np.float32]]
        A dictionary mapping sample names (stems) to NumPy arrays.
    """
    base_dir = Path(data_dir) if data_dir else Path(DirectoryConfig.DEFAULT_DATA)
    npz_path = base_dir / f"{component.name.lower()}.npz"

    if not npz_path.exists():
        logger.error(f"Required data file missing: {npz_path}")
        return {}

    try:
        with np.load(npz_path) as data:
            return cast(
                dict[str, NDArray[np.float32]], {k: data[k] for k in sorted(data.files)}
            )
    except (OSError, KeyError) as e:
        logger.error(f"Failed to load {npz_path}: {e}")
        return {}


def load_image(image_path: FileLike) -> NDArray[np.float32]:
    """Load an image from disk and convert it to a normalized float32 numpy array.

    Parameters
    ----------
    image_path : FileLike
        Filesystem path to the image file to load.

    Returns
    -------
    NDArray[np.float32]
        RGB image as a float32 array with shape (H, W, 3) and values
        normalized to the range [0, 1].
    """
    img_pil: Image.Image = Image.open(image_path).convert("RGB")
    img_np = np.array(img_pil).astype(np.float32, copy=False) / 255.0
    return cast(NDArray[np.float32], img_np)


@lru_cache(maxsize=8)
def get_global_stats(data_dir: str | None = None) -> dict[str, dict[str, float]]:
    """Retrieve or compute global min/max statistics for all data components.

    Statistics are cached in data/stats.json. If the file exists, it is loaded.
    Otherwise, statistics (min, max, mean) are computed from .npz archives and
    persisted for reuse.

    Parameters
    ----------
    data_dir : str, optional
        Path to the data directory. If None, uses DirectoryConfig.DEFAULT_DATA.

    Returns
    -------
    dict[str, dict[str, float]]
        A nested dictionary mapping component names (e.g., 'SEISMIC', 'VP') to
        dictionaries containing min, max, and mean values.
    """
    base_dir = Path(data_dir if data_dir else DirectoryConfig.DEFAULT_DATA)
    stats_path = base_dir / CheckpointFilenames.STATS

    if stats_path.exists():
        try:
            with open(stats_path, "r") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to load stats from %s: %e", stats_path, e)

    # Compute and save if not exists
    print("Computing global dataset statistics from .npz archives...")
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
                    all_values: list[NDArray[np.float32]] = []
                    for key in data_dict.files:
                        all_values.append(data_dict[key].flatten())

                    if all_values:
                        combined = np.concatenate(all_values)
                        stats[comp.name] = {
                            StatKey.MIN: float(combined.min()),
                            StatKey.MAX: float(combined.max()),
                            StatKey.MEAN: float(combined.mean()),
                        }
            except (OSError, KeyError, ValueError) as e:
                logger.warning(f"Failed to compute stats for {comp.name}: {e}")
        else:
            # For derived components, compute if base components exist
            if comp in [DataFiles.Ip, DataFiles.Is, DataFiles.VP_VS]:
                stats[comp.name] = _compute_derived_global_stats(base_dir, comp)

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)

    return stats


def get_effective_global_stats(
    data_dir: str | None = None,
    vp_vs_robust_range: bool = False,
    vp_vs_robust_percentiles: tuple[float, float] = (1.0, 99.0),
) -> dict[str, dict[str, float]]:
    """Return global stats with optional robust VP/VS min/max override.

    When robust mode is enabled, VP/VS min/max are replaced with percentile-
    based bounds while preserving other component stats.
    """
    stats = dict(get_global_stats(data_dir))
    if not vp_vs_robust_range:
        return stats

    low, high = float(vp_vs_robust_percentiles[0]), float(vp_vs_robust_percentiles[1])
    if not (0.0 <= low < high <= 100.0):
        logger.warning(
            "Invalid VP/VS robust percentiles (%s, %s); using global min/max.",
            low,
            high,
        )
        return stats

    pmin, pmax = _compute_vp_vs_percentile_range(data_dir, low, high)
    if pmax <= pmin:
        logger.warning(
            "Invalid robust VP/VS range [%.6f, %.6f]; using global min/max.", pmin, pmax
        )
        return stats

    vp_vs_stats = dict(stats.get(DataFiles.VP_VS.name, {}))
    vp_vs_stats[StatKey.MIN] = float(pmin)
    vp_vs_stats[StatKey.MAX] = float(pmax)
    stats[DataFiles.VP_VS.name] = vp_vs_stats
    print(
        f"Using robust VP/VS stats from percentiles {low:.2f}/{high:.2f}: [{pmin:.6f}, {pmax:.6f}]"
    )
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
        derived = _derive_rock_physics_component(
            component, name, vp_samples, vs_samples, rho_samples
        )

        if derived is not None:
            c_min = min(c_min, float(derived.min()))
            c_max = max(c_max, float(derived.max()))
            c_sum += float(derived.sum())
            c_count += derived.size

    if c_min == float("inf"):
        return {str(StatKey.MIN): 0.0, str(StatKey.MAX): 1.0, str(StatKey.MEAN): 0.5}
    return {
        str(StatKey.MIN): c_min,
        str(StatKey.MAX): c_max,
        str(StatKey.MEAN): c_sum / c_count if c_count > 0 else (c_min + c_max) / 2.0,
    }


def _derive_rock_physics_component(
    component: DataFiles,
    name: str,
    vp_samples: dict[str, NDArray[np.float32]],
    vs_samples: dict[str, NDArray[np.float32]],
    rho_samples: dict[str, NDArray[np.float32]],
) -> NDArray[np.float32] | None:
    """Compute one sample's derived rock-physics array (Ip, Is, or Vp/Vs).

    Returns ``None`` when the required base samples are not available.
    """
    if component == DataFiles.Ip:
        if name in vp_samples and name in rho_samples:
            return np.multiply(
                np.multiply(vp_samples[name], PhysicsConfig.VP_MS_SCALE),
                rho_samples[name],
            )
    elif component == DataFiles.Is:
        if name in vs_samples and name in rho_samples:
            return np.multiply(
                np.multiply(vs_samples[name], PhysicsConfig.VP_MS_SCALE),
                rho_samples[name],
            )
    elif component == DataFiles.VP_VS:
        if name in vp_samples and name in vs_samples:
            vp_data = vp_samples[name]
            vs_data = vs_samples[name]
            return np.divide(
                vp_data, vs_data, out=np.zeros_like(vp_data), where=vs_data != 0
            )
    return None


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


from interpolators.config import InterpolationStrategy, InterpolatorConfig
from interpolators.neural import NeuralSmoother
from interpolators.numeric import NumericInterpolator
from interpolators.seismic import SeismicInterpolator
from interpolators.well import WellInterpolator


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

    base_dir = Path(data_dir) if data_dir else Path(DirectoryConfig.DEFAULT_DATA)
    npz_path = base_dir / f"{comp_name}.npz"

    if not npz_path.exists():
        logger.warning(f"Data file {npz_path} not found. Returning empty pyramids.")
        return tuple(_empty_pyramid_tensor(s, channels_last) for s in scale_list)

    pyramids_list: list[list[torch.Tensor]] = [[] for _ in range(len(scale_list))]

    print(f"Loading {data_file.name} from {npz_path}")
    with np.load(npz_path) as data_dict:
        for key in sorted(data_dict.files):
            data = data_dict[key]
            pyramid = interpolator.interpolate_array(data, scale_list)
            for i in range(len(scale_list)):
                pyramids_list[i].append(pyramid[i])

    return tuple(_stack_and_format(pyramids_list, channels_last, is_mask))


def _to_generic_pyramid(
    data_file: DataFiles,
    scale_list: tuple[tuple[int, ...], ...],
    data_dir: str | None = None,
    channels_last: bool = False,
    global_stats: dict[str, dict[str, float]] | None = None,
    normalization_range: tuple[float, float] = (0.0, 1.0),
) -> tuple[torch.Tensor, ...]:
    """Generic helper to load and interpolate numeric data into a pyramid."""
    stats = global_stats.get(data_file.name, {}) if global_stats else {}
    interpolator = NumericInterpolator(
        InterpolatorConfig(
            channels_last=channels_last,
            strategy=InterpolationStrategy.CONTINUOUS,
            data_min=stats.get(StatKey.MIN),
            data_max=stats.get(StatKey.MAX),
            normalization_range=normalization_range,
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
    num_classes: int = DomainConfig.NUM_FACIES,
    normalization_range: tuple[float, float] = NORMALIZATION_RANGE,
) -> tuple[torch.Tensor, ...]:
    """Generate multi-scale pyramid tensors for facies images using neural interpolation.

    Images are loaded from ``{data_dir}/facies/facies_images.npz`` (key ``images``,
    shape ``(N, H, W, 3)``) and the corresponding trained ResidualMLP checkpoints
    are loaded from ``{data_dir}/facies/facies_checkpoints.ptz`` (keys
    ``xz_crossline_000`` … ``xz_crossline_NNN``).  Each image/checkpoint pair is
    processed with :class:`~interpolators.neural.NeuralSmoother` to produce
    smoothed multi-scale tensors whose values are linearly mapped to
    *normalization_range*.

    Parameters
    ----------
    scale_list : tuple[tuple[int, ...], ...]
        Tuple of scale descriptors produced by :func:`generate_scales`. Each
        element is a 4-tuple ``(batch, channels, height, width)``.
    data_dir : str, optional
        Base data directory.  Defaults to :data:`DirectoryConfig.DEFAULT_DATA`.
    channels_last : bool, optional
        Whether to use channels-last layout ``(N, H, W, C)`` instead of the
        default ``(N, C, H, W)``.  Defaults to ``False``.
    num_classes : int, optional
        Number of facies classes used when building the :class:`NeuralSmoother`.
        Defaults to :class:`DomainConfig.NUM_FACIES`.
    normalization_range : tuple[float, float], optional
        Target value range ``(min, max)`` for the output tensors.  The neural
        smoother natively outputs values in ``[0, 1]``; this parameter linearly
        maps them to the desired range.  Defaults to :data:`NORMALIZATION_RANGE`.

    Returns
    -------
    tuple[torch.Tensor, ...]
        One tensor per scale, each with shape ``(N, C, H, W)`` (or ``(N, H, W, C)``
        when *channels_last* is ``True``) and values in *normalization_range*.
    """
    base_dir = Path(data_dir) if data_dir else Path(DirectoryConfig.DEFAULT_DATA)
    images_path = base_dir / "facies" / "facies_images.npz"
    checkpoints_path = base_dir / "facies" / "facies_checkpoints.ptz"

    if not images_path.exists() or not checkpoints_path.exists():
        logger.warning(
            "Facies bundle files not found (%s, %s). Returning empty pyramids.",
            images_path,
            checkpoints_path,
        )
        return tuple(
            _empty_pyramid_tensor(scale, channels_last) for scale in scale_list
        )

    # Load all images — shape (N, H, W, 3), uint8 or float32
    with np.load(str(images_path)) as npz:
        images_array: np.ndarray = npz["images"]

    # Normalize to float32 [0, 1]
    images_f32 = images_array.astype(np.float32, copy=False)
    if images_f32.max() > 1.0:
        images_f32 = images_f32 / 255.0

    # Load all checkpoints — {xz_crossline_NNN: state_dict}
    import torch as _torch  # local alias to keep the function import-safe under joblib

    all_checkpoints: dict[str, Any] = _torch.load(
        str(checkpoints_path), map_location=DeviceType.CPU, weights_only=False
    )

    n = len(images_f32)
    config = InterpolatorConfig(channels_last=channels_last, num_classes=num_classes)
    pyramids_list: list[list[torch.Tensor]] = [[] for _ in range(len(scale_list))]

    norm_min = float(normalization_range[0])
    norm_max = float(normalization_range[1])
    # NeuralSmoother outputs values in [0, 1]; linearly map to normalization_range.
    # If range is already [0, 1] this is a no-op (scale=1, shift=0).
    norm_scale = norm_max - norm_min
    norm_shift = norm_min

    for i in range(n):
        key = f"xz_crossline_{i:03d}"
        state = all_checkpoints.get(key)
        if state is None:
            logger.warning(
                "Checkpoint key %r not found in %s; skipping.", key, checkpoints_path
            )
            continue

        smoother = NeuralSmoother.from_state_dict(state, config)
        pyramid = smoother.interpolate_from_array(images_f32[i], scale_list)
        for j in range(len(scale_list)):
            t = pyramid[j]
            if norm_scale != 1.0 or norm_shift != 0.0:
                t = t * norm_scale + norm_shift
            pyramids_list[j].append(t)

    if not pyramids_list[0]:
        return tuple(
            _empty_pyramid_tensor(scale, channels_last) for scale in scale_list
        )

    return tuple(_stack_and_format(pyramids_list, channels_last))


# NOTE: The following cached wrappers must remain as *separate named functions*
# because joblib.Memory.cache uses the function object's identity as part of the
# cache key.  Collapsing them into a single parameterized function would cause
# all derived-pyramid flavors to share one cache entry.
@memory.cache  # type: ignore
def to_ip_pyramids(
    scale_list: tuple[tuple[int, ...], ...],
    data_dir: str | None = None,
    channels_last: bool = False,
    normalization_range: tuple[float, float] = NORMALIZATION_RANGE,
) -> tuple[torch.Tensor, ...]:
    """Generate multiscale pyramid tensors from Ip volume."""
    global_stats = get_global_stats(data_dir)
    return _to_derived_pyramid(
        DataFiles.Ip,
        scale_list,
        data_dir,
        channels_last,
        global_stats,
        normalization_range,
    )


@memory.cache  # type: ignore
def to_is_pyramids(
    scale_list: tuple[tuple[int, ...], ...],
    data_dir: str | None = None,
    channels_last: bool = False,
    normalization_range: tuple[float, float] = NORMALIZATION_RANGE,
) -> tuple[torch.Tensor, ...]:
    """Generate multiscale pyramid tensors from Is volume."""
    global_stats = get_global_stats(data_dir)
    return _to_derived_pyramid(
        DataFiles.Is,
        scale_list,
        data_dir,
        channels_last,
        global_stats,
        normalization_range,
    )


@memory.cache  # type: ignore
def to_vp_vs_pyramids(
    scale_list: tuple[tuple[int, ...], ...],
    data_dir: str | None = None,
    channels_last: bool = False,
    normalization_range: tuple[float, float] = NORMALIZATION_RANGE,
    use_robust_range: bool = False,
    robust_percentiles: tuple[float, float] = (1.0, 99.0),
) -> tuple[torch.Tensor, ...]:
    """Generate multiscale pyramid tensors from Vp/Vs volume."""
    global_stats = get_effective_global_stats(
        data_dir,
        vp_vs_robust_range=use_robust_range,
        vp_vs_robust_percentiles=robust_percentiles,
    )

    return _to_derived_pyramid(
        DataFiles.VP_VS,
        scale_list,
        data_dir,
        channels_last,
        global_stats,
        normalization_range,
    )


@lru_cache(maxsize=8)
def _compute_vp_vs_percentile_range(
    data_dir: str | None, low_percentile: float, high_percentile: float
) -> tuple[float, float]:
    """Compute VP/VS percentile-based robust physical range."""
    base_dir = Path(data_dir if data_dir else DirectoryConfig.DEFAULT_DATA)
    vp_vs_path = base_dir / f"{DataFiles.VP_VS.name.lower()}.npz"

    values: list[NDArray[np.float32]] = []
    if vp_vs_path.exists():
        with np.load(vp_vs_path) as data_dict:
            for key in data_dict.files:
                values.append(data_dict[key].reshape(-1).astype(np.float32, copy=False))
    else:
        vp_samples = load_samples(DataFiles.VP, str(base_dir))
        vs_samples = load_samples(DataFiles.VS, str(base_dir))
        for name, vp_data in vp_samples.items():
            vs_data = vs_samples.get(name)
            if vs_data is None:
                continue
            ratio = np.divide(
                vp_data,
                vs_data,
                out=np.zeros_like(vp_data, dtype=np.float32),
                where=vs_data != 0,
            )
            values.append(ratio.reshape(-1).astype(np.float32, copy=False))

    if not values:
        return 0.0, 1.0

    all_values = np.concatenate(values)
    pmin = float(np.percentile(all_values, low_percentile))
    pmax = float(np.percentile(all_values, high_percentile))
    return pmin, pmax


def _to_derived_pyramid(
    component: DataFiles,
    scale_list: tuple[tuple[int, ...], ...],
    data_dir: str | None = None,
    channels_last: bool = False,
    global_stats: dict[str, dict[str, float]] | None = None,
    normalization_range: tuple[float, float] = NORMALIZATION_RANGE,
) -> tuple[torch.Tensor, ...]:
    """Generic helper to derive Ip, Is or Vp/Vs from Vp, Vs and Rho if missing."""
    base_dir = Path(data_dir) if data_dir else Path(DirectoryConfig.DEFAULT_DATA)
    npz_path = base_dir / f"{component.name.lower()}.npz"

    if npz_path.exists():
        return _to_generic_pyramid(
            component,
            scale_list,
            data_dir,
            channels_last,
            global_stats,
            normalization_range,
        )

    # Derive from base components (VP, VS, RHO)
    vp_samples = load_samples(DataFiles.VP, data_dir)
    vs_samples = load_samples(DataFiles.VS, data_dir)
    rho_samples = load_samples(DataFiles.RHO, data_dir)

    if not vp_samples:
        return tuple(_empty_pyramid_tensor(s, channels_last) for s in scale_list)

    stats = global_stats.get(component.name, {}) if global_stats else {}
    interpolator = NumericInterpolator(
        InterpolatorConfig(
            channels_last=channels_last,
            strategy=InterpolationStrategy.CONTINUOUS,
            data_min=stats.get(StatKey.MIN),
            data_max=stats.get(StatKey.MAX),
            normalization_range=normalization_range,
        )
    )

    pyramids_list: list[list[torch.Tensor]] = [[] for _ in range(len(scale_list))]
    print(f"Deriving {component.name} from VP, VS, RHO...")

    for name, vp_data in vp_samples.items():
        derived = _derive_rock_physics_component(
            component, name, vp_samples, vs_samples, rho_samples
        )

        if derived is None:
            derived = np.zeros_like(vp_data, dtype=np.float32)

        pyramid = interpolator.interpolate_array(derived.astype(np.float32), scale_list)
        for i in range(len(scale_list)):
            pyramids_list[i].append(pyramid[i])

    return tuple(_stack_and_format(pyramids_list, channels_last))


@memory.cache  # type: ignore
def to_vp_pyramids(
    scale_list: tuple[tuple[int, ...], ...],
    data_dir: str | None = None,
    channels_last: bool = False,
    normalization_range: tuple[float, float] = NORMALIZATION_RANGE,
) -> tuple[torch.Tensor, ...]:
    """Generate multiscale pyramid tensors from VP volume."""
    global_stats = get_global_stats(data_dir)
    return _to_generic_pyramid(
        DataFiles.VP,
        scale_list,
        data_dir,
        channels_last,
        global_stats,
        normalization_range,
    )


@memory.cache  # type: ignore
def to_vs_pyramids(
    scale_list: tuple[tuple[int, ...], ...],
    data_dir: str | None = None,
    channels_last: bool = False,
    normalization_range: tuple[float, float] = NORMALIZATION_RANGE,
) -> tuple[torch.Tensor, ...]:
    """Generate multiscale pyramid tensors from VS volume."""
    global_stats = get_global_stats(data_dir)
    return _to_generic_pyramid(
        DataFiles.VS,
        scale_list,
        data_dir,
        channels_last,
        global_stats,
        normalization_range,
    )


@memory.cache  # type: ignore
def to_rho_pyramids(
    scale_list: tuple[tuple[int, ...], ...],
    data_dir: str | None = None,
    channels_last: bool = False,
    normalization_range: tuple[float, float] = NORMALIZATION_RANGE,
) -> tuple[torch.Tensor, ...]:
    """Generate multiscale pyramid tensors from RHO volume."""
    global_stats = get_global_stats(data_dir)
    return _to_generic_pyramid(
        DataFiles.RHO,
        scale_list,
        data_dir,
        channels_last,
        global_stats,
        normalization_range,
    )


@memory.cache  # type: ignore
def to_seismic_pyramids(
    scale_list: tuple[tuple[int, ...], ...],
    data_dir: str | None = None,
    channels_last: bool = False,
    normalization_range: tuple[float, float] = NORMALIZATION_RANGE,
) -> tuple[torch.Tensor, ...]:
    """Generate multiscale pyramid tensors for seismic data."""
    global_stats = get_global_stats(data_dir)
    stats = global_stats.get(DataFiles.SEISMIC.name, {})
    interpolator = SeismicInterpolator(
        InterpolatorConfig(
            channels_last=channels_last,
            data_min=stats.get(StatKey.MIN),
            data_max=stats.get(StatKey.MAX),
            normalization_range=normalization_range,
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
    num_classes: int = DomainConfig.NUM_FACIES,
    normalization_range: tuple[float, float] = NORMALIZATION_RANGE,
) -> tuple[torch.Tensor, ...]:
    """Generate multiscale RGB pyramid tensors for well location data.

    Uses :class:`~interpolators.well.WellInterpolator` with Majority Vote (Mode
    Pooling) to downsample vertical well traces. The categorical labels are
    mapped to a standard RGB palette and normalized to *normalization_range*.
    Non-well columns are filled with ``normalization_range[0]``.

    Parameters
    ----------
    scale_list : tuple[tuple[int, ...], ...]
        Tuple of scale descriptors produced by :func:`generate_scales`.
    data_dir : str, optional
        Base data directory.  Defaults to :data:`DirectoryConfig.DEFAULT_DATA`.
    channels_last : bool, optional
        Whether to use channels-last layout.  Defaults to ``False``.
    num_classes : int, optional
        Number of facies classes.  Defaults to :class:`DomainConfig.NUM_FACIES`.
    normalization_range : tuple[float, float], optional
        Target value range. Defaults to :data:`NORMALIZATION_RANGE`.

    Returns
    -------
    tuple[torch.Tensor, ...]
        One tensor per scale with shape ``(N, C, H, W)`` or ``(N, H, W, C)``
        and values in *normalization_range*.
    """
    base_dir = Path(data_dir) if data_dir else Path(DirectoryConfig.DEFAULT_DATA)
    wells_path = base_dir / "wells.npz"

    if not wells_path.exists():
        logger.warning("Wells file %s not found. Returning empty pyramids.", wells_path)
        return tuple(_empty_pyramid_tensor(scale, channels_last) for scale in scale_list)

    # Standard RGB palette for facies classes (0:Black, 1:Red, 2:Blue, 3:Green)
    # Scaled to [0, 1] for processing.
    palette = torch.tensor(
        [
            [0, 0, 0],  # 0: Floodplain
            [255, 0, 0],  # 1: Point bar
            [0, 0, 255],  # 2: Channel
            [0, 255, 0],  # 3: Boundary
        ],
        dtype=torch.float32,
    ) / 255.0

    wells_data = dict(np.load(str(wells_path)))

    config = InterpolatorConfig(channels_last=channels_last, num_classes=num_classes)
    interpolator = WellInterpolator(config)

    pyramids_list: list[list[torch.Tensor]] = [[] for _ in range(len(scale_list))]

    norm_min, norm_max = float(normalization_range[0]), float(normalization_range[1])
    norm_scale = norm_max - norm_min
    norm_shift = norm_min

    for _, key in enumerate(sorted(wells_data.keys())):
        well_labels = wells_data[key]
        # Identify original well columns to mask out background later
        # (Assuming 0 is the background facies but can still be part of a well)
        nonzero_cols = np.where(well_labels.any(axis=0))[0]

        # Generate categorical label pyramid (H, W)
        pyramid = interpolator.interpolate_array(well_labels, scale_list)

        for scale_idx, resolution in enumerate(scale_list):
            if channels_last:
                _, _, new_w, _ = resolution
            else:
                _, _, _, new_w = resolution

            # labels is (H, W)
            labels = pyramid[scale_idx]

            # 2. Map labels to RGB (H, W, 3)
            # Ensure palette is on the same device
            rgb = palette[labels.clamp(0, palette.size(0) - 1)].to(labels.device)

            # 3. Mask non-well columns (Zero RGB = Black)
            well_width = float(well_labels.shape[1])
            scaled_indices = (nonzero_cols.astype(np.float32) * new_w / well_width).astype(np.int32)
            scaled_cols = np.unique(np.clip(scaled_indices, 0, new_w - 1))
            col_mask = torch.zeros(new_w, dtype=torch.bool, device=labels.device)
            col_mask[scaled_cols] = True
            rgb[:, ~col_mask, :] = 0.0

            # 4. Normalize to [-1, 1] (or target normalization_range)
            # Black (0, 0, 0) becomes (-1, -1, -1)
            rgb = rgb * norm_scale + norm_shift

            # 5. Permute if not channels_last
            if not channels_last:
                rgb = rgb.permute(2, 0, 1)

            pyramids_list[scale_idx].append(rgb)

    if not pyramids_list[0]:
        return tuple(_empty_pyramid_tensor(scale, channels_last) for scale in scale_list)

    return tuple(torch.stack(pyramid, dim=0) for pyramid in pyramids_list)


@memory.cache  # type: ignore
def to_masks_pyramids(
    scale_list: tuple[tuple[int, ...], ...],
    data_dir: str | None = None,
    channels_last: bool = False,
    num_classes: int = DomainConfig.NUM_FACIES,
    normalization_range: tuple[float, float] = NORMALIZATION_RANGE,
) -> tuple[torch.Tensor, ...]:
    """Generate multiscale binary mask pyramids from well locations.

    Reuses the cached output of :func:`to_wells_pyramids`. A pixel is set to
    ``1.0`` (well trace present) if any channel at that location differs from
    the background fill value (``normalization_range[0]``), and ``0.0``
    otherwise. The result is a single-channel binary tensor.

    Parameters
    ----------
    scale_list : tuple[tuple[int, ...], ...]
        Tuple of scale descriptors produced by :func:`generate_scales`.
    data_dir : str, optional
        Base data directory.  Defaults to :data:`DirectoryConfig.DEFAULT_DATA`.
    channels_last : bool, optional
        Whether to use channels-last layout.  Defaults to ``False``.
    num_classes : int, optional
        Number of facies classes. Defaults to :data:`DomainConfig.NUM_FACIES`.
    normalization_range : tuple[float, float], optional
        Target value range; forwarded to :func:`to_wells_pyramids`. Defaults to
        :data:`NORMALIZATION_RANGE`.

    Returns
    -------
    tuple[torch.Tensor, ...]
        One tensor per scale with shape ``(N, 1, H, W)`` or ``(N, H, W, 1)``
        containing binary ``{0, 1}`` values.
    """
    wells_pyramids = to_wells_pyramids(
        scale_list, data_dir, channels_last, num_classes, normalization_range
    )
    if not wells_pyramids or wells_pyramids[0].numel() == 0:
        return tuple(
            _empty_pyramid_tensor(scale, channels_last) for scale in scale_list
        )

    bg_value = float(normalization_range[0])
    channel_dim = -1 if channels_last else 1
    masks_pyramids: list[torch.Tensor] = []

    for well_t in wells_pyramids:
        # Check across channels: if any channel != bg_value, there is a well trace
        mask = (well_t != bg_value).any(dim=channel_dim, keepdim=True).float()
        masks_pyramids.append(mask)

    return tuple(masks_pyramids)


def build_conditioning_pyramids(
    options: TrainingOptions,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Build wells and seismic conditioning pyramid dicts for inference.

    Parameters
    ----------
    options : TrainingOptions
        Training configuration specifying which conditioning inputs to include.

    Returns
    -------
    tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]
        A tuple of (wells_pyramid, seismic_pyramid) where each is a dictionary
        mapping scale indices to corresponding tensors. Scales with no data are
        omitted from the dictionaries.
    """
    scales = generate_scales(options)
    wells_pyramid: dict[int, torch.Tensor] = {}
    seismic_pyramid: dict[int, torch.Tensor] = {}

    if options.use_wells:
        wp = to_wells_pyramids(
            scales, data_dir=options.input_path, num_classes=options.num_facies
        )
        for s, w in enumerate(wp):
            if w.numel() > 0:
                wells_pyramid[s] = w

    if options.use_seismic:
        normalization_range: tuple[float, float] = (
            float(options.normalization_range[0]),
            float(options.normalization_range[1]),
        )
        sp = to_seismic_pyramids(
            scales, data_dir=options.input_path, normalization_range=normalization_range
        )
        for s, se in enumerate(sp):
            if se.numel() > 0:
                seismic_pyramid[s] = se

    return wells_pyramid, seismic_pyramid
