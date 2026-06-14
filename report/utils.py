import os
from pathlib import Path
from typing import Callable

import torch

from options import TrainingOptions

REFERENCED_IMAGES: set[Path] = set()


def md_relpath(target: Path, start: Path) -> str:
    """Compute the relative path for Markdown links."""
    REFERENCED_IMAGES.add(target.resolve())
    return Path(os.path.relpath(target, start=start)).as_posix()


def get_milestones(epoch_indices: list[int]) -> list[int]:
    """Dynamically select milestones based on the final training epoch."""
    if not epoch_indices:
        return []
    final_epoch = epoch_indices[-1]
    targets = [0, final_epoch // 3, 2 * final_epoch // 3, final_epoch]
    milestones: list[int] = []
    for target in targets:
        closest = min(epoch_indices, key=lambda x: abs(x - target))
        milestones.append(closest)
    return sorted(list(set(milestones)))


def get_denormalize_fn(
    data_dir: Path,
) -> Callable[[torch.Tensor, str, TrainingOptions], torch.Tensor]:
    """Return a function that denormalizes property values using effective global stats."""
    from datasets.utils import get_effective_global_stats

    _stats_cache: dict[
        tuple[bool, tuple[float, ...]],
        dict[str, dict[str, float]],
    ] = {}

    def denormalize(
        data: torch.Tensor, prop_key: str, opt: TrainingOptions
    ) -> torch.Tensor:
        # Cache stats based on robust settings in opt
        robust = bool(getattr(opt, "vp_vs_robust_range", False))
        percentiles_attr = getattr(opt, "vp_vs_robust_percentiles", (1.0, 99.0))
        percentiles = (float(percentiles_attr[0]), float(percentiles_attr[1]))
        cache_key = (robust, percentiles)

        if cache_key not in _stats_cache:
            _stats_cache[cache_key] = get_effective_global_stats(
                str(data_dir),
                vp_vs_robust_range=robust,
                vp_vs_robust_percentiles=percentiles,
            )

        stats = _stats_cache[cache_key]

        # Map property keys to stats.json keys
        stat_key = {
            "ip": "Ip",
            "is": "Is",
            "vp_vs": "VP_VS",
            "vpvs": "VP_VS",
            "seismic": "SEISMIC",
        }.get(prop_key.lower(), prop_key.upper())

        if stat_key not in stats:
            return data

        s = stats[stat_key]
        p_min, p_max = s["min"], s["max"]

        # Get normalization range from options
        n_min, n_max = opt.normalization_range[0], opt.normalization_range[1]

        # Map from [n_min, n_max] -> [0, 1]
        data_01 = (data - n_min) / (n_max - n_min + 1e-8)
        # Map from [0, 1] -> [p_min, p_max]
        return data_01 * (p_max - p_min) + p_min

    return denormalize


def smooth(values: list[float], weight: float = 0.85) -> list[float]:
    """Apply Exponential Moving Average smoothing to a list of values."""
    if not values:
        return []
    smoothed: list[float] = []
    last = values[0]
    for val in values:
        smoothed_val = last * weight + (1.0 - weight) * val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def format_value(value: object) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def load_configs(outputs_dir: Path) -> dict[str, dict[str, object]]:
    """Load options.json configurations for all variants."""
    import json

    from report.constants import VARIANTS

    configs: dict[str, dict[str, object]] = {}
    for variant in VARIANTS:
        opts_path = outputs_dir / variant / "options.json"
        if opts_path.exists():
            with open(opts_path, encoding="utf-8") as handle:
                try:
                    configs[variant] = json.load(handle)
                except Exception:
                    pass
    return configs


def ensure_all_report_images(outputs_dir: Path, data_dir: Path) -> None:
    """Ensure all required report images are generated."""
    from report.image_utils import (
        ensure_distribution_histograms,
        ensure_loss_plots,
        ensure_pyramid_images,
        ensure_rock_physics_crossplots,
        ensure_seismic_images,
        ensure_training_pyramid_image,
        ensure_variogram_plots,
        ensure_well_images,
    )

    ensure_seismic_images(data_dir)
    ensure_well_images(data_dir)
    ensure_pyramid_images(outputs_dir)
    ensure_training_pyramid_image(outputs_dir, data_dir)
    ensure_loss_plots(outputs_dir)
    # ensure_rock_physics_crossplots(outputs_dir, data_dir)
    ensure_distribution_histograms(outputs_dir, data_dir)
    ensure_variogram_plots(outputs_dir, data_dir)
