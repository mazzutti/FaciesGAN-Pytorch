import os
from pathlib import Path
from typing import Any, Callable

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


def save_dual_theme_plot(fig: Any, axes: Any, out_path: Path, dpi: int = 120) -> None:
    """Saves the figure in dark theme, then modifies it to light theme and saves a _light version."""
    from typing import Any as AnyType
    import matplotlib.colors as mcolors

    # 1. Save dark version (the figure is assumed to be styled for dark mode already)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="#151b26")

    # 2. Flatten axes to handle 1D, 2D arrays, lists or single axes objects
    if hasattr(axes, "flat"):
        ax_list = list(axes.flat)
    elif isinstance(axes, (list, tuple)):
        ax_list = list(axes)
    else:
        ax_list = [axes]

    # Include colorbar axes (and any other axes not in the grid)
    all_fig_axes = fig.get_axes()
    ax_set = set(id(a) for a in ax_list)
    for a in all_fig_axes:
        if id(a) not in ax_set:
            ax_list.append(a)

    # Helper to check if a color matches #f0f4f9
    def is_white_ish(color: AnyType) -> bool:
        if color is None:
            return False
        try:
            rgba = mcolors.to_rgba(color)
            return rgba[0] > 0.9 and rgba[1] > 0.9 and rgba[2] > 0.9
        except Exception:
            return False

    # 3. Modify for light theme
    fig.patch.set_facecolor("#ffffff")

    if hasattr(fig, "_suptitle") and fig._suptitle is not None:
        fig._suptitle.set_color("#1f2937")

    for ax in ax_list:
        if ax is None:
            continue
        ax.set_facecolor("#ffffff")

        if getattr(ax, "title", None) is not None:
            ax.title.set_color("#1f2937")

        if getattr(ax, "xaxis", None) is not None:
            ax.xaxis.label.set_color("#4b5563")
        if getattr(ax, "yaxis", None) is not None:
            ax.yaxis.label.set_color("#4b5563")

        ax.tick_params(axis="both", which="both", colors="#4b5563")

        for spine in ax.spines.values():
            if hasattr(spine, "set_color"):
                spine.set_color("#d0d7de")

        for line in ax.get_xgridlines() + ax.get_ygridlines():
            line.set_color("#e1e4e8")
            line.set_alpha(0.6)

        # Convert #f0f4f9 (off-white) elements to a visible dark gray
        # This handles lines (e.g. variogram reference) and patches (e.g. histograms)
        for line in ax.lines:
            if is_white_ish(line.get_color()):
                line.set_color("#4b5563")
        
        for patch in ax.patches:
            if is_white_ish(patch.get_facecolor()):
                # Keep alpha if it exists
                alpha = patch.get_alpha()
                if alpha is None:
                    alpha = patch.get_facecolor()[3]
                rgba = mcolors.to_rgba("#4b5563", alpha=alpha)
                patch.set_facecolor(rgba)

            if is_white_ish(patch.get_edgecolor()):
                alpha = patch.get_alpha()
                if alpha is None:
                    alpha = patch.get_edgecolor()[3]
                rgba = mcolors.to_rgba("#4b5563", alpha=alpha)
                patch.set_edgecolor(rgba)

        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor("#ffffff")
            legend.get_frame().set_edgecolor("#d0d7de")
            for text in legend.get_texts():
                text.set_color("#1f2937")
            
            # Recolor legend handles (like color boxes in histograms)
            handles = getattr(legend, "legend_handles", getattr(legend, "legendHandles", []))
            for handle in handles:
                if hasattr(handle, "get_color") and is_white_ish(handle.get_color()):
                    handle.set_color("#4b5563")
                if hasattr(handle, "get_facecolor") and is_white_ish(handle.get_facecolor()):
                    alpha = handle.get_alpha()
                    if alpha is None and hasattr(handle.get_facecolor(), "__getitem__"):
                        alpha = handle.get_facecolor()[3]
                    handle.set_facecolor(mcolors.to_rgba("#4b5563", alpha=alpha))
                if hasattr(handle, "get_edgecolor") and is_white_ish(handle.get_edgecolor()):
                    alpha = handle.get_alpha()
                    if alpha is None and hasattr(handle.get_edgecolor(), "__getitem__"):
                        alpha = handle.get_edgecolor()[3]
                    handle.set_edgecolor(mcolors.to_rgba("#4b5563", alpha=alpha))

    # 4. Save light version
    light_path = out_path.with_name(out_path.stem + "_light" + out_path.suffix)
    fig.savefig(light_path, dpi=dpi, bbox_inches="tight", facecolor="#ffffff")


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
    ensure_pyramid_images(outputs_dir, data_dir)
    ensure_training_pyramid_image(outputs_dir, data_dir)
    ensure_loss_plots(outputs_dir)
    # ensure_rock_physics_crossplots(outputs_dir, data_dir)
    ensure_distribution_histograms(outputs_dir, data_dir)
    ensure_variogram_plots(outputs_dir, data_dir)
