"""Visualization logic for experiments."""

import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, cast

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

# Use non-interactive backend
matplotlib.use("Agg")

import utils
from device import device_manager
from enums import DataFiles, ExperimentVariant


def setup_imshow_for_kind(ax: Axes, img: np.ndarray, data_kind: str):
    """Helper to configure imshow with correct colormap and settings."""
    cmap = (
        "RdBu"
        if data_kind == DataFiles.SEISMIC.name.lower()
        else (
            "magma"
            if data_kind in [DataFiles.Ip.name.lower(), DataFiles.Is.name.lower()]
            else ("viridis" if data_kind == DataFiles.VP_VS.name.lower() else None)
        )
    )
    return ax.imshow(img, cmap=cmap, interpolation="nearest", aspect="auto")  # type: ignore


def get_method_label(method: str) -> str:
    """Standardize method names (e.g., 'tsne' -> 't-SNE')."""
    return "t-SNE" if method.lower() == "tsne" else method.upper()


def get_data_kind_suffix(data_kind: str) -> str:
    """Return a filename suffix for non-facies data kinds."""
    return f"_{data_kind}" if data_kind != DataFiles.FACIES.name.lower() else ""


def get_gen_output_dir(base_output: str, variant_name: str) -> str:
    """Return the standard path for generated artifacts of a variant."""
    return os.path.join(base_output, variant_name, "generated")


def facies_to_rgb_img(img_arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Convert facies categorical/one-hot indices to RGB (H, W, 3).

    Handles both channels-first ``(C, H, W)`` (PyTorch / generator output)
    and channels-last ``(H, W, C)`` (from ``torch2np`` / real data) layouts.
    ``utils.facies_to_rgb`` expects ``(C, H, W)``.
    """
    if img_arr is None:
        return None
    if img_arr.ndim == 2:  # Categorical (H, W)
        return utils.facies_to_rgb(img_arr).transpose(1, 2, 0)
    if img_arr.ndim == 3:
        # Distinguish layout by comparing first vs last dim:
        # channels-first (C, H, W): shape[0] is small (num_classes), shape[-1] is large (W)
        # channels-last  (H, W, C): shape[0] is large (H), shape[-1] is small (num_classes)
        if img_arr.shape[0] < img_arr.shape[-1]:
            # Already (C, H, W)
            return utils.facies_to_rgb(img_arr).transpose(1, 2, 0)
        else:
            # (H, W, C) — transpose to (C, H, W) first
            return utils.facies_to_rgb(np.transpose(img_arr, (2, 0, 1))).transpose(
                1, 2, 0
            )
    return img_arr


def setup_scatter_plot(ax: Axes, method: str, data_kind: str, title: str) -> None:
    """Add standardized titles, labels, and legends to scatter plots."""
    label = get_method_label(method)
    ax.set_title(title)  # type: ignore
    ax.set_xlabel(f"{label} Dimension 1")  # type: ignore
    ax.set_ylabel(f"{label} Dimension 2")  # type: ignore
    ax.legend(loc="upper right", fontsize=8)  # type: ignore


def plot_sample_grid(
    all_generated: Dict[str, List[np.ndarray]],
    real_samples: Optional[np.ndarray],
    base_output: str,
    data_kind: str,
    all_mask_indexes: Optional[Dict[str, torch.Tensor]] = None,
    num_samples: int = 5,
    seed: Optional[int] = 42,
) -> None:
    """Create a comparison grid where each row is a different variant and columns are samples.

    Columns: [Real | Sample 1 | Sample 2 | ... ]
    Rows: [Variant A | Variant B | ... ]
    """
    # Find variants that actually have data
    active_variants = [
        v for v in ExperimentVariant if v.id in all_generated and all_generated[v.id]
    ]
    if not active_variants:
        print(f"  No {data_kind} data available; skipping comparison grid.")
        return

    # Use the first active variant to determine sample count and indices
    ref_name = active_variants[0].id
    available_samples = len(all_generated[ref_name])
    actual_num_samples = min(num_samples, available_samples)
    if real_samples is not None:
        print(f"    [Debug] real_samples count: {len(real_samples)}")

    # 1. Determine which generated indices to plot (Random selection)
    rng = np.random.RandomState(seed if seed is not None else 42)
    indices_to_plot: np.ndarray
    unique_conds: List[int] = []

    if (
        all_mask_indexes
        and ref_name in all_mask_indexes
        and len(all_mask_indexes[ref_name]) >= actual_num_samples
    ):
        # 1. Group all generated indices by their absolute conditioning index
        cond_map: Dict[int, List[int]] = defaultdict(list)
        all_mi = device_manager.to_numpy(all_mask_indexes[ref_name])
        for i, val in enumerate(all_mi):
            cond_map[int(val)].append(i)

        unique_conds = sorted(cond_map.keys())

        if len(unique_conds) >= actual_num_samples:
            # Pick a diverse set of unique conditionings
            selected_conds = rng.choice(unique_conds, actual_num_samples, replace=False)
            # For each selected conditioning, pick a random realization from the generated set
            indices_to_plot = np.array(
                [rng.choice(cond_map[int(c)]) for c in selected_conds]
            )
        else:
            # If we want more rows than unique conditionings, allow duplicates but still randomize
            indices_to_plot = rng.choice(
                available_samples, actual_num_samples, replace=False
            )
    else:
        indices_to_plot = rng.choice(
            available_samples, actual_num_samples, replace=False
        )

    # Sort indices to keep them in order of generation for cleaner labels
    indices_to_plot.sort()

    if all_mask_indexes and ref_name in all_mask_indexes:
        # Map absolute indices back to their 0-indexed position in the unique conditioning list
        abs_vals = device_manager.to_numpy(
            all_mask_indexes[ref_name][indices_to_plot]
        ).tolist()
        # unique_conds was sorted, so we use it as the reference for 'Conditioning ID'
        cond_ids = [unique_conds.index(int(v)) for v in abs_vals]

        print(f"    [Plot] Selection: {len(indices_to_plot)} samples")
        print(f"    [Plot] Conditioning ID (0-{len(unique_conds)-1}): {cond_ids}")
        print(
            f"    [Plot] Realization ID (0-{available_samples-1}): {indices_to_plot.tolist()}"
        )
        print(f"    [Plot] Absolute Dataset Index: {abs_vals}")
    else:
        print(
            f"    [Plot] Selection: {len(indices_to_plot)} samples at indices {indices_to_plot.tolist()}"
        )

    rows: List[Tuple[str, Optional[np.ndarray], List[np.ndarray]]] = []
    for s_idx in indices_to_plot:
        real_img: Optional[np.ndarray] = None

        # Determine real image for this sample row using mask indices if available
        if (
            all_mask_indexes
            and ref_name in all_mask_indexes
            and s_idx < len(all_mask_indexes[ref_name])
        ):
            real_idx = int(all_mask_indexes[ref_name][s_idx])
            if real_samples is not None and real_idx < len(real_samples):
                real_img = real_samples[real_idx]
        elif real_samples is not None and s_idx < len(real_samples):
            # Fallback to direct indexing if mask info is missing
            real_img = real_samples[s_idx]
            print(f"    [Debug] Row for s_idx={s_idx} uses fallback real_idx={s_idx}")

        # Convert Real to RGB if facies
        if data_kind == DataFiles.FACIES.name.lower():
            real_img = facies_to_rgb_img(real_img)

        # Collect generated samples from each variant for this row
        variant_imgs: List[np.ndarray] = []
        for v in active_variants:
            raw_img = cast(np.ndarray, all_generated[v.id][s_idx])
            img = cast(Optional[np.ndarray], raw_img)

            if data_kind == DataFiles.FACIES.name.lower():
                img = facies_to_rgb_img(img)

            if img is not None:
                variant_imgs.append(np.squeeze(img))

        rows.append((f"Sample {s_idx + 1}", real_img, variant_imgs))

    if not rows:
        print(f"  No {data_kind} data available; skipping comparison grid.")
        return

    # Columns: 1 (Real) + one for each active variant
    n_cols = 1 + len(active_variants)
    n_rows = len(rows)
    fig, axes = plt.subplots(  # type: ignore
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )

    # Column headers
    axes[0][0].set_title("Real", fontsize=11, fontweight="bold")
    for c, v in enumerate(active_variants):
        axes[0][c + 1].set_title(v.value.label, fontsize=11, fontweight="bold")

    for r, (label, real_img, variant_imgs) in enumerate(rows):
        # Column 0: real sample
        ax = axes[r][0]
        if real_img is not None:
            setup_imshow_for_kind(ax, real_img, data_kind)
        else:
            ax.set_facecolor("#111")
        ax.axis("off")
        ax.set_ylabel(label, fontsize=10, rotation=90, labelpad=10)

    for r, (_, _, variant_imgs) in enumerate(rows):
        # Columns 1+: variant samples
        for c in range(1, n_cols):
            ax = axes[r][c]
            vi = c - 1
            if vi < len(variant_imgs):
                setup_imshow_for_kind(ax, variant_imgs[vi], data_kind)
            else:
                ax.set_facecolor("#111")
            ax.axis("off")

    kind_title = data_kind.capitalize()
    fig.suptitle(  # type: ignore
        f"Real vs Generated {kind_title} — All Variants", fontsize=14
    )
    fig.tight_layout()
    out_path = os.path.join(base_output, f"{data_kind}_comparison_all_variants.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")  # type: ignore
    plt.close(fig)
    print(f"  {kind_title} comparison grid -> {out_path}")


def plot_per_variant_embedding(
    method: str,
    real_reduced: np.ndarray,
    fake_reduced: np.ndarray,
    save_path: str,
    data_kind: str = DataFiles.FACIES.name.lower(),
) -> None:
    """Save a single per-variant embedding plot using pre-computed coordinates."""
    kind_title = data_kind.capitalize()
    fig, ax = plt.subplots()  # type: ignore
    ax.scatter(  # type: ignore
        real_reduced[:, 0], real_reduced[:, 1], alpha=0.6, label=f"Real {kind_title}"
    )
    ax.scatter(  # type: ignore
        fake_reduced[:, 0], fake_reduced[:, 1], alpha=0.6, label=f"Fake {kind_title}"
    )
    setup_scatter_plot(
        ax,
        method,
        data_kind,
        f"{get_method_label(method)}: Real vs Generated {kind_title}",
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")  # type: ignore
    plt.close(fig)


def plot_combined_embeddings(
    method: str,
    shared_embedding: Tuple[np.ndarray, Dict[str, np.ndarray]],
    base_output: str,
    num_iter: int = 0,
    data_kind: str = DataFiles.FACIES.name.lower(),
) -> None:
    """Create a 2x2 combined embedding plot."""
    kind_title = data_kind.capitalize()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # type: ignore
    fig.suptitle(  # type: ignore
        f"{get_method_label(method)} Comparison: Real vs Generated {kind_title}",
        fontsize=14,
    )

    for ax, variant in zip(axes.flat, ExperimentVariant):
        name = variant.id
        if name not in shared_embedding[1]:
            ax.set_title(f"{variant.value.label} (no data)")
            ax.axis("off")
            continue
        real_reduced = shared_embedding[0]
        fake_reduced = shared_embedding[1][name]

        ax.scatter(real_reduced[:, 0], real_reduced[:, 1], alpha=0.6, label="Real")
        ax.scatter(fake_reduced[:, 0], fake_reduced[:, 1], alpha=0.6, label="Generated")
        setup_scatter_plot(ax, method, data_kind, variant.value.label)

    plt.tight_layout()
    epoch_tag = f"_epoch{num_iter}" if num_iter > 0 else ""
    combined_path = os.path.join(
        base_output, f"{method}_{data_kind}_comparison_all_variants{epoch_tag}.png"
    )
    plt.savefig(combined_path, dpi=150, bbox_inches="tight")  # type: ignore
    plt.close(fig)
    print(f"  Combined {get_method_label(method)} {kind_title} grid -> {combined_path}")


def plot_per_facies_embedding(
    method: str,
    real_reduced: np.ndarray,
    fake_reduced_for_facies: np.ndarray,
    facies_idx: int,
    save_path: str,
    data_kind: str = DataFiles.FACIES.name.lower(),
) -> None:
    """Save an embedding plot for a single conditioning crossline index."""
    kind_title = data_kind.capitalize()
    fig, ax = plt.subplots()  # type: ignore
    ax.scatter(  # type: ignore
        real_reduced[:, 0],
        real_reduced[:, 1],
        alpha=0.4,
        s=10,
        label="Real (all)",
        c="steelblue",
    )
    ax.scatter(  # type: ignore
        fake_reduced_for_facies[:, 0],
        fake_reduced_for_facies[:, 1],
        alpha=0.8,
        s=20,
        label=f"Generated (crossline {facies_idx})",
        c="tomato",
    )
    setup_scatter_plot(
        ax,
        method,
        data_kind,
        f"{get_method_label(method)} — {kind_title} crossline {facies_idx}",
    )
    plt.savefig(save_path, dpi=120, bbox_inches="tight")  # type: ignore
    plt.close(fig)


def save_per_facies_embeddings(
    shared: Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray]]],
    all_mask_indexes: Dict[str, torch.Tensor],
    base_output: str,
    methods: List[str],
    data_kind: str = DataFiles.FACIES.name.lower(),
) -> None:
    """Generate per-crossline embedding plots for every variant and method."""
    for variant in ExperimentVariant:
        name = variant.id
        if name not in all_mask_indexes:
            continue

        # Check if any method has data for this variant
        has_data = False
        for m in methods:
            if m in shared and name in shared[m][1]:
                has_data = True
                break
        if not has_data:
            continue

        mi_list = np.array(all_mask_indexes[name])
        unique_idxs = sorted(set(mi_list.tolist()))

        gen_output = get_gen_output_dir(base_output, name)
        per_emb_dir = os.path.join(gen_output, f"per_{data_kind}_embeddings")
        os.makedirs(per_emb_dir, exist_ok=True)

        for method in methods:
            if method not in shared:
                continue
            real_reduced, per_variant_fakes = shared[method]
            if name not in per_variant_fakes:
                continue
            fake_all = per_variant_fakes[name]  # (N_generated, 2)

            for idx in unique_idxs:
                mask = mi_list == idx
                fake_for_idx = fake_all[mask]
                if fake_for_idx.shape[0] == 0:
                    continue
                save_path = os.path.join(
                    per_emb_dir,
                    f"{method}_{data_kind}_crossline_{idx:04d}.png",
                )
                plot_per_facies_embedding(
                    method,
                    real_reduced,
                    fake_for_idx,
                    idx,
                    save_path,
                    data_kind=data_kind,
                )
        n_plots = len(unique_idxs) * len(methods)
        print(f"      → Per-{data_kind} embeddings ({name}): {n_plots} plots")


def plot_method_all_variants(
    shared: Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray]]],
    method: str,
    base_output: str,
    num_iter: int,
    data_kind: str,
    all_mask_indexes: Optional[Dict[str, torch.Tensor]] = None,
    embedding_per_facies: bool = False,
) -> None:
    """Helper to generate all plots for a given embedding method and data kind."""
    if method not in shared:
        return

    real_reduced, per_variant_fakes = shared[method]
    label = get_method_label(method)
    suffix = get_data_kind_suffix(data_kind)
    kind_title = data_kind.capitalize()

    print(f"\n  --- {label} ({kind_title}) ---")

    # 1. Per-variant individual plots
    for variant in ExperimentVariant:
        name = variant.id
        if name not in per_variant_fakes:
            continue
        gen_output = get_gen_output_dir(base_output, name)
        plot_path = os.path.join(gen_output, f"{method}{suffix}_comparison.png")
        plot_per_variant_embedding(
            method,
            real_reduced,
            per_variant_fakes[name],
            plot_path,
            data_kind=data_kind,
        )
        print(f"      → {variant.value.label} plot: {os.path.basename(plot_path)}")

    # 2. Combined 2×2 grid
    plot_combined_embeddings(
        method, shared[method], base_output, num_iter, data_kind=data_kind
    )

    # 3. Per-crossline embeddings
    if embedding_per_facies and all_mask_indexes:
        save_per_facies_embeddings(
            shared, all_mask_indexes, base_output, [method], data_kind=data_kind
        )
