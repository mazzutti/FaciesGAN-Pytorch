"""Visualization logic for experiments."""

import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

import utils

from .constants import VARIANT_LABELS, VARIANT_NAMES


def setup_imshow_for_kind(ax: Axes, img: np.ndarray, data_kind: str):
    """Helper to configure imshow with correct colormap and settings."""
    cmap = (
        "RdBu" if data_kind == "seismic" else ("viridis" if data_kind == "ip" else None)
    )
    return ax.imshow(img, cmap=cmap, interpolation="nearest", aspect="auto")  # type: ignore


def get_method_label(method: str) -> str:
    """Standardize method names (e.g., 'tsne' -> 't-SNE')."""
    return "t-SNE" if method.lower() == "tsne" else method.upper()


def get_data_kind_suffix(data_kind: str) -> str:
    """Return a filename suffix for non-facies data kinds."""
    return f"_{data_kind}" if data_kind != "facies" else ""


def get_gen_output_dir(base_output: str, variant_name: str) -> str:
    """Return the standard path for generated artifacts of a variant."""
    return os.path.join(base_output, variant_name, "generated")


def facies_to_rgb_img(img_arr: np.ndarray | None) -> np.ndarray | None:
    """Convert facies categorical/one-hot indices to RGB (H, W, 3)."""
    if img_arr is None:
        return None
    if img_arr.ndim == 2:  # Categorical
        return utils.facies_to_rgb(img_arr).transpose(1, 2, 0)
    if img_arr.shape[-1] > 3:  # One-hot
        return utils.facies_to_rgb(np.transpose(img_arr, (2, 0, 1))).transpose(1, 2, 0)
    return img_arr


def setup_scatter_plot(ax: Axes, method: str, data_kind: str, title: str) -> None:
    """Add standardized titles, labels, and legends to scatter plots."""
    label = get_method_label(method)
    ax.set_title(title)  # type: ignore
    ax.set_xlabel(f"{label} Dimension 1")  # type: ignore
    ax.set_ylabel(f"{label} Dimension 2")  # type: ignore
    ax.legend(loc="upper right", fontsize=8)  # type: ignore


def plot_sample_grid(
    all_generated: dict[str, list[np.ndarray]],
    real_samples: np.ndarray | None,
    base_output: str,
    data_kind: str,
    num_samples: int = 5,
) -> None:
    """Create a comparison grid of real vs generated samples per variant."""
    variant_labels = VARIANT_LABELS

    # Build per-variant rows: (label, real_img_or_None, [generated_imgs])
    rows: list[tuple[str, np.ndarray | None, list[np.ndarray]]] = []
    for i, name in enumerate(VARIANT_NAMES):
        if name not in all_generated or not all_generated[name]:
            continue

        samples = all_generated[name][:num_samples]
        imgs = [s.squeeze(0) if s.ndim == 4 else s for s in samples]
        real_img: np.ndarray | None = None
        if real_samples is not None and i < len(real_samples):
            real_img = real_samples[i]

        # Convert to RGB if plotting facies
        if data_kind == "facies":
            real_img = facies_to_rgb_img(real_img)
            rgb_imgs: list[np.ndarray] = []
            for img in imgs:
                rgb = facies_to_rgb_img(img)
                if rgb is not None:
                    rgb_imgs.append(rgb)
            imgs = rgb_imgs

        rows.append((variant_labels.get(name, name), real_img, imgs))

    if not rows:
        print(f"  No {data_kind} data available; skipping comparison grid.")
        return

    # Columns: 1 (Real) + num generated
    max_gen = max(len(imgs) for _, _, imgs in rows)
    n_cols = 1 + max_gen
    n_rows = len(rows)
    fig, axes = plt.subplots(  # type: ignore
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )

    # Column headers
    axes[0][0].set_title("Real", fontsize=11, fontweight="bold")
    for c in range(1, n_cols):
        axes[0][c].set_title(f"Generated {c}", fontsize=11)

    for r, (label, real_img, gen_imgs) in enumerate(rows):
        # Column 0: real sample
        ax = axes[r][0]
        if real_img is not None:
            img = real_img
            setup_imshow_for_kind(ax, img, data_kind)
        else:
            ax.set_facecolor("#111")
        ax.axis("off")
        ax.set_ylabel(label, fontsize=10, rotation=90, labelpad=10)

        # Columns 1+: generated samples
        for c in range(1, n_cols):
            ax = axes[r][c]
            gi = c - 1
            if gi < len(gen_imgs):
                img = gen_imgs[gi]
                setup_imshow_for_kind(ax, img, data_kind)
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
    data_kind: str = "facies",
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
    shared_embedding: tuple[np.ndarray, dict[str, np.ndarray]],
    base_output: str,
    num_iter: int = 0,
    data_kind: str = "facies",
) -> None:
    """Create a 2x2 combined embedding plot."""
    variant_labels = VARIANT_LABELS
    kind_title = data_kind.capitalize()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # type: ignore
    fig.suptitle(  # type: ignore
        f"{get_method_label(method)} Comparison: Real vs Generated {kind_title}",
        fontsize=14,
    )

    for ax, name in zip(axes.flat, VARIANT_NAMES):
        if name not in shared_embedding[1]:
            ax.set_title(f"{variant_labels.get(name, name)} (no data)")
            ax.axis("off")
            continue
        real_reduced = shared_embedding[0]
        fake_reduced = shared_embedding[1][name]

        ax.scatter(real_reduced[:, 0], real_reduced[:, 1], alpha=0.6, label="Real")
        ax.scatter(fake_reduced[:, 0], fake_reduced[:, 1], alpha=0.6, label="Generated")
        setup_scatter_plot(ax, method, data_kind, variant_labels.get(name, name))

    plt.tight_layout()
    epoch_tag = f"_epoch{num_iter}" if num_iter > 0 else ""
    combined_path = os.path.join(
        base_output, f"{method}_{data_kind}_comparison_all_variants{epoch_tag}.png"
    )
    plt.savefig(combined_path, dpi=150, bbox_inches="tight")  # type: ignore
    plt.close(fig)
    print(f"\nCombined {get_method_label(method)} {kind_title} plot -> {combined_path}")


def plot_per_facies_embedding(
    method: str,
    real_reduced: np.ndarray,
    fake_reduced_for_facies: np.ndarray,
    facies_idx: int,
    save_path: str,
    data_kind: str = "facies",
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
    shared: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]],
    all_mask_indexes: dict[str, torch.Tensor],
    base_output: str,
    methods: list[str],
    data_kind: str = "facies",
) -> None:
    """Generate per-crossline embedding plots for every variant and method."""
    for name in VARIANT_NAMES:
        if name not in all_mask_indexes:
            continue
        if not methods or not any(
            name in shared.get(m, (None, {}))[1] for m in methods  # type: ignore[index]
        ):
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
        print(
            f"  Per-{data_kind} embeddings ({name}): {n_plots} plots -> {per_emb_dir}"
        )


def plot_method_all_variants(
    shared: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]],
    method: str,
    base_output: str,
    num_iter: int,
    data_kind: str,
    all_mask_indexes: dict[str, torch.Tensor] | None = None,
    embedding_per_facies: bool = False,
) -> None:
    """Helper to generate all plots for a given embedding method and data kind."""
    if method not in shared:
        return

    real_reduced, per_variant_fakes = shared[method]

    # 1. Per-variant individual plots
    for name in VARIANT_NAMES:
        if name not in per_variant_fakes:
            continue
        gen_output = get_gen_output_dir(base_output, name)
        suffix = get_data_kind_suffix(data_kind)
        plot_path = os.path.join(gen_output, f"{method}{suffix}_comparison.png")
        plot_per_variant_embedding(
            method,
            real_reduced,
            per_variant_fakes[name],
            plot_path,
            data_kind=data_kind,
        )
        print(f"  {get_method_label(method)}{suffix} plot -> {plot_path}")

    # 2. Combined 2×2 grid
    plot_combined_embeddings(
        method, shared[method], base_output, num_iter, data_kind=data_kind
    )

    # 3. Per-crossline embeddings
    if embedding_per_facies and all_mask_indexes:
        save_per_facies_embeddings(
            shared, all_mask_indexes, base_output, [method], data_kind=data_kind
        )
