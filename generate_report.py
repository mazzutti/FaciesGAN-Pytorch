import argparse
import logging
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.image import imread

# Use relative imports if needed, but since this is usually run as a script:
from config import DirectoryConfig

logger = logging.getLogger(__name__)

VARIANTS = ["wells_seismic", "wells_only", "seismic_only", "unconditional"]
VARIANT_LABELS = {
    "wells_seismic": "Wells + Seismic",
    "wells_only": "Wells Only",
    "seismic_only": "Seismic Only",
    "unconditional": "Unconditional",
}
EMBEDDING_METHODS = ["mds", "umap", "isomap", "tsne"]
EMBEDDING_LABELS = {"mds": "MDS", "umap": "UMAP", "isomap": "Isomap", "tsne": "t-SNE"}
EPOCHS_MILESTONES = [100, 200, 300, 400, 500]
FINAL_SCALE = 6
TRAINING_EPOCHS = [99, 199, 299, 399, 499, 599]

# Key hyperparameters to show in the report
KEY_HPARAMS = [
    ("num_iter", "Training epochs"),
    ("batch_size", "Batch size"),
    ("lr_g", "Generator LR"),
    ("lr_d", "Discriminator LR"),
    ("rec_facies_loss_penalty", "Reconstruction weight (α)"),
    ("diversity_loss_penalty", "Diversity loss weight"),
    ("discriminator_steps", "Discriminator steps"),
    ("generator_steps", "Generator steps"),
    ("scale0_noise_amp", "Scale-0 noise amp"),
    ("min_noise_amp", "Min noise amp"),
    ("stop_scale", "Stop scale"),
    ("num_parallel_scales", "Parallel scales"),
    ("num_train_pyramids", "Training pyramids"),
    ("well_loss_penalty", "Well loss penalty"),
    ("use_wells", "Uses wells"),
    ("use_seismic", "Uses seismic"),
]


def _add_title_page(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(11, 8.5))  # type: ignore
    fig.text(  # type: ignore
        0.5,
        0.55,
        "FaciesGAN — Conditioning Ablation Study",
        ha="center",
        va="center",
        fontsize=26,
        fontweight="bold",
    )
    fig.text(  # type: ignore
        0.5,
        0.45,
        "Experiment Report",
        ha="center",
        va="center",
        fontsize=18,
        color="gray",
    )

    variant_text = "Variants: " + " · ".join(VARIANT_LABELS.values())
    fig.text(0.5, 0.35, variant_text, ha="center", va="center", fontsize=12)  # type: ignore

    pdf.savefig(fig)  # type: ignore
    plt.close(fig)


def _add_hyperparams_page(pdf: PdfPages, outputs_dir: Path) -> None:
    """One page with a table comparing hyperparameters across variants."""
    configs: dict[str, dict[str, float]] = {}
    for v in VARIANTS:
        opts_path = outputs_dir / v / "options.json"
        if opts_path.exists():
            with open(opts_path) as f:
                configs[v] = json.load(f)

    if not configs:
        return

    fig, ax = plt.subplots(figsize=(11, 8.5))  # type: ignore
    ax.axis("off")
    fig.suptitle("Training Hyperparameters", fontsize=18, fontweight="bold", y=0.95)  # type: ignore

    col_labels = ["Parameter"] + [VARIANT_LABELS[v] for v in VARIANTS if v in configs]
    rows: list[list[str]] = []
    for key, label in KEY_HPARAMS:
        row = [label]
        for v in VARIANTS:
            if v in configs:
                val = configs[v].get(key, "—")
                row.append(str(val))
        rows.append(row)

    table = ax.table(  # type: ignore
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", fontweight="bold")  # type: ignore

    # Alternate row shading
    for i in range(1, len(rows) + 1):
        for j in range(len(col_labels)):
            if i % 2 == 0:
                table[i, j].set_facecolor("#D9E2F3")

    pdf.savefig(fig)  # type: ignore
    plt.close(fig)


def _add_seismic_data_page(pdf: PdfPages, data_dir: Path) -> None:
    """2x3 grid showing seismic data examples."""
    images: list[np.ndarray] = []
    titles: list[str] = []

    # Legacy path: pre-rendered seismic PNGs.
    seismic_dir = data_dir / "seismic"
    seismic_files = sorted(seismic_dir.glob("xz_crossline_*.png"))
    if seismic_files:
        # Pick 6 evenly spaced images
        indices = [int(i * (len(seismic_files) - 1) / 5) for i in range(6)]
        for i in indices:
            img_path = seismic_files[i]
            images.append(imread(str(img_path)))  # type: ignore
            titles.append(img_path.stem)  # type: ignore
    else:
        # Current data layout: seismic stored in NPZ file.
        seismic_npz = data_dir / "seismic.npz"
        if seismic_npz.exists():
            try:
                with np.load(seismic_npz) as data:
                    keys = sorted(data.files)
                    if keys:
                        n_examples = min(6, len(keys))
                        if n_examples == 1:
                            idxs = [0]
                        else:
                            idxs = [
                                int(i * (len(keys) - 1) / (n_examples - 1))
                                for i in range(n_examples)
                            ]
                        seen: set[int] = set()
                        for i in idxs:
                            if i in seen:
                                continue
                            seen.add(i)
                            key = keys[i]
                            arr = np.asarray(data[key]).squeeze()
                            if arr.ndim == 3:
                                # Accept CHW or HWC and extract first channel.
                                if arr.shape[0] <= 4 and arr.shape[0] < arr.shape[-1]:
                                    arr = arr[0]
                                else:
                                    arr = arr[..., 0]
                            if arr.ndim != 2:
                                continue
                            images.append(arr)
                            titles.append(key)
            except Exception:
                return

    if not images:
        return

    fig, axes = plt.subplots(2, 3, figsize=(11, 7.5))  # type: ignore
    fig.suptitle(  # type: ignore
        "Seismic Data Examples",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    for i in range(2):
        for j in range(3):
            idx = i * 3 + j
            ax = axes[i][j]
            if idx < len(images):  # type: ignore
                ax.imshow(images[idx], cmap="gray")
                ax.set_title(titles[idx], fontsize=10)
            ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.95])  # type: ignore
    pdf.savefig(fig)  # type: ignore
    plt.close(fig)


def _add_image_page(
    pdf: PdfPages,
    img_path: Path | None,
    title: str,
    *,
    img_array=None,  # type: ignore
) -> None:
    """Full-page image with a title."""
    if img_array is not None:
        img = img_array  # type: ignore
    elif img_path is not None and img_path.exists():
        img = imread(str(img_path))
    else:
        return
    h, w = img.shape[:2]  # type: ignore
    aspect = w / h  # type: ignore

    fig_w, fig_h = 11, 8.5
    fig = plt.figure(figsize=(fig_w, fig_h))  # type: ignore
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)  # type: ignore

    # Leave room for title
    usable_h = fig_h - 0.6
    usable_w = fig_w - 0.4
    if aspect > usable_w / usable_h:
        ax_w = usable_w
        ax_h = ax_w / aspect  # type: ignore
    else:
        ax_h = usable_h
        ax_w = ax_h * aspect  # type: ignore

    left = (fig_w - ax_w) / 2 / fig_w  # type: ignore
    bottom = (fig_h - 0.5 - ax_h) / 2 / fig_h  # type: ignore
    ax = fig.add_axes([left, bottom, ax_w / fig_w, ax_h / fig_h])  # type: ignore
    ax.imshow(img)  # type: ignore
    ax.axis("off")  # type: ignore

    pdf.savefig(fig)  # type: ignore
    plt.close(fig)


def _add_training_progression_page(
    pdf: PdfPages,
    outputs_dir: Path,
    variant: str,
) -> None:
    """Show training sample progression at the final scale across epochs."""
    sample_dir = outputs_dir / variant / str(FINAL_SCALE) / "real_x_generated_facies"
    images = []
    epoch_labels = []
    for epoch in TRAINING_EPOCHS:
        # Pattern: gen_{scale}_{batch}_{epoch}.png — batch index varies
        candidates = sorted(sample_dir.glob(f"gen_{FINAL_SCALE}_*_{epoch}.png"))
        if candidates:
            images.append(imread(str(candidates[0])))  # type: ignore
            epoch_labels.append(f"Epoch {epoch + 1}")  # type: ignore

    if not images:
        return

    for img, label in zip(images, epoch_labels):  # type: ignore
        _add_image_page(
            pdf,
            None,
            f"{VARIANT_LABELS[variant]} — {label} (Scale {FINAL_SCALE})",
            img_array=img,  # type: ignore
        )


def _add_section_divider(pdf: PdfPages, title: str, subtitle: str = "") -> None:
    fig = plt.figure(figsize=(11, 8.5))  # type: ignore
    fig.text(  # type: ignore
        0.5,
        0.55,
        title,
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
    )
    if subtitle:
        fig.text(  # type: ignore
            0.5,
            0.43,
            subtitle,
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
        )
    pdf.savefig(fig)  # type: ignore
    plt.close(fig)


def _add_combined_embedding_grid(
    pdf: PdfPages,
    outputs_dir: Path,
    method: str,
) -> None:
    """2×3 grid showing epoch progression for one embedding method across all variants."""
    label = EMBEDDING_LABELS[method]
    images = []
    titles = []
    for epoch in EPOCHS_MILESTONES:
        path = outputs_dir / f"{method}_comparison_all_variants_epoch{epoch}.png"
        if path.exists():
            images.append(imread(str(path)))  # type: ignore
            titles.append(f"Epoch {epoch}")  # type: ignore

    if not images:
        return

    n = len(images)  # type: ignore
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, 8.5))  # type: ignore
    fig.suptitle(  # type: ignore
        f"{label} — All Variants Comparison (Epoch Progression)",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            ax = axes[i][j]
            if idx < n:
                ax.imshow(images[idx])
                ax.set_title(titles[idx], fontsize=10)
            ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.95])  # type: ignore
    pdf.savefig(fig)  # type: ignore
    plt.close(fig)


def generate_report(
    outputs_dir: Path | None = None,
    data_dir: Path | None = None,
    output: str | None = None,
) -> str:
    if outputs_dir is None:
        outputs_dir = Path(DirectoryConfig.OUTPUTS)
    else:
        outputs_dir = Path(outputs_dir)

    if data_dir is None:
        data_dir = Path(DirectoryConfig.DATA)
    else:
        data_dir = Path(data_dir)

    if output is None:
        output = str(outputs_dir / "experiment_report.pdf")

    print(f"Generating report from: {outputs_dir}")

    with PdfPages(output) as pdf:
        # --- Title page ---
        _add_title_page(pdf)

        # --- Seismic data examples ---
        _add_seismic_data_page(pdf, data_dir)

        # --- Hyperparameters ---
        _add_hyperparams_page(pdf, outputs_dir)

        # --- Per-variant sections ---
        for variant in VARIANTS:
            label = VARIANT_LABELS[variant]
            _add_section_divider(
                pdf,
                label,
                f"Training progression, generated samples, and embedding analysis",
            )

            # Training progression at final scale
            _add_training_progression_page(pdf, outputs_dir, variant)

            # Per-variant embedding comparison plots
            for method in EMBEDDING_METHODS:
                img_path = (
                    outputs_dir / variant / "generated" / f"{method}_comparison.png"
                )
                _add_image_page(
                    pdf,
                    img_path,
                    f"{label} — {EMBEDDING_LABELS[method]} Embedding Comparison",
                )

        # --- Combined all-variants comparison ---
        _add_section_divider(
            pdf,
            "Cross-Variant Comparison",
            "Embedding analysis across all conditioning variants over training epochs",
        )

        for method in EMBEDDING_METHODS:
            _add_combined_embedding_grid(pdf, outputs_dir, method)

    print(f"Report saved to: {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a PDF report presenting FaciesGAN conditioning-ablation experiment outputs."
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=DirectoryConfig.OUTPUTS,
        help="Path to the directory containing experiment outputs.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DirectoryConfig.DATA,
        help="Path to the directory containing input data.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the PDF report. Defaults to outputs_dir/experiment_report.pdf.",
    )

    args = parser.parse_args()
    generate_report(
        outputs_dir=args.outputs_dir, data_dir=args.data_dir, output=args.output
    )
