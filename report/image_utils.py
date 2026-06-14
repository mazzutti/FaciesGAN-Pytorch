# pyright: reportMissingTypeStubs=false

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from report.constants import VARIANT_LABELS, VARIANTS
from report.utils import get_denormalize_fn, smooth


def ensure_seismic_images(data_dir: Path) -> None:
    """Ensure that sample seismic crossline PNG files exist in data/seismic/."""
    seismic_dir = data_dir / "seismic"
    seismic_npz = data_dir / "seismic.npz"
    if not seismic_npz.exists():
        return

    seismic_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt

        with np.load(seismic_npz) as data:
            keys = sorted(data.files)
            if not keys:
                return
            n_examples = min(5, len(keys))
            idxs = (
                [0]
                if n_examples == 1
                else [
                    int(i * (len(keys) - 1) / (n_examples - 1))
                    for i in range(n_examples)
                ]
            )

            for idx in idxs:
                key = keys[idx]
                png_path = seismic_dir / f"{key}.png"
                array: np.ndarray = data[key]
                # Symmetric stretch around zero for RdBu colormap
                # We center the data by subtracting the mean (DC bias)
                array_plot = array - np.mean(array)
                p_lo = float(np.percentile(array_plot, 2))
                p_hi = float(np.percentile(array_plot, 98))
                max_abs = max(abs(p_lo), abs(p_hi), 1e-6)

                plt.figure(figsize=(6, 4.5))  # type: ignore
                plt.imshow(  # type: ignore
                    array_plot,
                    cmap="RdBu",
                    aspect="auto",
                    vmin=-max_abs,
                    vmax=max_abs,
                    interpolation="bilinear",
                )
                plt.title(  # type: ignore
                    f"Seismic Profile: {key}",
                    fontsize=11,
                    fontweight="bold",
                    pad=10,
                )
                plt.xlabel("Traces (x)", fontsize=9)  # type: ignore
                plt.ylabel("Depth (z)", fontsize=9)  # type: ignore
                plt.colorbar(label="Amplitude")  # type: ignore
                plt.tight_layout()
                plt.savefig(png_path, dpi=150)  # type: ignore
                plt.close()
                print(f"Generated seismic crossline image: {png_path}")
    except Exception as e:
        print(f"Error generating seismic crossline images: {e}")


def ensure_well_images(data_dir: Path) -> None:
    """Ensure that sample well conditioning crossline PNG files exist in data/wells_visualizations/."""
    well_dir = data_dir / "wells_visualizations"
    wells_npz = data_dir / "wells.npz"
    if not wells_npz.exists():
        return

    well_dir.mkdir(parents=True, exist_ok=True)

    # Check if we already have the PNG files
    png_files = list(well_dir.glob("well_crossline_*.png"))
    if len(png_files) >= 5:
        return

    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch

        # Define custom colormap matching report colors
        # 0: Charcoal, 1: Red, 2: Blue, 3: Green
        well_colors = ["#1a1a1a", "#ff3f3f", "#3867d6", "#20bf6b"]
        cmap = ListedColormap(well_colors)

        with np.load(wells_npz) as data:
            keys = sorted(data.files)
            if not keys:
                return
            n_examples = min(5, len(keys))
            idxs = (
                [0]
                if n_examples == 1
                else [
                    int(i * (len(keys) - 1) / (n_examples - 1))
                    for i in range(n_examples)
                ]
            )

            for idx in idxs:
                key = keys[idx]
                png_path = well_dir / f"well_{key}.png"
                if not png_path.exists():
                    arr = data[key]
                    h, w = arr.shape

                    nonzero_cols = np.where(arr.any(axis=0))[0]
                    plot_arr = np.full((h, w), np.nan)

                    # Fill well columns and widen them to 3 pixels for visibility
                    for c in nonzero_cols:
                        for offset in range(-1, 2):
                            col_idx = c + offset
                            if 0 <= col_idx < w:
                                plot_arr[:, col_idx] = arr[:, c]

                    plt.figure(figsize=(6, 4.5))  # type: ignore
                    ax = plt.gca()
                    ax.set_facecolor("#ffffff")  # White background

                    ax.imshow(  # type: ignore
                        plot_arr,
                        cmap=cmap,
                        aspect="auto",
                        vmin=0,
                        vmax=3,
                    )

                    plt.title(  # type: ignore
                        f"Well Conditioning Traces: {key}",
                        fontsize=11,
                        fontweight="bold",
                        pad=10,
                    )
                    plt.xlabel("Traces (x)", fontsize=9)  # type: ignore
                    plt.ylabel("Depth (z)", fontsize=9)  # type: ignore

                    # Create custom colorbar legend
                    legend_elements = [
                        Patch(facecolor="#1a1a1a", label="Floodplain (0)"),
                        Patch(facecolor="#ff3f3f", label="Point bar (1)"),
                        Patch(facecolor="#3867d6", label="Channel (2)"),
                        Patch(facecolor="#20bf6b", label="Boundary (3)"),
                    ]
                    ax.legend(  # type: ignore
                        handles=legend_elements,
                        loc="upper right",
                        fontsize=8,
                    )

                    plt.tight_layout()
                    plt.savefig(png_path, dpi=150)  # type: ignore
                    plt.close()
                    print(f"Generated well crossline image: {png_path}")
    except Exception as e:
        print(f"Error generating well crossline images: {e}")


def ensure_pyramid_images(outputs_dir: Path) -> None:
    """Generate composite pyramid visualizations showing all scales for each variant."""
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError:
        return
    plt = cast(Any, plt)

    for variant in VARIANTS:
        variant_dir = outputs_dir / variant
        if not variant_dir.exists():
            continue

        out_path = variant_dir / "pyramid_overview.png"

        # Load option parameters from options.json for this variant
        options_path = variant_dir / "options.json"
        use_ip, use_is, use_vpvs = True, True, True
        if options_path.exists():
            try:
                with open(options_path, encoding="utf-8") as f:
                    opt_dict = json.load(f)
                    use_ip = opt_dict.get("use_ip", True)
                    use_is = opt_dict.get("use_is", True)
                    use_vpvs = opt_dict.get("use_vpvs", True)
            except Exception:
                pass

        properties = ["Facies"]
        prop_labels = ["Facies"]
        if use_ip:
            properties.append("Ip")
            prop_labels.append("Ip")
        if use_is:
            properties.append("Is")
            prop_labels.append("Is")
        if use_vpvs:
            properties.append("VpVs")
            prop_labels.append("Vp/Vs")
        if use_ip:  # seismic requires Ip
            properties.append("Seismic")
            prop_labels.append("Seismic")

        # Discover available scales
        scale_dirs = sorted(
            [d for d in variant_dir.iterdir() if d.is_dir() and d.name.isdigit()],
            key=lambda d: int(d.name),
        )
        if not scale_dirs:
            continue

        # Find the final epoch available in the finest scale's training_visualizations
        finest_scale = max(int(d.name) for d in scale_dirs)
        finest_viz = variant_dir / "training_visualizations" / f"Scale_{finest_scale}"
        if not finest_viz.is_dir():
            continue

        epoch_indices: list[int] = []
        for p in finest_viz.glob("Facies_epoch_*.png"):
            parts = p.stem.split("_")
            if len(parts) >= 3:
                try:
                    epoch_indices.append(int(parts[2]))
                except ValueError:
                    continue
        if not epoch_indices:
            continue

        final_epoch = max(epoch_indices)
        num_scales = finest_scale + 1

        # Check all images exist
        all_exist = True
        for scale in range(num_scales):
            viz_dir = variant_dir / "training_visualizations" / f"Scale_{scale}"
            for prop in properties:
                img_path = viz_dir / f"{prop}_epoch_{final_epoch:05d}.png"
                if not img_path.exists():
                    all_exist = False
                    break
            if not all_exist:
                break

        if not all_exist:
            continue

        # Build composite figure: rows = properties, cols = scales
        fig, axes = plt.subplots(
            len(properties),
            num_scales,
            figsize=(3 * num_scales, 3 * len(properties)),
            squeeze=False,
        )
        fig.patch.set_facecolor("#151b26")

        for row, (prop, prop_label) in enumerate(zip(properties, prop_labels)):
            for col in range(num_scales):
                ax = axes[row, col]
                viz_dir = variant_dir / "training_visualizations" / f"Scale_{col}"
                img_path = viz_dir / f"{prop}_epoch_{final_epoch:05d}.png"
                img = Image.open(img_path)
                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_facecolor("#111622")
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if row == 0:
                    w, h = img.size
                    ax.set_title(
                        f"Scale {col}\n({w}×{h})",
                        fontsize=10,
                        fontweight="bold",
                        color="#f0f4f9",
                    )
                if col == 0:
                    ax.set_ylabel(
                        prop_label,
                        fontsize=12,
                        fontweight="bold",
                        rotation=90,
                        labelpad=15,
                        color="#f0f4f9",
                    )

        fig.suptitle(  # type: ignore
            f"Multi-Scale Training Pyramid — {VARIANT_LABELS[variant]} (Epoch {final_epoch})",
            fontsize=14,
            fontweight="bold",
            y=1.01,
            color="#f0f4f9",
        )
        plt.tight_layout()
        plt.savefig(  # type: ignore
            out_path,
            dpi=120,
            bbox_inches="tight",
            facecolor="#151b26",
        )
        plt.close()
        print(f"Generated pyramid overview: {out_path}")


def ensure_training_pyramid_image(outputs_dir: Path, data_dir: Path) -> None:
    """Generate a visualization of the actual training dataset pyramids."""
    out_path = outputs_dir / "training_pyramid_samples.png"

    # Try to load latest hyperparameters from the wells_seismic variant if available
    base_options_path = outputs_dir / "wells_seismic" / "options.json"
    latest_params: dict[str, Any] = {}
    if base_options_path.exists():
        try:
            with open(base_options_path, encoding="utf-8") as f:
                latest_params = json.load(f)
            print(f"Loaded latest hyperparameters from {base_options_path}")
        except Exception as e:
            print(f"Warning: Could not load latest hyperparameters: {e}")

    print("Generating training pyramid samples visualization...")
    try:
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        import torch

        import utils
        from datasets.dataset import PyramidsDataset
        from models.palette import PALETTE_RGB
        from options import TrainingOptions

        plt = cast(Any, plt)

        opt = TrainingOptions()
        # Override with latest parameters from options.json
        for key, val in latest_params.items():
            if key and hasattr(opt, key):
                setattr(opt, key, val)

        opt.input_path = str(data_dir)
        opt.use_rock_physics = True
        opt.use_seismic = True
        opt.use_wells = True

        dataset = PyramidsDataset(opt)
        num_scales = len(dataset.scales)
        num_facies_ch = opt.num_facies_channels

        use_ip = getattr(opt, "use_ip", True)
        use_is = getattr(opt, "use_is", True)
        use_vpvs = getattr(opt, "use_vpvs", True)

        active_props = ["Facies"]
        prop_labels = ["Facies (Real)"]
        if use_ip:
            active_props.append("Ip")
            prop_labels.append("Ip")
        if use_is:
            active_props.append("Is")
            prop_labels.append("Is")
        if use_vpvs:
            active_props.append("Vp/Vs")
            prop_labels.append("Vp/Vs")
        active_props.append("Wells")
        prop_labels.append("Wells")
        active_props.append("Seismic")
        prop_labels.append("Seismic")

        num_rows = len(active_props)

        # Determine channel indices dynamically
        ch_idx = num_facies_ch
        ip_ch_idx = -1
        is_ch_idx = -1
        vpvs_ch_idx = -1
        if use_ip:
            ip_ch_idx = ch_idx
            ch_idx += 1
        if use_is:
            is_ch_idx = ch_idx
            ch_idx += 1
        if use_vpvs:
            vpvs_ch_idx = ch_idx
            ch_idx += 1

        # Use squeeze=False to ensure axes is always 2D
        fig, axes = plt.subplots(  # type: ignore
            num_rows, num_scales, figsize=(3 * num_scales, 3 * num_rows), squeeze=False
        )
        fig.patch.set_facecolor("#151b26")

        for scale in range(num_scales):
            facies_batch, wells_batch, _, seismic_batch = dataset.get_scale_data(scale)
            if facies_batch.shape[0] == 0:
                continue

            idx = 0
            f = facies_batch[idx].cpu()
            w = wells_batch[idx].cpu() if wells_batch.shape[0] > idx else None
            s = seismic_batch[idx].cpu() if seismic_batch.shape[0] > idx else None

            num_channels = f.shape[0]

            def clean_axis(ax_obj: Any) -> None:
                ax_obj.set_xticks([])
                ax_obj.set_yticks([])
                ax_obj.set_facecolor("#111622")
                for spine in ax_obj.spines.values():
                    spine.set_visible(False)

            for row_idx, prop in enumerate(active_props):
                ax = axes[row_idx, scale]

                if prop == "Facies":
                    if num_facies_ch == 3:
                        f_idx = utils.rgb_to_facies(f[:3])
                    else:
                        f_idx = torch.argmax(f[:num_facies_ch], dim=0).numpy()
                    ax.imshow(utils.facies_to_rgb(f_idx).transpose(1, 2, 0))
                    h, w_facies = f_idx.shape[0], f_idx.shape[1]
                    ax.set_title(
                        f"Scale {scale}\n({w_facies}×{h})",
                        fontsize=10,
                        fontweight="bold",
                        color="#f0f4f9",
                    )

                elif prop == "Ip":
                    if ip_ch_idx != -1 and num_channels > ip_ch_idx:
                        ip = f[ip_ch_idx].numpy()
                        ax.imshow(ip, cmap="magma")

                elif prop == "Is":
                    if is_ch_idx != -1 and num_channels > is_ch_idx:
                        is_ = f[is_ch_idx].numpy()
                        ax.imshow(is_, cmap="magma")

                elif prop == "Vp/Vs":
                    if vpvs_ch_idx != -1 and num_channels > vpvs_ch_idx:
                        vpvs = f[vpvs_ch_idx].numpy()
                        ax.imshow(vpvs, cmap="viridis")

                elif prop == "Wells":
                    if w is not None:
                        well_cmap = mcolors.ListedColormap(PALETTE_RGB)
                        if w.shape[0] == 3:
                            w_idx = utils.rgb_to_facies(w).astype(float)
                        else:
                            w_idx = torch.argmax(w, dim=0).numpy().astype(float)

                        mask = (
                            (w.abs().sum(dim=0) < 1e-3)
                            if opt.normalization_range[0] == 0
                            else (w < opt.normalization_range[0] + 0.1).all(dim=0)
                        )
                        w_idx[mask] = np.nan
                        ax.imshow(w_idx, cmap=well_cmap, vmin=0, vmax=len(PALETTE_RGB) - 1)

                elif prop == "Seismic":
                    if s is not None:
                        s_np = s[0].numpy() if s.ndim == 3 else s.numpy()
                        s_plot = s_np - np.mean(s_np)
                        p_lo = float(np.percentile(s_plot, 2))
                        p_hi = float(np.percentile(s_plot, 98))
                        max_abs = max(abs(p_lo), abs(p_hi), 1e-6)
                        ax.imshow(s_plot, cmap="RdBu", vmin=-max_abs, vmax=max_abs)

                clean_axis(ax)

                if scale == 0:
                    ax.set_ylabel(
                        prop_labels[row_idx],
                        fontsize=12,
                        fontweight="bold",
                        rotation=90,
                        labelpad=15,
                        color="#f0f4f9",
                    )

        fig.suptitle(  # type: ignore
            "Ground Truth Training Pyramids — Multi-Resolution Dataset",
            fontsize=16,
            fontweight="bold",
            y=1.01,
            color="#f0f4f9",
        )
        plt.tight_layout()
        plt.savefig(  # type: ignore
            out_path,
            dpi=120,
            bbox_inches="tight",
            facecolor="#151b26",
        )
        plt.close()
        print(f"Generated training pyramid samples: {out_path}")
    except Exception as e:
        print(f"Error generating training pyramid visualization: {e}")


def ensure_rock_physics_crossplots(outputs_dir: Path, data_dir: Path) -> None:
    """Generate individual rock physics crossplots (Ip vs Is colored by facies) for Real vs Generated variants."""
    print("Generating individual rock physics crossplots (Ip vs Is)...")
    try:
        import matplotlib.pyplot as plt
        import torch

        import utils
        from datasets.dataset import PyramidsDataset
        from options import TrainingOptions

        plt = cast(Any, plt)

        # 1. Load Real Data
        base_options_path = outputs_dir / "wells_seismic" / "options.json"
        latest_params: dict[str, Any] = {}
        if base_options_path.exists():
            try:
                with open(base_options_path, encoding="utf-8") as f:
                    latest_params = json.load(f)
            except Exception:
                pass

        opt = TrainingOptions()
        for key, val in latest_params.items():
            if hasattr(opt, key):
                setattr(opt, key, val)
        use_ip = getattr(opt, "use_ip", True)
        use_is = getattr(opt, "use_is", True)
        if not (use_ip and use_is):
            print("Ip or Is disabled; skipping rock physics crossplots.")
            return

        dataset = PyramidsDataset(opt)
        num_scales = len(dataset.scales)
        num_facies_ch = opt.num_facies_channels

        # Load finest scale data
        facies_batch, _, _, _ = dataset.get_scale_data(num_scales - 1)
        if facies_batch.shape[0] == 0:
            return

        real_ips: list[NDArray[Any]] = []
        real_iss: list[NDArray[Any]] = []
        real_facies: list[NDArray[Any]] = []

        n_samples = min(10, facies_batch.shape[0])
        for idx in range(n_samples):
            f = facies_batch[idx].cpu()
            num_channels = f.shape[0]
            if num_channels > num_facies_ch + 1:
                # Facies Index
                if num_facies_ch == 3:
                    f_idx = utils.rgb_to_facies(f[:3])
                else:
                    f_idx = torch.argmax(f[:num_facies_ch], dim=0).numpy()

                ip = f[num_facies_ch].numpy()
                is_ = f[num_facies_ch + 1].numpy()

                real_facies.append(f_idx.flatten())
                real_ips.append(ip.flatten())
                real_iss.append(is_.flatten())

        if not real_ips:
            return

        real_facies_arr = np.concatenate(real_facies)
        real_ips_arr = np.concatenate(real_ips)
        real_iss_arr = np.concatenate(real_iss)

        well_colors = ["#00e5ff", "#ff2d9b", "#ffb300", "#76ff03"]
        facies_labels = ["Floodplain", "Point Bar", "Channel", "Boundary"]

        def save_scatter_plot(
            ip_data: NDArray[Any],
            is_data: NDArray[Any],
            facies_data: NDArray[Any],
            title: str,
            out_filename: str,
        ) -> None:
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            fig.patch.set_facecolor("#151b26")
            ax.set_facecolor("#111622")

            n_points = len(ip_data)
            if n_points > 2500:
                indices = np.random.choice(n_points, 2500, replace=False)
                ip_sub = ip_data[indices]
                is_sub = is_data[indices]
                facies_sub = facies_data[indices]
            else:
                ip_sub = ip_data
                is_sub = is_data
                facies_sub = facies_data

            for facies_class in range(4):
                mask = facies_sub == facies_class
                if np.any(mask):
                    ax.scatter(
                        ip_sub[mask],
                        is_sub[mask],
                        c=well_colors[facies_class],
                        label=facies_labels[facies_class],
                        alpha=0.6,
                        edgecolors="none",
                        s=18,
                    )
            ax.set_title(title, fontsize=12, fontweight="bold", color="#f0f4f9")
            ax.grid(True, alpha=0.2, color="#8c9eb5")
            ax.set_xlabel("Ip (m/s * g/cm³)", fontsize=10, color="#8c9eb5")
            ax.set_ylabel("Is (m/s * g/cm³)", fontsize=10, color="#8c9eb5")
            ax.tick_params(axis="both", which="major", labelsize=9, colors="#8c9eb5")
            for spine in ax.spines.values():
                spine.set_color("#222d41")

            ax.legend(
                loc="upper right",
                fontsize=8,
                frameon=True,
                facecolor="#151b26",
                edgecolor="#222d41",
                labelcolor="#f0f4f9",
            )

            out_path = outputs_dir / out_filename
            plt.tight_layout()
            plt.savefig(  # type: ignore
                out_path,
                dpi=120,
                bbox_inches="tight",
                facecolor="#151b26",
            )
            plt.close()
            print(f"Generated individual rock physics plot: {out_path}")

        # Plot Real Data
        save_scatter_plot(
            real_ips_arr,
            real_iss_arr,
            real_facies_arr,
            "Real Validation Data",
            "rock_physics_real.png",
        )

        # Plot Generated Data for each variant
        variants_to_plot = [
            ("wells_seismic", "wells_seismic", "rock_physics_wells_seismic.png"),
            ("wells_only", "wells_only", "rock_physics_wells_only.png"),
            ("seismic_only", "seismic_only", "rock_physics_seismic_only.png"),
            ("unconditional", "unconditional", "rock_physics_unconditional.png"),
        ]

        for variant, _label_key, out_file in variants_to_plot:
            var_dir = outputs_dir / variant / "generated"
            facies_dir = var_dir / "facies"
            ip_dir = var_dir / "ip"
            is_dir = var_dir / "is"

            gen_ips: list[NDArray[Any]] = []
            gen_iss: list[NDArray[Any]] = []
            gen_facies: list[NDArray[Any]] = []

            if facies_dir.exists() and ip_dir.exists() and is_dir.exists():
                npy_facies = sorted(list(facies_dir.glob("*.npy")))
                npy_ip = sorted(list(ip_dir.glob("*.npy")))
                npy_is = sorted(list(is_dir.glob("*.npy")))

                n_samples_gen = min(20, len(npy_facies), len(npy_ip), len(npy_is))
                for idx in range(n_samples_gen):
                    try:
                        f_idx = np.load(npy_facies[idx])
                        ip = np.load(npy_ip[idx])
                        is_ = np.load(npy_is[idx])

                        gen_facies.append(f_idx.flatten())
                        gen_ips.append(ip.flatten())
                        gen_iss.append(is_.flatten())
                    except Exception:
                        pass

            if gen_ips:
                gen_facies_arr = np.concatenate(gen_facies)
                gen_ips_arr = np.concatenate(gen_ips)
                gen_iss_arr = np.concatenate(gen_iss)
                save_scatter_plot(
                    gen_ips_arr,
                    gen_iss_arr,
                    gen_facies_arr,
                    VARIANT_LABELS[variant],
                    out_file,
                )
            else:
                # Save placeholder scatter plot if not found
                fig, ax = plt.subplots(figsize=(6.5, 4.2))
                fig.patch.set_facecolor("#151b26")
                ax.set_facecolor("#111622")
                ax.text(
                    0.5,
                    0.5,
                    "Data Not Available",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="#8c9eb5",
                    fontsize=12,
                )
                ax.set_title(
                    VARIANT_LABELS[variant],
                    fontsize=12,
                    fontweight="bold",
                    color="#f0f4f9",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color("#222d41")
                out_path = outputs_dir / out_file
                plt.tight_layout()
                plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="#151b26")
                plt.close()
                print(f"Generated placeholder rock physics plot: {out_path}")

    except Exception as e:
        print(f"Error generating rock physics crossplots: {e}")


def ensure_distribution_histograms(outputs_dir: Path, data_dir: Path) -> None:
    """Generate overlaid histogram plots comparing Real vs Generated distributions for Ip, Is, Vp/Vs, Seismic and Facies."""
    print("Generating property distribution histograms...")
    try:
        import matplotlib.pyplot as plt
        import torch

        import utils
        from datasets.dataset import PyramidsDataset
        from options import TrainingOptions

        plt = cast(Any, plt)

        # 1. Load Real Data
        base_options_path = outputs_dir / "wells_seismic" / "options.json"
        latest_params: dict[str, Any] = {}
        if base_options_path.exists():
            try:
                with open(base_options_path, encoding="utf-8") as f:
                    latest_params = json.load(f)
            except Exception:
                pass

        opt = TrainingOptions()
        for key, val in latest_params.items():
            if hasattr(opt, key):
                setattr(opt, key, val)
        opt.input_path = str(data_dir)
        opt.use_rock_physics = True
        opt.use_seismic = True
        opt.use_wells = True

        dataset = PyramidsDataset(opt)
        num_scales = len(dataset.scales)
        num_facies_ch = opt.num_facies_channels

        facies_batch, _, _, seismic_batch = dataset.get_scale_data(num_scales - 1)
        if facies_batch.shape[0] == 0:
            return

        denormalize = get_denormalize_fn(data_dir)
        denormalize_any = cast(Any, denormalize)

        active_properties: list[str] = []
        if getattr(opt, "use_ip", True):
            active_properties.append("Ip")
        if getattr(opt, "use_is", True):
            active_properties.append("Is")
        if getattr(opt, "use_vpvs", True):
            active_properties.append("VP_VS")

        real_ips: list[NDArray[Any]] = []
        real_iss: list[NDArray[Any]] = []
        real_vpvs: list[NDArray[Any]] = []
        real_seismic: list[NDArray[Any]] = []
        real_facies: list[NDArray[Any]] = []

        prop_indices = {name: i for i, name in enumerate(active_properties)}

        n_samples = min(10, facies_batch.shape[0])
        for idx in range(n_samples):
            f = facies_batch[idx].cpu()
            num_channels = f.shape[0]

            if num_facies_ch == 3:
                f_idx = utils.rgb_to_facies(f[:3]).flatten()
            else:
                f_idx = torch.argmax(f[:num_facies_ch], dim=0).numpy().flatten()
            real_facies.append(f_idx)

            if "Ip" in prop_indices:
                ch_idx = num_facies_ch + prop_indices["Ip"]
                if num_channels > ch_idx:
                    ip = f[ch_idx].numpy().flatten()
                    ip = np.asarray(denormalize_any(ip, "ip", opt))
                    real_ips.append(ip)

            if "Is" in prop_indices:
                ch_idx = num_facies_ch + prop_indices["Is"]
                if num_channels > ch_idx:
                    is_ = f[ch_idx].numpy().flatten()
                    is_ = np.asarray(denormalize_any(is_, "is", opt))
                    real_iss.append(is_)

            if "VP_VS" in prop_indices:
                ch_idx = num_facies_ch + prop_indices["VP_VS"]
                if num_channels > ch_idx:
                    vpvs = f[ch_idx].numpy().flatten()
                    vpvs = np.asarray(denormalize_any(vpvs, "vpvs", opt))
                    real_vpvs.append(vpvs)

            if getattr(opt, "use_ip", True) and seismic_batch.shape[0] > idx:
                s = seismic_batch[idx].cpu().numpy().flatten()
                s = np.asarray(denormalize_any(s, "seismic", opt))
                real_seismic.append(s)

        real_ip_arr = np.concatenate(real_ips) if real_ips else None
        real_is_arr = np.concatenate(real_iss) if real_iss else None
        real_vpvs_arr = np.concatenate(real_vpvs) if real_vpvs else None
        real_seismic_arr = np.concatenate(real_seismic) if real_seismic else None
        real_facies_arr = np.concatenate(real_facies)

        variant_colors = {
            "wells_seismic": "#00e5ff",
            "wells_only": "#ff2d9b",
            "seismic_only": "#ffb300",
            "unconditional": "#76ff03",
        }

        properties = [
            ("facies", "Facies Categories", real_facies_arr),
        ]
        if getattr(opt, "use_ip", True) and real_ip_arr is not None:
            properties.append(("ip", "Acoustic Impedance (Ip)", real_ip_arr))
        if getattr(opt, "use_is", True) and real_is_arr is not None:
            properties.append(("is", "Shear Impedance (Is)", real_is_arr))
        if getattr(opt, "use_vpvs", True) and real_vpvs_arr is not None:
            properties.append(("vpvs", "Vp/Vs Ratio", real_vpvs_arr))
        if getattr(opt, "use_ip", True) and real_seismic_arr is not None:
            properties.append(
                ("seismic", "Synthetic Seismic Amplitude", real_seismic_arr)
            )

        facies_labels = ["Floodplain", "Point Bar", "Channel", "Boundary"]

        for prop_key, prop_label, real_data in properties:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            fig.patch.set_facecolor("#151b26")
            ax.set_facecolor("#111622")

            is_facies = prop_key == "facies"
            is_seismic = prop_key == "seismic"

            if is_seismic:
                real_data = (real_data - np.mean(real_data)) / (
                    np.std(real_data) + 1e-8
                )
                prop_label = f"{prop_label} (Standardized)"

            bins: int | list[float]
            if is_facies:
                bins = (np.arange(5, dtype=float) - 0.5).tolist()
            else:
                bins = 80

            ax.hist(
                real_data,
                bins=bins,
                density=True,
                alpha=0.22,
                color="#f0f4f9",
                edgecolor="none",
                label="Real Data",
                zorder=2,
                rwidth=0.85 if is_facies else 1.0,
            )

            for variant in VARIANTS:
                gen_dir = outputs_dir / variant / "generated" / prop_key
                if prop_key == "vpvs":
                    gen_dir = outputs_dir / variant / "generated" / "vp_vs"
                if not gen_dir.exists():
                    continue
                npy_files = sorted(list(gen_dir.glob("*.npy")))[:200]
                if not npy_files:
                    continue

                all_gen_vals: list[NDArray[Any]] = []
                for f in npy_files:
                    v: NDArray[Any] = np.load(f).flatten()
                    if is_seismic:
                        v = (v - np.mean(v)) / (np.std(v) + 1e-8)
                    all_gen_vals.append(v)

                gen_vals = np.concatenate(all_gen_vals)

                ax.hist(
                    gen_vals,
                    bins=bins,
                    density=True,
                    alpha=0.07 if not is_facies else 0.12,
                    color=variant_colors[variant],
                    histtype="stepfilled" if not is_facies else "bar",
                    linewidth=0,
                    rwidth=0.65 if is_facies else 1.0,
                    zorder=3,
                )
                ax.hist(
                    gen_vals,
                    bins=bins,
                    density=True,
                    alpha=1.0,
                    color=variant_colors[variant],
                    label=VARIANT_LABELS[variant],
                    histtype="step",
                    linewidth=2.2,
                    rwidth=0.65 if is_facies else 1.0,
                    zorder=10,
                )

            ax.set_title(
                f"Distribution: {prop_label}",
                fontsize=13,
                fontweight="bold",
                color="#f0f4f9",
            )
            ax.set_xlabel(prop_label, fontsize=10, color="#8c9eb5")
            ax.set_ylabel("Density", fontsize=10, color="#8c9eb5")

            if is_facies:
                ax.set_xticks(range(4))
                ax.set_xticklabels(facies_labels)

            ax.tick_params(axis="both", which="major", labelsize=9, colors="#8c9eb5")
            ax.grid(True, alpha=0.15, color="#8c9eb5")
            for spine in ax.spines.values():
                spine.set_color("#222d41")
            ax.legend(
                loc="upper right",
                fontsize=8,
                frameon=True,
                facecolor="#151b26",
                edgecolor="#222d41",
                labelcolor="#f0f4f9",
            )

            out_path = outputs_dir / f"distribution_hist_{prop_key}.png"
            plt.tight_layout()
            plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="#151b26")
            plt.close()
            print(f"Generated distribution histogram: {out_path}")

    except Exception as e:
        print(f"Error generating distribution histograms: {e}")


def ensure_variogram_plots(outputs_dir: Path, data_dir: Path) -> None:
    """Compute and plot omnidirectional + horizontal experimental variograms for Ip and Is."""
    print("Generating variogram plots...")
    try:
        import matplotlib.pyplot as plt

        plt = cast(Any, plt)

        # Load options to see what is enabled
        base_options_path = outputs_dir / "wells_seismic" / "options.json"
        use_ip, use_is, use_vpvs = True, True, True
        if base_options_path.exists():
            try:
                with open(base_options_path, encoding="utf-8") as f:
                    opt_dict = json.load(f)
                    use_ip = opt_dict.get("use_ip", True)
                    use_is = opt_dict.get("use_is", True)
                    use_vpvs = opt_dict.get("use_vpvs", True)
            except Exception:
                pass

        def compute_variogram(
            field: NDArray[Any], max_lag: int = 30, direction: str = "omni"
        ) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
            _h, _w = field.shape
            lags = np.arange(1, max_lag + 1, dtype=np.int_)
            gamma = np.zeros(len(lags), dtype=np.float64)
            counts = np.zeros(len(lags), dtype=np.float64)

            for i, lag in enumerate(lags):
                if direction in ("omni", "horizontal"):
                    diff_h = field[:, lag:] - field[:, :-lag]
                    gamma[i] += np.sum(diff_h**2)
                    counts[i] += diff_h.size
                if direction in ("omni", "vertical"):
                    diff_v = field[lag:, :] - field[:-lag, :]
                    gamma[i] += np.sum(diff_v**2)
                    counts[i] += diff_v.size

            valid = counts > 0
            gamma[valid] = gamma[valid] / (2.0 * counts[valid])
            return lags, gamma

        variant_colors = {
            "Real": "#f0f4f9",
            "wells_seismic": "#00e5ff",
            "wells_only": "#ff2d9b",
            "seismic_only": "#ffb300",
            "unconditional": "#76ff03",
        }

        properties_list = [
            ("facies", "Facies Categories"),
        ]
        if use_ip:
            properties_list.append(("ip", "Acoustic Impedance (Ip)"))
        if use_is:
            properties_list.append(("is", "Shear Impedance (Is)"))
        if use_vpvs:
            properties_list.append(("vp_vs", "Vp/Vs Ratio"))
        if use_ip:  # seismic requires Ip
            properties_list.append(("seismic", "Synthetic Seismic"))

        for prop_key, prop_label in properties_list:
            # Grid of 4 variants (rows) x 2 directions (columns)
            fig, axes = plt.subplots(4, 2, figsize=(14, 18))
            fig.patch.set_facecolor("#151b26")

            # 1. Precompute Real Variograms (Ground Truth)
            real_gammas: dict[str, NDArray[np.float64]] = {}
            real_lags: NDArray[np.int_] = np.array([], dtype=np.int_)
            try:
                import torch
                import utils
                from datasets.dataset import PyramidsDataset
                from options import TrainingOptions

                base_options_path = outputs_dir / "wells_seismic" / "options.json"
                latest_params: dict[str, Any] = {}
                if base_options_path.exists():
                    with open(base_options_path, encoding="utf-8") as f_opts:
                        latest_params = json.load(f_opts)
                opt = TrainingOptions()
                for key, val in latest_params.items():
                    if hasattr(opt, key):
                        setattr(opt, key, val)
                opt.input_path = str(data_dir)
                opt.use_rock_physics = True
                opt.use_seismic = True
                opt.use_wells = True
                dataset = PyramidsDataset(opt)
                num_scales = len(dataset.scales)
                num_facies_ch = opt.num_facies_channels
                (
                    facies_batch,
                    _,
                    _,
                    seismic_batch,
                ) = dataset.get_scale_data(num_scales - 1)

                denormalize = get_denormalize_fn(data_dir)
                denormalize_any = cast(Any, denormalize)

                for direction in ["omni", "horizontal"]:
                    if prop_key == "seismic":
                        data_source = seismic_batch
                        ch_idx = 0
                    else:
                        data_source = facies_batch
                        active_properties = []
                        if getattr(opt, "use_ip", True):
                            active_properties.append("Ip")
                        if getattr(opt, "use_is", True):
                            active_properties.append("Is")
                        if getattr(opt, "use_vpvs", True):
                            active_properties.append("VP_VS")
                        prop_indices = {
                            name: idx
                            for idx, name in enumerate(active_properties)
                        }

                        if prop_key == "ip" and "Ip" in prop_indices:
                            ch_idx = num_facies_ch + prop_indices["Ip"]
                        elif prop_key == "is" and "Is" in prop_indices:
                            ch_idx = num_facies_ch + prop_indices["Is"]
                        elif prop_key == "vp_vs" and "VP_VS" in prop_indices:
                            ch_idx = num_facies_ch + prop_indices["VP_VS"]
                        else:
                            ch_idx = -1

                    all_gammas: list[NDArray[np.float64]] = []
                    for s_idx in range(min(10, data_source.shape[0])):
                        if prop_key == "facies":
                            f = data_source[s_idx].cpu()
                            if num_facies_ch == 3:
                                field = utils.rgb_to_facies(f[:3]).astype(float)
                            else:
                                field = (
                                    torch.argmax(f[:num_facies_ch], dim=0)
                                    .numpy()
                                    .astype(float)
                                )
                        else:
                            if ch_idx == -1:
                                continue
                            field = data_source[s_idx, ch_idx].cpu().numpy()
                            field = np.asarray(denormalize_any(field, prop_key, opt))

                        if prop_key == "seismic":
                            field = (field - np.mean(field)) / (np.std(field) + 1e-8)

                        real_lags, g = compute_variogram(
                            field, max_lag=30, direction=direction
                        )
                        var_val = np.var(field)
                        if var_val > 1e-8:
                            g = g / var_val
                        all_gammas.append(g)
                    if all_gammas:
                        real_gammas[direction] = np.mean(all_gammas, axis=0)
            except Exception as e:
                print(f"Error computing Real variogram reference for {prop_key}: {e}")

            # 2. Plot variants in a grid
            for v_idx, variant_name in enumerate(VARIANTS):
                for d_idx, (direction, dir_label) in enumerate(
                    [("omni", "Omnidirectional"), ("horizontal", "Horizontal")]
                ):
                    ax = axes[v_idx, d_idx]
                    ax.set_facecolor("#111622")

                    # Plot Real background reference
                    if direction in real_gammas:
                        ax.plot(
                            real_lags,
                            real_gammas[direction],
                            color="#f0f4f9",
                            linestyle="--",
                            linewidth=2.0,
                            label="Real Data (Reference)",
                            alpha=0.4,
                            zorder=1,
                        )

                    # Compute and plot Variant
                    v_dir_key = "vp_vs" if prop_key == "vp_vs" else prop_key
                    gen_dir = outputs_dir / variant_name / "generated" / v_dir_key
                    if gen_dir.exists():
                        npy_files = sorted(list(gen_dir.glob("*.npy")))[:50]
                        if npy_files:
                            all_gammas = []
                            v_lags = np.array([], dtype=np.int_)
                            for nf in npy_files:
                                field = np.asarray(np.load(nf))
                                if field.ndim == 3:
                                    field = field[0]

                                if prop_key == "seismic":
                                    field = (field - np.mean(field)) / (
                                        np.std(field) + 1e-8
                                    )

                                v_lags, g = compute_variogram(
                                    field, max_lag=30, direction=direction
                                )
                                var_val = np.var(field)
                                if var_val > 1e-8:
                                    g = g / var_val
                                all_gammas.append(g)

                            if all_gammas:
                                mean_gamma = np.mean(all_gammas, axis=0)
                                ax.plot(
                                    v_lags,
                                    mean_gamma,
                                    color=variant_colors[variant_name],
                                    linewidth=2.5,
                                    label=f"Variant: {VARIANT_LABELS[variant_name]}",
                                    alpha=1.0,
                                    zorder=5,
                                )

                    # Subplot styling
                    if v_idx == 0:
                        ax.set_title(
                            f"{dir_label}",
                            fontsize=12,
                            fontweight="bold",
                            color="#f0f4f9",
                            pad=10,
                        )
                    if d_idx == 0:
                        ax.set_ylabel(
                            f"{VARIANT_LABELS[variant_name]}\n\nγ(h) / Variance",
                            fontsize=10,
                            fontweight="bold",
                            color=variant_colors[variant_name],
                        )
                    else:
                        ax.set_ylabel("γ(h) / Variance", fontsize=9, color="#8c9eb5")

                    if v_idx == 3:
                        ax.set_xlabel("Lag (pixels)", fontsize=10, color="#8c9eb5")

                    ax.tick_params(
                        axis="both", which="major", labelsize=8, colors="#8c9eb5"
                    )
                    ax.grid(True, alpha=0.1, color="#8c9eb5", linestyle=":")
                    for spine in ax.spines.values():
                        spine.set_color("#222d41")

                    ax.legend(
                        loc="lower right",
                        fontsize=7,
                        frameon=True,
                        facecolor="#151b26",
                        edgecolor="#222d41",
                        labelcolor="#f0f4f9",
                    )

            y_label_extra = " (Standardized)" if prop_key == "seismic" else ""
            fig.suptitle(
                f"Spatial Continuity Analysis: {prop_label}{y_label_extra}\nComparison of Experimental Variograms vs. Real Data",
                fontsize=16,
                fontweight="bold",
                color="#f0f4f9",
                y=0.985,
            )
            out_path = outputs_dir / f"variogram_{prop_key}.png"
            plt.tight_layout(rect=(0.0, 0.02, 1.0, 0.965))
            plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="#151b26")
            plt.close()
            print(f"Generated variogram grid plot: {out_path}")

    except Exception as e:
        print(f"Error generating variogram plots: {e}")


def ensure_loss_plots(outputs_dir: Path) -> None:
    """Extract scalars from TensorBoard logs and generate loss plots."""
    try:
        import matplotlib.pyplot as plt
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )  # type: ignore[reportMissingTypeStubs]
    except ImportError:
        return
    plt = cast(Any, plt)

    for variant in VARIANTS:
        variant_dir = outputs_dir / variant
        log_dir = variant_dir / "tensorboard_logs"
        if not log_dir.exists():
            continue

        print(f"Generating smoothed loss plots for {variant}...")
        try:
            ea = cast(Any, EventAccumulator(str(log_dir)))
            ea.Reload()

            tags = cast(list[str], ea.Tags().get("scalars", []))
            loss_tags = sorted(
                [t for t in tags if t.startswith("Mean/G_") or t.startswith("Mean/D_")]
            )

            loss_colors = [
                "#3867d6",
                "#20bf6b",
                "#eb3b5a",
                "#fa8231",
                "#8854d0",
                "#0fb9b1",
                "#f7b731",
                "#fd9644",
                "#fc5c65",
                "#2bcbba",
                "#a55eed",
                "#fed330",
            ]

            for idx, tag in enumerate(loss_tags):
                events = cast(list[Any], ea.Scalars(tag))
                steps = [int(getattr(e, "step", 0)) for e in events]
                values = [float(getattr(e, "value", 0.0)) for e in events]
                smoothed = smooth(values, weight=0.85)

                clean_name = tag.replace("Mean/", "").replace("/", "_")
                out_path = variant_dir / f"loss_{clean_name}.png"

                fig, ax = plt.subplots(figsize=(6.5, 3.8))
                fig.patch.set_facecolor("#151b26")
                ax.set_facecolor("#111622")

                color = loss_colors[idx % len(loss_colors)]
                ax.plot(steps, values, alpha=0.15, color=color)
                ax.plot(steps, smoothed, color=color, linewidth=2.0)

                title = clean_name.replace("_", " ")
                ax.set_title(title, fontsize=11, fontweight="bold", color="#f0f4f9")
                ax.grid(True, alpha=0.2, color="#8c9eb5")

                for spine in ax.spines.values():
                    spine.set_color("#222d41")

                ax.tick_params(
                    axis="both", which="major", labelsize=9, colors="#8c9eb5"
                )
                ax.set_xlabel("Epoch", fontsize=9, color="#8c9eb5")  # type: ignore[reportUnknownMemberType]
                ax.set_ylabel("Loss Value", fontsize=9, color="#8c9eb5")  # type: ignore[reportUnknownMemberType]

                plt.tight_layout()
                plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="#151b26")  # type: ignore[reportUnknownMemberType]
                plt.close()
        except Exception as e:
            print(f"Error generating loss plots for {variant}: {e}")
