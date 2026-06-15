import json
import re
from pathlib import Path
from typing import Any, Callable, cast

from jinja2 import Environment, FileSystemLoader

import report.metrics as report_metrics
from report.constants import (
    EMBEDDING_LABELS,
    EMBEDDING_METHODS,
    HPARAM_CATEGORIES,
    VARIANT_LABELS,
    VARIANTS,
)
from report.utils import (
    ensure_all_report_images,
    format_value,
    get_milestones,
    load_configs,
    md_relpath,
)

compute_quantitative_results_fn = cast(
    Callable[[Path, Path], dict[str, Any]],
    getattr(report_metrics, "compute_quantitative_results"),
)
compute_channel_connectivity_fn = cast(
    Callable[[Path, Path], dict[str, dict[str, float]]],
    getattr(report_metrics, "compute_channel_connectivity"),
)
compute_distribution_metrics_fn = cast(
    Callable[[Path, Path], dict[str, Any]],
    getattr(report_metrics, "compute_distribution_metrics"),
)
get_performance_data_fn = cast(
    Callable[[Path], dict[str, dict[str, float]]],
    getattr(report_metrics, "get_performance_data"),
)


def generate_html_report(
    outputs_dir: Path | None = None,
    data_dir: Path | None = None,
    output: str | None = None,
) -> str:
    if outputs_dir is None:
        outputs_dir = Path("outputs/experiments")
    else:
        outputs_dir = Path(outputs_dir)

    if data_dir is None:
        data_dir = Path("data")
    else:
        data_dir = Path(data_dir)

    ensure_all_report_images(outputs_dir, data_dir)

    # Compute new metrics
    channel_connectivity: dict[str, dict[str, float]] = compute_channel_connectivity_fn(
        outputs_dir, data_dir
    )
    distribution_metrics: dict[str, Any] = compute_distribution_metrics_fn(
        outputs_dir, data_dir
    )
    performance_data: dict[str, dict[str, float]] = get_performance_data_fn(outputs_dir)

    if output is None:
        output = str(outputs_dir.parent.parent / "index.html")

    print(f"Generating HTML report to: {output}")
    report_dir = Path(output).parent

    # Gathers same data as markdown report
    configs = load_configs(outputs_dir)

    quant: dict[str, Any] = compute_quantitative_results_fn(outputs_dir, data_dir)
    results: dict[str, dict[str, Any]] = cast(
        dict[str, dict[str, Any]], quant["results"]
    )
    real_facies_proportions: list[float] = cast(
        list[float], quant["real_facies_proportions"]
    )
    real_ip_mean = float(quant["real_ip_mean"])
    real_ip_std = float(quant["real_ip_std"])
    real_is_mean = float(quant["real_is_mean"])
    real_is_std = float(quant["real_is_std"])
    real_vpvs_mean = float(quant["real_vpvs_mean"])
    real_vpvs_std = float(quant["real_vpvs_std"])
    real_seismic_mean = float(quant["real_seismic_mean"])
    real_seismic_std = float(quant["real_seismic_std"])

    lowest_rmse_var: str | None = None
    min_rmse = 999.0
    for var in VARIANTS:
        if var in results:
            rmse_val = float(results[var]["rmse_error"])
            if rmse_val < min_rmse:
                min_rmse = rmse_val
                lowest_rmse_var = var

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

    # Compute robust variant scorecard
    scorecard: dict[str, dict[str, float]] = {}
    for var in VARIANTS:
        if var not in results:
            continue
        r = results[var]
        # Calculate deviations for the scorecard
        ip_dev = (
            abs(((r["ip_mean"] - real_ip_mean) / real_ip_mean) * 100) if use_ip else 0.0
        )
        is_dev = (
            abs(((r["is_mean"] - real_is_mean) / real_is_mean) * 100) if use_is else 0.0
        )
        vpvs_dev = (
            abs(((r["vpvs_mean"] - real_vpvs_mean) / real_vpvs_mean) * 100)
            if use_vpvs
            else 0.0
        )
        facies_rmse = float(r["rmse_error"]) * 100

        # Channel Connectivity Deviation (largest connected component fraction difference)
        real_conn = channel_connectivity.get("Real", {}).get("largest_frac", 1.0)
        var_conn = channel_connectivity.get(var, {}).get("largest_frac", 0.0)
        conn_dev = abs(real_conn - var_conn) * 100

        # Spatial Conditioning Fidelity (SSIM deviation from 1.0)
        # For unconditional, avg_ssim is 0, giving a 100% ssim_dev penalty.
        ssim_val = float(r.get("avg_ssim", 0.0))
        ssim_dev = (1.0 - ssim_val) * 100

        # Weighted score (lower is better, each of the 4 pillars has a 25% weight)
        active_rp_count = sum([use_ip, use_is, use_vpvs])
        if active_rp_count > 0:
            rp_dev = (
                (ip_dev if use_ip else 0.0)
                + (is_dev if use_is else 0.0)
                + (vpvs_dev if use_vpvs else 0.0)
            ) / active_rp_count
        else:
            rp_dev = 0.0

        overall = (
            (facies_rmse * 0.25)
            + (rp_dev * 0.25)
            + (conn_dev * 0.25)
            + (ssim_dev * 0.25)
        )

        scorecard[var] = {
            "facies_rmse": facies_rmse,
            "ip_dev": ip_dev,
            "is_dev": is_dev,
            "vpvs_dev": vpvs_dev,
            "conn_dev": conn_dev,
            "ssim_dev": ssim_dev,
            "overall": overall,
        }

    # Sort variants by overall score
    sorted_variants: list[str] = sorted(
        scorecard.keys(), key=lambda v: scorecard[v]["overall"]
    )
    rank_map: dict[str, int] = {var: i + 1 for i, var in enumerate(sorted_variants)}
    medal_map = {1: "🥇", 2: "🥈", 3: "🥉"}

    # 1. Seismic images
    seismic_dir = data_dir / "seismic"
    seismic_files = (
        sorted(seismic_dir.glob("xz_crossline_*.png")) if seismic_dir.exists() else []
    )
    seismic_images: list[dict[str, str]] = []
    if seismic_files:
        indices = [
            int(i * (len(seismic_files) - 1) / 5)
            for i in range(min(5, len(seismic_files)))
        ]
        for i in indices:
            img_path = seismic_files[i]
            rel = md_relpath(img_path, report_dir)
            seismic_images.append({"name": img_path.stem, "rel_path": rel})

    # 2. Well images
    well_dir = data_dir / "wells_visualizations"
    well_files = (
        sorted(well_dir.glob("well_xz_crossline_*.png")) if well_dir.exists() else []
    )
    well_images: list[dict[str, str]] = []
    if well_files:
        indices = [
            int(i * (len(well_files) - 1) / 5) for i in range(min(5, len(well_files)))
        ]
        for i in indices:
            img_path = well_files[i]
            rel = md_relpath(img_path, report_dir)
            well_images.append(
                {"name": img_path.stem.replace("well_", ""), "rel_path": rel}
            )

    # 3. Available training pyramids
    available_pyramids: list[dict[str, str]] = []
    gt_path = outputs_dir / "training_pyramid_samples.png"
    if gt_path.exists():
        available_pyramids.append(
            {
                "label": "Ground Truth (Real)",
                "rel_path": md_relpath(gt_path, report_dir),
                "description": "Dataset overview showing actual real samples at each scale.",
            }
        )
    for variant in VARIANTS:
        pyramid_path = outputs_dir / variant / "pyramid_overview.png"
        if pyramid_path.exists():
            label = VARIANT_LABELS[variant]
            available_pyramids.append(
                {
                    "label": label,
                    "rel_path": md_relpath(pyramid_path, report_dir),
                    "description": f"Composite visualization showing generated realizations at all scales for the {label.lower()} variant.",
                }
            )

    # 4. Hyperparameters Table Setup
    hparams_table: list[dict[str, Any]] = []
    variant_labels_cols = [VARIANT_LABELS[v] for v in VARIANTS if v in configs]
    num_variant_cols = 1 + len(variant_labels_cols)
    if configs:
        for cat_name, params in HPARAM_CATEGORIES.items():
            row_params: list[dict[str, Any]] = []
            
            # Categories where we want to hide "disabled" (0.0) parameters
            hide_if_disabled_categories = [
                "⚖️ Loss & Consistency Penalties",
                "🌐 Physics Properties",
            ]
            should_filter = cat_name in hide_if_disabled_categories

            for key, label in params:
                # Check if this parameter is "active" in at least one variant
                is_active = False
                variant_vals: list[dict[str, Any]] = []
                
                for v in VARIANTS:
                    if v in configs:
                        val = configs[v].get(key)
                        
                        # Parameter activity logic
                        if val not in (None, 0, 0.0, False, "—"):
                            is_active = True
                            
                        # If filtering is active for this row, and value is disabled, show empty/dash
                        if should_filter and val in (None, 0, 0.0, False):
                            formatted = "—"
                        else:
                            formatted = format_value(val) if val is not None else "—"

                        variant_vals.append(
                            {
                                "formatted": formatted,
                                "is_yes": formatted == "Yes",
                                "is_no": formatted == "No",
                            }
                        )
                
                # Only add row if it is not filtered or if it is active in at least one variant
                if not should_filter or is_active:
                    row_params.append({"label": label, "values": variant_vals})
            
            if row_params:
                hparams_table.append({"category": cat_name, "params": row_params})

    # 5. Realization Comparison Grids
    grids = [
        ("facies", "Facies (Categorical)", "facies_comparison_all_variants.png"),
        ("ip", "Acoustic Impedance (Ip)", "ip_comparison_all_variants.png"),
        ("is", "Shear Impedance (Is)", "is_comparison_all_variants.png"),
        ("vp_vs", "Vp/Vs Ratio", "vp_vs_comparison_all_variants.png"),
        (
            "seismic",
            "Synthetic Seismic Amplitude",
            "seismic_comparison_all_variants.png",
        ),
    ]
    comparison_grids: list[dict[str, str]] = []
    for key, label, filename in grids:
        img_path = outputs_dir / filename
        if img_path.exists():
            comparison_grids.append(
                {
                    "key": key,
                    "label": label,
                    "rel_path": md_relpath(img_path, report_dir),
                }
            )

    # 6. Quantitative Analysis Formatting

    # Facies Proportions
    facies_proportions: dict[str, Any] = {
        "real": [f"{v * 100:.2f}%" for v in real_facies_proportions],
        "variants": [],
    }
    for var in VARIANTS:
        if var in results:
            p = cast(list[float], results[var]["proportions"])
            rmse_val = float(results[var]["rmse_error"])
            is_best = var == lowest_rmse_var
            facies_proportions["variants"].append(
                {
                    "label": VARIANT_LABELS[var],
                    "proportions": [f"{v * 100:.2f}%" for v in p],
                    "rmse": f"{rmse_val * 100:.2f}%",
                    "is_best": is_best,
                }
            )

    # Rock Physics Consistency
    def format_dev_vpvs(val: float, dev: float) -> str:
        sign = "+" if dev >= 0 else ""
        color = (
            "#34d399"
            if abs(dev) < 5.0
            else ("#fbbf24" if abs(dev) < 15.0 else "#f87171")
        )
        if abs(val) < 1e-4:
            val = 0.0
        val_str = f"{val:.3f}"
        return f"<span style='font-family: monospace; font-size: 0.9rem;'>{val_str}</span> <span style='font-size: 0.75rem; color: {color}; font-weight: 500;'>({sign}{dev:+.2f}%)</span>"

    def format_dev_other(val: float, dev: float) -> str:
        sign = "+" if dev >= 0 else ""
        color = (
            "#34d399"
            if abs(dev) < 5.0
            else ("#fbbf24" if abs(dev) < 15.0 else "#f87171")
        )
        val_str = f"{val:.1f}"
        return f"<span style='font-family: monospace; font-size: 0.9rem;'>{val_str}</span> <span style='font-size: 0.75rem; color: {color}; font-weight: 500;'>({sign}{dev:+.1f}%)</span>"

    rock_physics_stats: dict[str, Any] = {
        "real": {
            "ip_mean": f"{real_ip_mean:.1f}",
            "ip_std": f"{real_ip_std:.1f}",
            "is_mean": f"{real_is_mean:.1f}",
            "is_std": f"{real_is_std:.1f}",
            "vpvs_mean": f"{real_vpvs_mean:.3f}",
            "vpvs_std": f"{real_vpvs_std:.3f}",
            "seismic_mean": f"{real_seismic_mean:.3f}",
            "seismic_std": f"{real_seismic_std:.3f}",
        },
        "variants": [],
    }
    for var in VARIANTS:
        if var in results:
            r = results[var]
            ip_mean_dev = ((r["ip_mean"] - real_ip_mean) / (abs(real_ip_mean) + 1e-8)) * 100
            ip_std_dev = ((r["ip_std"] - real_ip_std) / (abs(real_ip_std) + 1e-8)) * 100
            is_mean_dev = ((r["is_mean"] - real_is_mean) / (abs(real_is_mean) + 1e-8)) * 100
            is_std_dev = ((r["is_std"] - real_is_std) / (abs(real_is_std) + 1e-8)) * 100
            vpvs_mean_dev = ((r["vpvs_mean"] - real_vpvs_mean) / (abs(real_vpvs_mean) + 1e-8)) * 100
            vpvs_std_dev = ((r["vpvs_std"] - real_vpvs_std) / (abs(real_vpvs_std) + 1e-8)) * 100
            if abs(real_seismic_mean) < 0.01:
                seismic_mean_dev = ((r["seismic_mean"] - real_seismic_mean) / (real_seismic_std + 1e-8)) * 100
            else:
                seismic_mean_dev = ((r["seismic_mean"] - real_seismic_mean) / (abs(real_seismic_mean) + 1e-8)) * 100
            seismic_std_dev = ((r["seismic_std"] - real_seismic_std) / (abs(real_seismic_std) + 1e-8)) * 100

            rock_physics_stats["variants"].append(
                {
                    "label": VARIANT_LABELS[var],
                    "ip_mean_html": format_dev_other(r["ip_mean"], ip_mean_dev),
                    "ip_std_html": format_dev_other(r["ip_std"], ip_std_dev),
                    "is_mean_html": format_dev_other(r["is_mean"], is_mean_dev),
                    "is_std_html": format_dev_other(r["is_std"], is_std_dev),
                    "vpvs_mean_html": format_dev_vpvs(r["vpvs_mean"], vpvs_mean_dev),
                    "vpvs_std_html": format_dev_vpvs(r["vpvs_std"], vpvs_std_dev),
                    "seismic_mean_html": format_dev_vpvs(r["seismic_mean"], seismic_mean_dev),
                    "seismic_std_html": format_dev_vpvs(r["seismic_std"], seismic_std_dev),
                }
            )

    # Variant scorecard table
    scorecard_table: list[dict[str, Any]] = []
    if scorecard:
        for var in sorted_variants:
            sc = scorecard[var]
            rank = rank_map[var]
            medal = medal_map.get(rank, f"#{rank}")
            bar_width = max(5, min(100, int(100 - sc["overall"])))
            scorecard_table.append(
                {
                    "rank": rank,
                    "medal": medal,
                    "label": VARIANT_LABELS[var],
                    "facies_rmse": f"{sc['facies_rmse']:.2f}",
                    "ip_dev": f"{sc['ip_dev']:.2f}",
                    "is_dev": f"{sc['is_dev']:.2f}",
                    "vpvs_dev": f"{sc['vpvs_dev']:.2f}",
                    "conn_dev": f"{sc['conn_dev']:.2f}",
                    "ssim_dev": f"{sc['ssim_dev']:.2f}",
                    "overall": f"{sc['overall']:.2f}",
                    "bar_width": bar_width,
                    "is_first": rank == 1,
                }
            )

    # KL-Divergence & Wasserstein Distance table
    distribution_table: list[dict[str, Any]] = []
    if distribution_metrics:
        prop_headers = [("ip", "Ip"), ("is", "Is"), ("vpvs", "Vp/Vs")]
        for var in VARIANTS:
            if var not in distribution_metrics:
                continue
            dm = distribution_metrics[var]
            row_metrics: list[dict[str, Any]] = []
            for pk, _pl in prop_headers:
                if pk in dm:
                    kl_val = float(dm[pk]["kl"])
                    wd_val = float(dm[pk]["wasserstein"])
                    kl_color = (
                        "#34d399"
                        if kl_val < 0.05
                        else ("#fbbf24" if kl_val < 0.15 else "#f87171")
                    )
                    wd_color = (
                        "#34d399"
                        if wd_val < 0.05
                        else ("#fbbf24" if wd_val < 0.15 else "#f87171")
                    )
                    row_metrics.append(
                        {
                            "kl": f"{kl_val:.4f}",
                            "kl_color": kl_color,
                            "wd": f"{wd_val:.4f}",
                            "wd_color": wd_color,
                            "exists": True,
                        }
                    )
                else:
                    row_metrics.append({"exists": False})
            distribution_table.append(
                {"label": VARIANT_LABELS[var], "metrics": row_metrics}
            )

    # Channel Connectivity Table
    connectivity_table: list[dict[str, Any]] = []
    if channel_connectivity:
        cc_order = ["Real"] + VARIANTS
        for key in cc_order:
            if key not in channel_connectivity:
                continue
            cc = channel_connectivity[key]
            label = (
                "Real Validation Data"
                if key == "Real"
                else VARIANT_LABELS.get(key, key)
            )
            connectivity_table.append(
                {
                    "key": key,
                    "is_real": key == "Real",
                    "label": label,
                    "n_components": f"{cc['n_components']:.1f}",
                    "largest_frac": f"{cc['largest_frac'] * 100:.1f}%",
                    "mean_size": f"{cc['mean_size']:.1f}",
                }
            )

    # 7. Rock Physics Crossplots (Removed)
    existing_rp: list[dict[str, str]] = []

    # Global milestone targets for variant galleries
    global_epoch_indices: list[int] = []
    for var in VARIANTS:
        variant_dir = outputs_dir / var
        if not variant_dir.exists():
            continue
        scale_dirs = [
            d for d in variant_dir.iterdir() if d.is_dir() and d.name.isdigit()
        ]
        if scale_dirs:
            final_scale = max(int(d.name) for d in scale_dirs)
            sample_dir = (
                variant_dir / "training_visualizations" / f"Scale_{final_scale}"
            )
            if sample_dir.is_dir():
                for p in sample_dir.glob("Facies_epoch_*.png"):
                    parts = p.stem.split("_")
                    if len(parts) >= 3:
                        try:
                            global_epoch_indices.append(int(parts[2]))
                        except ValueError:
                            continue

    global_milestone_targets: list[int] = []
    if global_epoch_indices:
        global_epoch_indices = sorted(list(set(global_epoch_indices)))
        global_milestone_targets = get_milestones(global_epoch_indices)

    properties = [
        ("Facies", "Facies"),
        ("Ip", "Ip (<i>I<sub>p</sub></i>)"),
    ]
    if use_is:
        properties.append(("Is", "Is (<i>I<sub>s</sub></i>)"))
    if use_vpvs:
        properties.append(("VpVs", "Vp/Vs"))
    properties.append(("Seismic", "Seismic"))

    kinds = [
        ("facies", "Facies Categorical Map"),
        ("ip", "Acoustic Impedance (Ip)"),
    ]
    if use_is:
        kinds.append(("is", "Shear Impedance (Is)"))
    if use_vpvs:
        kinds.append(("vp_vs", "Vp/Vs Ratio"))
    kinds.append(("seismic", "Synthetic Seismic"))

    variant_details: list[dict[str, Any]] = []
    for var in VARIANTS:
        variant_dir = outputs_dir / var
        var_data: dict[str, Any] = {
            "id": var,
            "label": VARIANT_LABELS[var],
            "exists": variant_dir.exists(),
        }
        if variant_dir.exists():
            # Loss curves
            loss_paths_g = sorted(list(variant_dir.glob("loss_G_*.png")))
            loss_paths_d = sorted(list(variant_dir.glob("loss_D_*.png")))
            var_data["losses"] = {
                "has_g": len(loss_paths_g) > 0,
                "has_d": len(loss_paths_d) > 0,
                "g": [
                    {
                        "label": path.stem.replace("loss_", "").replace("_", " "),
                        "rel_path": md_relpath(path, report_dir),
                    }
                    for path in loss_paths_g
                ],
                "d": [
                    {
                        "label": path.stem.replace("loss_", "").replace("_", " "),
                        "rel_path": md_relpath(path, report_dir),
                    }
                    for path in loss_paths_d
                ],
            }
            # Multi-scale pyramid
            pyramid_path = variant_dir / "pyramid_overview.png"
            if pyramid_path.exists():
                var_data["pyramid_overview"] = md_relpath(pyramid_path, report_dir)
            else:
                var_data["pyramid_overview"] = None

            # Gallery evolution
            scale_dirs = [
                d for d in variant_dir.iterdir() if d.is_dir() and d.name.isdigit()
            ]
            if scale_dirs:
                final_scale = max(int(d.name) for d in scale_dirs)
                sample_dir = (
                    variant_dir / "training_visualizations" / f"Scale_{final_scale}"
                )
                if sample_dir.is_dir():
                    epoch_indices: list[int] = []
                    for p in sample_dir.glob("Facies_epoch_*.png"):
                        parts = p.stem.split("_")
                        if len(parts) >= 3:
                            try:
                                epoch_indices.append(int(parts[2]))
                            except ValueError:
                                continue
                    if epoch_indices:
                        epoch_indices = sorted(list(set(epoch_indices)))
                        all_epochs = sorted(epoch_indices, reverse=True)

                        # Find variant milestones closest to global targets
                        milestones: set[int] = set()
                        if global_milestone_targets:
                            for target in global_milestone_targets:
                                closest = min(
                                    epoch_indices, key=lambda x: abs(x - target)
                                )
                                milestones.add(closest)
                        else:
                            milestones = set(get_milestones(epoch_indices))

                        slides: list[dict[str, Any]] = []
                        for epoch in all_epochs:
                            epoch_label = (
                                f"Epoch {epoch} (Final)"
                                if epoch == all_epochs[0]
                                else f"Epoch {epoch}"
                            )
                            is_milestone = epoch in milestones

                            properties_images: list[dict[str, str]] = []
                            for prop_key, prop_label in properties:
                                img_filename = f"{prop_key}_epoch_{epoch:05d}.png"
                                img_path = sample_dir / img_filename
                                rel_img = ""
                                if img_path.exists():
                                    rel_img = md_relpath(img_path, report_dir)
                                else:
                                    for p in sample_dir.glob(
                                        f"*_epoch_{epoch:05d}.png"
                                    ):
                                        if p.name.lower() == img_filename.lower():
                                            rel_img = md_relpath(p, report_dir)
                                            break
                                properties_images.append(
                                    {"label": prop_label, "rel_path": rel_img}
                                )
                            slides.append(
                                {
                                    "epoch": epoch,
                                    "epoch_label": epoch_label,
                                    "is_milestone": is_milestone,
                                    "properties": properties_images,
                                }
                            )

                        var_data["gallery"] = {
                            "has_gallery": True,
                            "slides": slides,
                            "use_scrubber": len(all_epochs) > 10,
                            "default_epoch_label": f"Epoch {all_epochs[0]} (Final)",
                        }
                    else:
                        var_data["gallery"] = {"has_gallery": False}
                else:
                    var_data["gallery"] = {"has_gallery": False}
            else:
                var_data["gallery"] = {"has_gallery": False}

            # Projections
            gen_dir = outputs_dir / var / "generated"
            var_projections: list[dict[str, Any]] = []
            if gen_dir.exists():
                for key, label in kinds:
                    suffix = f"_{key}" if key != "facies" else ""
                    available_methods: list[dict[str, str]] = []
                    for m in EMBEDDING_METHODS:
                        img_path = gen_dir / f"{m}{suffix}_comparison.png"
                        if img_path.exists():
                            available_methods.append(
                                {
                                    "label": EMBEDDING_LABELS[m],
                                    "rel_path": md_relpath(img_path, report_dir),
                                }
                            )
                    if available_methods:
                        var_projections.append(
                            {"key": key, "label": label, "methods": available_methods}
                        )
            var_data["projections"] = var_projections
        variant_details.append(var_data)

    # 9. Cross-Variant Manifold Progression
    cross_variant_synthesis: list[dict[str, Any]] = []
    for key, label in kinds:
        method_data: list[dict[str, Any]] = []
        for method in EMBEDDING_METHODS:
            pattern = re.compile(
                rf"^{method}_{key}_comparison_all_variants_epoch(\d+)\.png$"
            )
            epoch_map: dict[int, Path] = {}
            for p in outputs_dir.glob("*.png"):
                m = pattern.match(p.name)
                if m:
                    try:
                        epoch = int(m.group(1))
                        epoch_map[epoch] = p
                    except ValueError:
                        continue
            if epoch_map:
                epochs_list: list[dict[str, str]] = []
                for epoch in sorted(epoch_map.keys()):
                    epochs_list.append(
                        {
                            "label": f"Epoch {epoch}",
                            "rel_path": md_relpath(epoch_map[epoch], report_dir),
                        }
                    )
                method_data.append(
                    {
                        "method_label": EMBEDDING_LABELS[method],
                        "type": "epochs",
                        "images": epochs_list,
                    }
                )
            else:
                fallback_path = (
                    outputs_dir / f"{method}_{key}_comparison_all_variants.png"
                )
                if fallback_path.exists():
                    method_data.append(
                        {
                            "method_label": EMBEDDING_LABELS[method],
                            "type": "fallback",
                            "images": [
                                {
                                    "label": "Final Grid",
                                    "rel_path": md_relpath(fallback_path, report_dir),
                                }
                            ],
                        }
                    )
        if method_data:
            cross_variant_synthesis.append(
                {"key": key, "label": label, "methods": method_data}
            )

    # 10. Distribution histograms
    dist_plots = [
        ("distribution_hist_facies.png", "Facies Categories"),
        ("distribution_hist_ip.png", "Acoustic Impedance (Ip)"),
    ]
    if use_is:
        dist_plots.append(("distribution_hist_is.png", "Shear Impedance (Is)"))
    if use_vpvs:
        dist_plots.append(("distribution_hist_vpvs.png", "Vp/Vs Ratio"))
    dist_plots.append(("distribution_hist_seismic.png", "Synthetic Seismic Amplitude"))

    distribution_analysis: list[dict[str, str]] = []
    for filename, label in dist_plots:
        img_path = outputs_dir / filename
        if img_path.exists():
            distribution_analysis.append(
                {"label": label, "rel_path": md_relpath(img_path, report_dir)}
            )

    # 11. Spatial Continuity variograms
    var_plots = [
        ("variogram_facies.png", "Facies Categories"),
        ("variogram_ip.png", "Acoustic Impedance (Ip)"),
    ]
    if use_is:
        var_plots.append(("variogram_is.png", "Shear Impedance (Is)"))
    if use_vpvs:
        var_plots.append(("variogram_vp_vs.png", "Vp/Vs Ratio"))
    var_plots.append(("variogram_seismic.png", "Synthetic Seismic"))

    spatial_continuity: list[dict[str, str]] = []
    for filename, label in var_plots:
        img_path = outputs_dir / filename
        if img_path.exists():
            spatial_continuity.append(
                {"label": label, "rel_path": md_relpath(img_path, report_dir)}
            )

    # 12. Performance Table
    performance_table: list[dict[str, str]] = []
    if performance_data:
        for var in VARIANTS:
            if var in performance_data:
                p = performance_data[var]
                dur_sec = float(p["duration_sec"])
                h = int(dur_sec // 3600)
                m = int((dur_sec % 3600) // 60)
                dur_str = f"{h}h {m}m" if h > 0 else f"{m}m"
                if dur_sec == 0:
                    dur_str = "—"
                performance_table.append(
                    {
                        "label": VARIANT_LABELS[var],
                        "duration": dur_str,
                        "size_mb": f"{p['size_mb']:.1f} MB",
                    }
                )

    # Prepare Jinja2 Environment & Context
    template_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("report_template.html.jinja2")

    context: dict[str, Any] = {
        "seismic_images": seismic_images,
        "well_images": well_images,
        "available_pyramids": available_pyramids,
        "hparams_table": hparams_table,
        "variant_labels_cols": variant_labels_cols,
        "num_variant_cols": num_variant_cols,
        "comparison_grids": comparison_grids,
        "facies_proportions": facies_proportions,
        "rock_physics_stats": rock_physics_stats,
        "scorecard_table": scorecard_table,
        "distribution_table": distribution_table,
        "connectivity_table": connectivity_table,
        "existing_rp": existing_rp,
        "variant_details": variant_details,
        "cross_variant_synthesis": cross_variant_synthesis,
        "distribution_analysis": distribution_analysis,
        "spatial_continuity": spatial_continuity,
        "performance_table": performance_table,
        "use_ip": use_ip,
        "use_is": use_is,
        "use_vpvs": use_vpvs,
    }

    html_content = template.render(context)

    html_path = Path(output)
    html_path.write_text(html_content, encoding="utf-8")
    print(f"HTML Dashboard saved to: {output}")

    return output
