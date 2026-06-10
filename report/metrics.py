# pyright: reportMissingTypeStubs=false

import json
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import label
from scipy.stats import entropy, wasserstein_distance

from report.constants import VARIANTS
from report.utils import get_denormalize_fn

FloatArray = NDArray[np.float64]


class VariantQuantResult(TypedDict):
    proportions: list[float]
    rmse_error: float
    facies_kl: float
    avg_ssim: float
    ip_mean: float
    ip_std: float
    is_mean: float
    is_std: float
    vpvs_mean: float
    vpvs_std: float


class QuantitativeResults(TypedDict):
    results: dict[str, VariantQuantResult]
    real_facies_proportions: list[float]
    real_ip_mean: float
    real_ip_std: float
    real_is_mean: float
    real_is_std: float
    real_vpvs_mean: float
    real_vpvs_std: float


ConnectivityMetrics = dict[str, dict[str, float]]
DistributionMetrics = dict[str, dict[str, dict[str, float]]]
PerformanceData = dict[str, dict[str, float]]


def _compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute a simplified Global Structural Similarity Index (SSIM) between two images."""
    data_range = max(img1.max(), img2.max()) - min(img1.min(), img2.min())
    if data_range == 0:
        return 1.0

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = np.mean(img1)
    mu2 = np.mean(img2)

    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean(img1 * img2) - mu1 * mu2

    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)

    return float(numerator / denominator)


def compute_quantitative_results(
    outputs_dir: Path, data_dir: Path
) -> QuantitativeResults:
    """Compute facies proportions, rock physics stats, and RMSE for all variants."""
    real_facies_proportions = [0.6782, 0.1197, 0.1447, 0.0574]  # default fallback
    real_indices_list: list[np.ndarray] = []
    real_images_path = data_dir / "facies/facies_images.npz"
    if real_images_path.exists():
        try:
            with np.load(real_images_path) as data:
                if "images" in data:
                    images = data["images"]
                    indices = np.zeros(images.shape[:-1], dtype=np.int32)
                    indices[images[..., 0] == 255] = 1
                    indices[images[..., 2] == 255] = 2
                    indices[images[..., 1] == 255] = 3
                    counts = np.bincount(indices.flatten(), minlength=4)
                    real_facies_proportions = (counts / counts.sum()).tolist()
                    for i in range(min(10, indices.shape[0])):
                        real_indices_list.append(indices[i])
        except Exception:
            pass

    real_ip_mean, real_ip_std = 7335.7, 1302.6
    real_is_mean, real_is_std = 3863.8, 613.7
    real_vpvs_mean, real_vpvs_std = 1.895, 0.163
    stats_json_path = data_dir / "stats.json"
    if stats_json_path.exists():
        try:
            with open(stats_json_path, encoding="utf-8") as f:
                stats = json.load(f)
                real_ip_mean = stats["Ip"]["mean"]
                real_is_mean = stats["Is"]["mean"]
                real_vpvs_mean = stats["VP_VS"]["mean"]
        except Exception:
            pass

    results: dict[str, VariantQuantResult] = {}
    for var in VARIANTS:
        facies_dir = outputs_dir / var / "generated" / "facies"
        proportions = [0.0, 0.0, 0.0, 0.0]
        ssim_scores: list[float] = []
        if facies_dir.exists():
            npy_files = sorted(list(facies_dir.glob("facies_*.npy")))
            if npy_files:
                counts = np.zeros(4, dtype=np.int64)
                for i, f in enumerate(npy_files[:200]):
                    arr = np.load(f)
                    counts += np.bincount(arr.flatten(), minlength=4)[:4]
                    if (
                        var != "unconditional"
                        and i < len(real_indices_list)
                        and arr.shape == real_indices_list[i].shape
                    ):
                        ssim_scores.append(_compute_ssim(arr, real_indices_list[i]))
                proportions = (counts / counts.sum()).tolist()

        rmse_error = float(
            np.sqrt(
                np.mean(
                    (np.array(proportions) - np.array(real_facies_proportions)) ** 2
                )
            )
        )

        eps = 1e-10
        p_facies = np.array(real_facies_proportions) + eps
        q_facies = np.array(proportions) + eps
        facies_kl = float(entropy(p_facies, q_facies))

        ip_mean, ip_std = 0.0, 0.0
        is_mean, is_std = 0.0, 0.0
        vpvs_mean, vpvs_std = 0.0, 0.0

        ip_dir = outputs_dir / var / "generated" / "ip"
        if ip_dir.exists():
            npy_files = list(ip_dir.glob("*.npy"))
            if npy_files:
                vals = [np.load(f) for f in npy_files[:200]]
                ip_mean, ip_std = float(np.mean(vals)), float(np.std(vals))

        is_dir = outputs_dir / var / "generated" / "is"
        if is_dir.exists():
            npy_files = list(is_dir.glob("*.npy"))
            if npy_files:
                vals = [np.load(f) for f in npy_files[:200]]
                is_mean, is_std = float(np.mean(vals)), float(np.std(vals))

        vpvs_dir = outputs_dir / var / "generated" / "vp_vs"
        if vpvs_dir.exists():
            npy_files = list(vpvs_dir.glob("*.npy"))
            if npy_files:
                vals = [np.load(f) for f in npy_files[:200]]
                vpvs_mean, vpvs_std = float(np.mean(vals)), float(np.std(vals))

        results[var] = {
            "proportions": proportions,
            "rmse_error": rmse_error,
            "facies_kl": facies_kl,
            "avg_ssim": float(np.mean(ssim_scores)) if ssim_scores else 0.0,
            "ip_mean": ip_mean,
            "ip_std": ip_std,
            "is_mean": is_mean,
            "is_std": is_std,
            "vpvs_mean": vpvs_mean,
            "vpvs_std": vpvs_std,
        }

    return {
        "results": results,
        "real_facies_proportions": real_facies_proportions,
        "real_ip_mean": real_ip_mean,
        "real_ip_std": real_ip_std,
        "real_is_mean": real_is_mean,
        "real_is_std": real_is_std,
        "real_vpvs_mean": real_vpvs_mean,
        "real_vpvs_std": real_vpvs_std,
    }


def compute_channel_connectivity(
    outputs_dir: Path, data_dir: Path
) -> ConnectivityMetrics:
    """Compute connected-component analysis on Channel facies (class 2) for each variant."""
    print("Computing channel connectivity metrics...")
    results: ConnectivityMetrics = {}
    try:
        real_images_path = data_dir / "facies" / "facies_images.npz"
        if real_images_path.exists():
            try:
                with np.load(real_images_path) as data:
                    if "images" in data:
                        images = data["images"]
                        indices = np.zeros(images.shape[:-1], dtype=np.int32)
                        indices[images[..., 0] == 255] = 1
                        indices[images[..., 2] == 255] = 2
                        indices[images[..., 1] == 255] = 3
                        n_components_list: list[int] = []
                        largest_frac_list: list[float] = []
                        mean_size_list: list[float] = []
                        for i in range(min(50, indices.shape[0])):
                            channel_mask = (indices[i] == 2).astype(np.int32)
                            if channel_mask.sum() == 0:
                                continue
                            labeled, n_comp = label(channel_mask)
                            sizes = np.bincount(labeled.flatten())[1:]
                            n_components_list.append(int(n_comp))
                            largest_frac_list.append(
                                float(sizes.max() / sizes.sum())
                                if sizes.sum() > 0
                                else 0.0
                            )
                            mean_size_list.append(
                                float(sizes.mean()) if len(sizes) > 0 else 0.0
                            )
                        if n_components_list:
                            results["Real"] = {
                                "n_components": float(np.mean(n_components_list)),
                                "largest_frac": float(np.mean(largest_frac_list)),
                                "mean_size": float(np.mean(mean_size_list)),
                            }
            except Exception:
                pass

        for variant in VARIANTS:
            facies_dir = outputs_dir / variant / "generated" / "facies"
            if not facies_dir.exists():
                continue
            npy_files = sorted(list(facies_dir.glob("facies_*.npy")))[:100]
            if not npy_files:
                continue
            n_components_list: list[int] = []
            largest_frac_list: list[float] = []
            mean_size_list: list[float] = []
            for nf in npy_files:
                arr = np.load(nf)
                if arr.ndim == 3:
                    arr = arr[0]
                channel_mask = (arr == 2).astype(np.int32)
                if channel_mask.sum() == 0:
                    continue
                labeled, n_comp = label(channel_mask)
                sizes = np.bincount(labeled.flatten())[1:]
                n_components_list.append(int(n_comp))
                largest_frac_list.append(
                    float(sizes.max() / sizes.sum()) if sizes.sum() > 0 else 0.0
                )
                mean_size_list.append(float(sizes.mean()) if len(sizes) > 0 else 0.0)
            if n_components_list:
                results[variant] = {
                    "n_components": float(np.mean(n_components_list)),
                    "largest_frac": float(np.mean(largest_frac_list)),
                    "mean_size": float(np.mean(mean_size_list)),
                }
    except Exception as e:
        print(f"Error computing channel connectivity: {e}")
    return results


def compute_distribution_metrics(
    outputs_dir: Path, data_dir: Path
) -> DistributionMetrics:
    """Compute KL-divergence and Wasserstein distance between real and generated distributions."""
    print("Computing distribution metrics (KL, Wasserstein)...")
    metrics: DistributionMetrics = {}
    try:
        from datasets.dataset import PyramidsDataset
        from options import TrainingOptions

        base_options_path = outputs_dir / "wells_seismic" / "options.json"
        latest_params: dict[str, Any] = {}
        if base_options_path.exists():
            try:
                with open(base_options_path, encoding="utf-8") as f:
                    latest_params = cast(dict[str, Any], json.load(f))
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

        facies_batch, _, _, _ = dataset.get_scale_data(num_scales - 1)
        if facies_batch.shape[0] == 0:
            return metrics

        denormalize = get_denormalize_fn(data_dir)
        denormalize_any = cast(Any, denormalize)

        active_properties: list[str] = []
        if getattr(opt, "use_ip", True):
            active_properties.append("Ip")
        if getattr(opt, "use_is", True):
            active_properties.append("Is")
        if getattr(opt, "use_vpvs", True):
            active_properties.append("VP_VS")

        real_ip: list[FloatArray] = []
        real_is: list[FloatArray] = []
        real_vpvs: list[FloatArray] = []
        n_samples = min(10, facies_batch.shape[0])
        
        prop_indices = {name: i for i, name in enumerate(active_properties)}

        for idx in range(n_samples):
            f = facies_batch[idx].cpu()
            if "Ip" in prop_indices:
                ch_idx = num_facies_ch + prop_indices["Ip"]
                if f.shape[0] > ch_idx:
                    ip_val = f[ch_idx].numpy().flatten()
                    ip_val = np.asarray(
                        denormalize_any(ip_val, "ip", opt), dtype=np.float64
                    )
                    real_ip.append(ip_val)

            if "Is" in prop_indices:
                ch_idx = num_facies_ch + prop_indices["Is"]
                if f.shape[0] > ch_idx:
                    is_val = f[ch_idx].numpy().flatten()
                    is_val = np.asarray(
                        denormalize_any(is_val, "is", opt), dtype=np.float64
                    )
                    real_is.append(is_val)

            if "VP_VS" in prop_indices:
                ch_idx = num_facies_ch + prop_indices["VP_VS"]
                if f.shape[0] > ch_idx:
                    vpvs_val = f[ch_idx].numpy().flatten()
                    vpvs_val = np.asarray(
                        denormalize_any(vpvs_val, "vpvs", opt), dtype=np.float64
                    )
                    real_vpvs.append(vpvs_val)

        if not real_ip and not real_is and not real_vpvs:
            return metrics

        real_ip_arr: FloatArray | None = np.asarray(np.concatenate(real_ip), dtype=np.float64) if real_ip else None
        real_is_arr: FloatArray | None = np.asarray(np.concatenate(real_is), dtype=np.float64) if real_is else None
        real_vpvs_arr: FloatArray | None = np.asarray(np.concatenate(real_vpvs), dtype=np.float64) if real_vpvs else None

        properties: list[tuple[str, FloatArray]] = []
        if real_ip_arr is not None:
            properties.append(("ip", real_ip_arr))
        if real_is_arr is not None:
            properties.append(("is", real_is_arr))
        if real_vpvs_arr is not None:
            properties.append(("vpvs", real_vpvs_arr))

        for variant in VARIANTS:
            variant_metrics: dict[str, dict[str, float]] = {}
            for prop_key, real_data in properties:
                gen_dir_key = "vp_vs" if prop_key == "vpvs" else prop_key
                gen_dir = outputs_dir / variant / "generated" / gen_dir_key
                if not gen_dir.exists():
                    continue
                npy_files = sorted(list(gen_dir.glob("*.npy")))[:200]
                if not npy_files:
                    continue
                gen_data: FloatArray = np.asarray(
                    np.concatenate([np.load(f).flatten() for f in npy_files]),
                    dtype=np.float64,
                )

                n_sub = min(50000, len(real_data), len(gen_data))
                r_sub = np.random.choice(real_data, n_sub, replace=False)
                g_sub = np.random.choice(gen_data, n_sub, replace=False)
                wd = float(wasserstein_distance(r_sub, g_sub))

                bins: FloatArray = np.asarray(
                    np.linspace(
                        min(real_data.min(), gen_data.min()),
                        max(real_data.max(), gen_data.max()),
                        100,
                    ),
                    dtype=np.float64,
                )
                p_hist_raw, _ = np.histogram(real_data, bins=bins, density=True)
                q_hist_raw, _ = np.histogram(gen_data, bins=bins, density=True)
                p_hist: FloatArray = np.asarray(p_hist_raw, dtype=np.float64)
                q_hist: FloatArray = np.asarray(q_hist_raw, dtype=np.float64)
                eps = 1e-10
                p_hist = p_hist + eps
                q_hist = q_hist + eps
                p_hist = p_hist / p_hist.sum()
                q_hist = q_hist / q_hist.sum()
                kl = float(entropy(p_hist, q_hist))

                variant_metrics[prop_key] = {"kl": kl, "wasserstein": wd}

            if variant_metrics:
                metrics[variant] = variant_metrics

    except Exception as e:
        print(f"Error computing distribution metrics: {e}")
    return metrics


def get_performance_data(outputs_dir: Path) -> PerformanceData:
    """Extract training duration and model size for each variant."""
    perf: PerformanceData = {}
    for variant in VARIANTS:
        var_dir = outputs_dir / variant
        if not var_dir.exists():
            continue

        total_size = 0
        for f in var_dir.glob("**/*.pt*"):
            total_size += f.stat().st_size

        duration = 0
        log_dir = var_dir / "tensorboard_logs"
        if log_dir.exists():
            try:
                from tensorboard.backend.event_processing.event_accumulator import (
                    EventAccumulator,
                )

                ea = cast(Any, EventAccumulator(str(log_dir)))
                ea.Reload()
                all_tags = cast(list[str], ea.Tags().get("scalars", []))
                if all_tags:
                    tag = "Mean/G_TOTAL" if "Mean/G_TOTAL" in all_tags else all_tags[0]
                    events = cast(list[Any], ea.Scalars(tag))
                    if events:
                        times: list[float] = [e.wall_time for e in events]
                        total_active_time = 0
                        MAX_GAP = 900
                        for i in range(len(times) - 1):
                            gap = times[i + 1] - times[i]
                            if 0 < gap < MAX_GAP:
                                total_active_time += gap
                        duration = total_active_time
            except Exception:
                pass

        perf[variant] = {
            "size_mb": float(total_size / (1024 * 1024)),
            "duration_sec": float(duration),
        }
    return perf


def compute_scorecard_data(
    variants: list[str],
    quant_results: dict[str, VariantQuantResult],
    dist_metrics: DistributionMetrics,
    connectivity: ConnectivityMetrics,
) -> dict[str, dict[str, object]]:  # pyright: ignore[reportUnusedFunction]
    """Calculate a consolidated score for each variant based on multiple metrics."""
    scores: dict[str, dict[str, object]] = {}

    all_metrics: list[tuple[str, dict[str, float]]] = []
    for var in variants:
        if var not in quant_results or var not in dist_metrics:
            continue

        qr = quant_results[var]

        m: dict[str, float] = {
            "facies_rmse": qr["rmse_error"],
            "facies_kl": qr["facies_kl"],
            "ip_kl": float(dist_metrics[var].get("ip", {}).get("kl", 1.0)),
            "ip_wd": float(dist_metrics[var].get("ip", {}).get("wasserstein", 1.0)),
            "is_kl": float(dist_metrics[var].get("is", {}).get("kl", 1.0)),
            "is_wd": float(dist_metrics[var].get("is", {}).get("wasserstein", 1.0)),
            "ssim_loss": 1.0 - qr["avg_ssim"],
        }

        if "Real" in connectivity and var in connectivity:
            real_conn = float(connectivity["Real"]["largest_frac"])
            var_conn = float(connectivity[var]["largest_frac"])
            m["conn_err"] = abs(real_conn - var_conn)
        else:
            m["conn_err"] = 1.0

        all_metrics.append((var, m))

    if not all_metrics:
        return {}

    weights = {
        "facies_rmse": 0.20,
        "facies_kl": 0.15,
        "ssim_loss": 0.15,
        "ip_kl": 0.10,
        "ip_wd": 0.10,
        "is_kl": 0.10,
        "is_wd": 0.10,
        "conn_err": 0.10,
    }

    for key in weights:
        vals = [m[key] for _, m in all_metrics]
        min_v = min(vals)
        max_v = max(vals)
        range_v = max_v - min_v if max_v > min_v else 1.0

        for _, m in all_metrics:
            norm = (m[key] - min_v) / range_v
            m[f"{key}_score"] = (1.0 - norm) * weights[key]

    for var, m in all_metrics:
        total_score = sum(m[f"{key}_score"] for key in weights) * 100
        scores[var] = {
            "score": total_score,
            "rank": 0,
            "details": m,
        }

    sorted_vars = sorted(
        scores.keys(), key=lambda v: cast(float, scores[v]["score"]), reverse=True
    )
    for i, var in enumerate(sorted_vars):
        scores[var]["rank"] = i + 1

    return scores
