"""Few-shot generative metrics: SIFID (fidelity) and intra-condition LPIPS (diversity).

SIFID
    Single-Image FID (Shaham et al., "SinGAN", ICCV 2019). Frechet distance
    between the internal patch statistics of two images, using the 64-channel
    features of the first Inception-v3 block (``pytorch-fid`` weights). Each
    generated realization is compared with the real section it was conditioned
    on. Lower is better.

Intra-condition LPIPS
    Mean LPIPS (Zhang et al., CVPR 2018, AlexNet backbone) between every pair
    of realizations generated from the same conditioning section. It measures
    how different the realizations of one well/seismic configuration are.
    Higher means more diverse.

Calibration values computed on real data are reported alongside:

* ``real_sifid_cross``: SIFID between two *different* real sections. A
  generator whose SIFID approaches this value produces samples that are as far
  from their reference as an unrelated real section is.
* ``real_lpips_cross``: LPIPS between two different real sections, an upper
  reference for perceptual variability in the data set.

Sample-to-condition mapping
    ``experiments/generation.py`` assigns realization ``k`` (1-based file
    index) to ``test_indices[(k - 1) % len(test_indices)]``, where the test
    split is the complement of ``PyramidsDataset.select_equally_spaced``.

Usage::

    python -m report.fewshot_metrics outputs/experiments --variants wells_seismic
"""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

import utils
from datasets.dataset import PyramidsDataset
from models.palette import PALETTE_RGB
from options import TrainingOptions
from report.constants import VARIANTS

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float32]


@dataclass
class FewShotResult:
    """Few-shot metrics for one experiment variant."""

    sifid_mean: float
    sifid_std: float
    lpips_intra: float
    lpips_intra_std: float
    n_samples: int
    n_conditions: int


@dataclass
class FewShotReference:
    """Calibration values computed on real sections only."""

    real_sifid_cross: float
    real_lpips_cross: float


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _load_options(outputs_dir: Path, data_dir: Path) -> TrainingOptions:
    """Training options of the first variant with an ``options.json``."""
    opts = TrainingOptions()
    for variant in VARIANTS:
        path = outputs_dir / variant / "options.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                raw: dict[str, Any] = json.load(f)
            for key, value in raw.items():
                if hasattr(opts, key):
                    setattr(opts, key, value)
            break
    opts.input_path = str(data_dir)
    # Same overrides as experiments/runner.py so the data set is identical.
    opts.use_wells = True
    opts.use_seismic = True
    opts.use_rock_physics = True
    return opts


def load_real_and_split(
    outputs_dir: Path, data_dir: Path
) -> tuple[IntArray, list[int]]:
    """Return real facies class maps ``(N, H, W)`` and the sorted test indices."""
    opts = _load_options(outputs_dir, data_dir)
    dataset = PyramidsDataset(opts)
    n_total = len(dataset)
    seen = PyramidsDataset(opts).select_equally_spaced(opts.num_train_pyramids)
    test = sorted(set(range(n_total)) - set(seen)) or list(range(n_total))

    facies, _, _, _ = dataset.get_scale_data(len(dataset.scales) - 1)
    real = np.stack(
        [
            np.asarray(utils.rgb_to_facies(facies[i, :3]), dtype=np.int64)
            for i in range(n_total)
        ]
    )
    return real, test


def load_generated(variant_dir: Path) -> list[tuple[int, IntArray]]:
    """Generated class maps as ``(1-based file index, array)`` pairs."""
    facies_dir = variant_dir / "generated" / "facies"
    out: list[tuple[int, IntArray]] = []
    for path in sorted(facies_dir.glob("facies_*.npy")):
        k = int(path.stem.split("_")[-1])
        out.append((k, np.asarray(np.load(path), dtype=np.int64)))
    return out


_PALETTE = np.asarray(PALETTE_RGB, dtype=np.float32)


def to_rgb_batch(class_maps: list[IntArray], device: torch.device) -> torch.Tensor:
    """Class maps to a ``(B, 3, H, W)`` tensor in ``[0, 1]`` using the facies palette."""
    rgb = np.stack([_PALETTE[np.clip(m, 0, len(_PALETTE) - 1)] for m in class_maps])
    return torch.as_tensor(rgb).permute(0, 3, 1, 2).contiguous().to(device)


# ---------------------------------------------------------------------------
# Models (third-party packages without type information are used via Any)
# ---------------------------------------------------------------------------


def frechet_distance(
    mu1: NDArray[np.float64],
    sigma1: NDArray[np.float64],
    mu2: NDArray[np.float64],
    sigma2: NDArray[np.float64],
    eps: float = 1e-6,
) -> float:
    """Frechet distance between two Gaussians.

    Same formula and numerical safeguards as
    ``pytorch_fid.fid_score.calculate_frechet_distance``, which cannot be
    used directly because it passes ``disp=False`` to ``scipy.linalg.sqrtm``,
    an argument removed in recent SciPy releases.
    """
    linalg: Any = importlib.import_module("scipy.linalg")
    diff = mu1 - mu2
    covmean: Any = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    covmean_arr: NDArray[Any] = np.asarray(covmean)
    if np.iscomplexobj(covmean_arr):
        imag_part = np.asarray(np.imag(covmean_arr), dtype=np.float64)
        if not np.allclose(np.diagonal(imag_part), 0, atol=1e-3):
            raise ValueError(
                f"Imaginary component {float(np.max(np.abs(imag_part)))} in sqrtm"
            )
        covmean_arr = np.asarray(np.real(covmean_arr), dtype=np.float64)
    return float(
        diff.dot(diff)
        + np.trace(sigma1)
        + np.trace(sigma2)
        - 2 * np.trace(covmean_arr)
    )


class SIFID:
    """Single-Image FID on the first Inception-v3 block (64 features)."""

    def __init__(self, device: torch.device) -> None:
        inception: Any = importlib.import_module("pytorch_fid.inception")
        block = inception.InceptionV3.BLOCK_INDEX_BY_DIM[64]
        self.model: Any = inception.InceptionV3([block]).to(device).eval()
        self.device = device

    @torch.no_grad()
    def stats(self, images: torch.Tensor) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Per-image mean and covariance of the spatial feature vectors."""
        feats: torch.Tensor = self.model(images)[0]  # (B, 64, h, w)
        b, c = feats.shape[0], feats.shape[1]
        flat = feats.permute(0, 2, 3, 1).reshape(b, -1, c).double().cpu().numpy()
        out: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
        for i in range(b):
            x = flat[i]
            mean = np.asarray(x.mean(axis=0), dtype=np.float64)
            cov = np.asarray(np.cov(x, rowvar=False), dtype=np.float64)
            out.append((mean, cov))
        return out

    def distance(
        self,
        a: tuple[NDArray[np.float64], NDArray[np.float64]],
        b: tuple[NDArray[np.float64], NDArray[np.float64]],
    ) -> float:
        """Frechet distance between two (mean, covariance) pairs."""
        return frechet_distance(a[0], a[1], b[0], b[1])


class LPIPSDistance:
    """LPIPS with the AlexNet backbone (inputs in ``[0, 1]``)."""

    def __init__(self, device: torch.device) -> None:
        lpips: Any = importlib.import_module("lpips")
        self.model: Any = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> NDArray[np.float64]:
        """Distances for aligned batches ``x[i]`` vs ``y[i]``."""
        d: torch.Tensor = self.model(x * 2 - 1, y * 2 - 1)
        return d.flatten().double().cpu().numpy()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_variant(
    variant_dir: Path,
    real: IntArray,
    test: list[int],
    sifid: SIFID,
    lpips_fn: LPIPSDistance,
    device: torch.device,
    max_conditions: int | None = None,
    batch_size: int = 16,
) -> FewShotResult | None:
    """SIFID against the conditioning section and intra-condition LPIPS.

    ``max_conditions`` keeps every realization of the first ``N`` conditioning
    sections, so a subset still has several realizations per condition.
    """
    generated = load_generated(variant_dir)
    cond = [test[(k - 1) % len(test)] for k, _ in generated]
    if max_conditions is not None:
        keep = set(sorted(set(cond))[:max_conditions])
        selected = [i for i, c in enumerate(cond) if c in keep]
        generated = [generated[i] for i in selected]
        cond = [cond[i] for i in selected]
    if not generated:
        return None

    # Real statistics, cached per conditioning section.
    real_stats: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    needed = sorted(set(cond))
    for i in range(0, len(needed), batch_size):
        chunk = needed[i : i + batch_size]
        for c, st in zip(chunk, sifid.stats(to_rgb_batch([real[c] for c in chunk], device))):
            real_stats[c] = st

    sifids: list[float] = []
    for i in range(0, len(generated), batch_size):
        maps = [m for _, m in generated[i : i + batch_size]]
        for j, st in enumerate(sifid.stats(to_rgb_batch(maps, device))):
            sifids.append(sifid.distance(real_stats[cond[i + j]], st))

    # Intra-condition LPIPS over every pair of realizations of one section.
    groups: dict[int, list[int]] = {}
    for pos, c in enumerate(cond):
        groups.setdefault(c, []).append(pos)
    pairs = [
        (a, b)
        for members in groups.values()
        for ai, a in enumerate(members)
        for b in members[ai + 1 :]
    ]
    lp: list[float] = []
    for i in range(0, len(pairs), batch_size):
        chunk_pairs = pairs[i : i + batch_size]
        x = to_rgb_batch([generated[a][1] for a, _ in chunk_pairs], device)
        y = to_rgb_batch([generated[b][1] for _, b in chunk_pairs], device)
        lp.extend(lpips_fn(x, y).tolist())

    return FewShotResult(
        sifid_mean=float(np.mean(sifids)),
        sifid_std=float(np.std(sifids)),
        lpips_intra=float(np.mean(lp)) if lp else float("nan"),
        lpips_intra_std=float(np.std(lp)) if lp else float("nan"),
        n_samples=len(generated),
        n_conditions=len(groups),
    )


def compute_reference(
    real: IntArray,
    test: list[int],
    sifid: SIFID,
    lpips_fn: LPIPSDistance,
    device: torch.device,
    n_pairs: int = 200,
    seed: int = 0,
    batch_size: int = 16,
) -> FewShotReference:
    """SIFID and LPIPS between random pairs of *different* real test sections."""
    rng = np.random.default_rng(seed)
    pairs: list[tuple[int, int]] = []
    while len(pairs) < n_pairs:
        a, b = rng.choice(test, size=2, replace=False)
        pairs.append((int(a), int(b)))

    sif: list[float] = []
    lp: list[float] = []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i : i + batch_size]
        xa = to_rgb_batch([real[a] for a, _ in chunk], device)
        xb = to_rgb_batch([real[b] for _, b in chunk], device)
        for sa, sb in zip(sifid.stats(xa), sifid.stats(xb)):
            sif.append(sifid.distance(sa, sb))
        lp.extend(lpips_fn(xa, xb).tolist())
    return FewShotReference(
        real_sifid_cross=float(np.mean(sif)), real_lpips_cross=float(np.mean(lp))
    )


def compute_fewshot_metrics(
    outputs_dir: Path,
    data_dir: Path,
    variants: list[str],
    device: torch.device | None = None,
    max_conditions: int | None = None,
    write_json: bool = True,
) -> tuple[dict[str, FewShotResult], FewShotReference]:
    """Compute few-shot metrics for every variant with generated outputs."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real, test = load_real_and_split(outputs_dir, data_dir)
    sifid = SIFID(dev)
    lpips_fn = LPIPSDistance(dev)

    results: dict[str, FewShotResult] = {}
    for variant in variants:
        res = compute_variant(
            outputs_dir / variant, real, test, sifid, lpips_fn, dev, max_conditions
        )
        if res is not None:
            results[variant] = res
    reference = compute_reference(real, test, sifid, lpips_fn, dev)

    if write_json:
        payload: dict[str, Any] = {
            "variants": {k: asdict(v) for k, v in results.items()},
            "reference": asdict(reference),
        }
        with open(outputs_dir / "fewshot_metrics.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return results, reference


def load_or_compute_fewshot(
    outputs_dir: Path,
    data_dir: Path,
    variants: list[str] | None = None,
) -> tuple[dict[str, FewShotResult], FewShotReference] | None:
    """Return cached few-shot metrics, recomputing when they are missing or stale.

    The cache in ``<outputs_dir>/fewshot_metrics.json`` is reused only when
    every variant with generated outputs is present with the same number of
    samples. Returns ``None`` when the metrics cannot be computed (missing
    packages, no GPU memory, ...), so callers can fall back gracefully.
    """
    names = [
        v
        for v in (variants or VARIANTS)
        if (outputs_dir / v / "generated" / "facies").is_dir()
    ]
    if not names:
        return None
    counts = {
        v: len(list((outputs_dir / v / "generated" / "facies").glob("facies_*.npy")))
        for v in names
    }

    cache = outputs_dir / "fewshot_metrics.json"
    if cache.exists():
        try:
            with open(cache, encoding="utf-8") as f:
                payload: dict[str, Any] = json.load(f)
            cached_variants: dict[str, Any] = payload.get("variants", {})
            if all(
                v in cached_variants and cached_variants[v].get("n_samples") == counts[v]
                for v in names
            ):
                results = {v: FewShotResult(**cached_variants[v]) for v in names}
                return results, FewShotReference(**payload["reference"])
        except (OSError, ValueError, KeyError, TypeError):
            pass

    try:
        return compute_fewshot_metrics(outputs_dir, data_dir, names)
    except Exception as exc:  # noqa: BLE001 - report generation must not fail
        print(f"[fewshot] metrics unavailable: {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "Few-shot generative metrics.").splitlines()[0]
    )
    parser.add_argument("outputs_dir", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    parser.add_argument(
        "--max-conditions",
        type=int,
        default=None,
        help="Use only the first N conditioning sections (all their realizations).",
    )
    args = parser.parse_args()

    results, ref = compute_fewshot_metrics(
        args.outputs_dir, args.data_dir, args.variants, max_conditions=args.max_conditions
    )
    print(f"{'variant':16s} {'SIFID':>16s} {'LPIPS intra':>18s} {'n':>6s} {'conds':>6s}")
    for name, r in results.items():
        print(
            f"{name:16s} {r.sifid_mean:8.4f} ± {r.sifid_std:6.4f}"
            f" {r.lpips_intra:9.4f} ± {r.lpips_intra_std:6.4f}"
            f" {r.n_samples:6d} {r.n_conditions:6d}"
        )
    print(
        f"{'real (cross)':16s} {ref.real_sifid_cross:8.4f}"
        f"          {ref.real_lpips_cross:9.4f}"
    )


if __name__ == "__main__":
    main()
