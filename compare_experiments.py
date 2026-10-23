#!/usr/bin/env python
"""Compare quantitative metrics across several experiment output directories.

Each directory must have the layout produced by ``python -m experiments``
(``<dir>/<variant>/generated/...`` plus ``<dir>/wells_seismic/options.json``).
Typical use: baseline ablation run versus a regularized run.

Example
-------
    python compare_experiments.py outputs/experiments outputs/experiments_reg \
        --labels baseline regularized --variants wells_seismic --fewshot

``--fewshot`` adds SIFID (fidelity) and intra-condition LPIPS (diversity),
see ``report/fewshot_metrics.py``. It needs the ``lpips`` and ``pytorch-fid``
packages and a GPU for reasonable speed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from report.constants import VARIANTS
from report.metrics import compute_quantitative_results

# (key, label, preference) — preference: "lower", "higher" or "ref" (closest to real).
METRICS: list[tuple[str, str, str]] = [
    ("facies_kl", "Facies KL (gen vs real)", "lower"),
    ("rmse_error", "Facies proportion RMSE", "lower"),
    ("avg_ssim", "SSIM vs real", "higher"),
    ("pairwise_ssim_mean", "Pairwise SSIM (lower = more diverse)", "lower"),
    ("proportion_std", "Proportion std across realizations", "ref:real_proportion_std"),
    ("ip_mean", "Ip mean", "ref:real_ip_mean"),
    ("ip_std", "Ip std", "ref:real_ip_std"),
    ("ip_realization_std_mean", "Ip std across realizations", "none"),
    ("seismic_mean", "Seismic mean", "ref:real_seismic_mean"),
    ("seismic_std", "Seismic std", "ref:real_seismic_std"),
]


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _best(values: list[float | None], pref: str, ref: float | None) -> int | None:
    valid = [(i, v) for i, v in enumerate(values) if v is not None]
    if len(valid) < 2:
        return None
    if pref == "lower":
        return min(valid, key=lambda t: t[1])[0]
    if pref == "higher":
        return max(valid, key=lambda t: t[1])[0]
    if pref.startswith("ref") and ref is not None:
        return min(valid, key=lambda t: abs(t[1] - ref))[0]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+", type=Path, help="Experiment output directories to compare.")
    parser.add_argument("--labels", nargs="+", help="Column labels (default: directory names).")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    parser.add_argument("--output", type=Path, default=None, help="Optional markdown file to write.")
    parser.add_argument(
        "--fewshot",
        action="store_true",
        help="Also compute SIFID and intra-condition LPIPS (slower; needs lpips and pytorch-fid).",
    )
    parser.add_argument(
        "--fewshot-max-conditions",
        type=int,
        default=None,
        help="Limit few-shot metrics to the first N conditioning sections (all their realizations).",
    )
    args = parser.parse_args()

    labels = args.labels or [d.name for d in args.dirs]
    if len(labels) != len(args.dirs):
        parser.error("--labels must match the number of directories")

    results = [compute_quantitative_results(d, args.data_dir) for d in args.dirs]

    fewshot: list[dict[str, Any]] = []
    fewshot_ref: Any = None
    if args.fewshot:
        from report.fewshot_metrics import compute_fewshot_metrics

        for d in args.dirs:
            res, fewshot_ref = compute_fewshot_metrics(
                d, args.data_dir, list(args.variants), max_conditions=args.fewshot_max_conditions
            )
            fewshot.append(dict(res))

    lines: list[str] = []
    for variant in args.variants:
        per_run = [r["results"].get(variant) for r in results]
        if not any(per_run):
            lines.append(f"\n## {variant}: no generated outputs in any run\n")
            continue
        lines.append(f"\n## {variant}\n")
        lines.append("| Metric | " + " | ".join(labels) + " | Real | Best |")
        lines.append("|---|" + "---|" * len(labels) + "---|---|")
        for key, label, pref in METRICS:
            values: list[float | None] = []
            for r in per_run:
                raw: Any = r.get(key) if r is not None else None  # type: ignore[arg-type]
                values.append(float(raw) if raw is not None else None)
            ref_key = pref.split(":", 1)[1] if pref.startswith("ref:") else None
            ref: float | None = None
            if ref_key is not None:
                ref_raw: Any = results[0].get(ref_key)  # type: ignore[arg-type]
                ref = float(ref_raw) if ref_raw is not None else None
            best = _best(values, pref, ref)
            cells = [_fmt(v) if v is not None else "-" for v in values]
            lines.append(
                f"| {label} | " + " | ".join(cells) + f" | {_fmt(ref) if ref is not None else '-'} | "
                + (labels[best] if best is not None else "-") + " |"
            )
        if fewshot and fewshot_ref is not None:
            fs_rows: list[tuple[str, str, str, float]] = [
                ("sifid_mean", "SIFID vs conditioning section (lower = better)", "lower", fewshot_ref.real_sifid_cross),
                ("lpips_intra", "LPIPS between realizations (higher = more diverse)", "higher", fewshot_ref.real_lpips_cross),
            ]
            for attr, label, pref, ref_val in fs_rows:
                fs_values: list[float | None] = []
                for fs in fewshot:
                    item: Any = fs.get(variant)
                    fs_values.append(float(getattr(item, attr)) if item is not None else None)
                best = _best(fs_values, pref, None)
                cells = [_fmt(v) if v is not None else "-" for v in fs_values]
                lines.append(
                    f"| {label} | " + " | ".join(cells) + f" | {_fmt(ref_val)}* | "
                    + (labels[best] if best is not None else "-") + " |"
                )
        props = [r["proportions"] if r is not None else None for r in per_run]
        real_props = results[0]["real_facies_proportions"]
        lines.append(
            "| Facies proportions | "
            + " | ".join("[" + ", ".join(f"{p:.3f}" for p in pr) + "]" if pr else "-" for pr in props)
            + " | [" + ", ".join(f"{p:.3f}" for p in real_props) + "] | - |"
        )

    if fewshot:
        lines.append(
            "\n\\* Few-shot reference values are measured between two *different* real sections: "
            "SIFID shows how far an unrelated real section is from the reference, "
            "LPIPS shows the perceptual variability of the real data set."
        )

    text = "\n".join(lines) + "\n"
    print(text)
    if args.output:
        args.output.write_text(text)
        print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
