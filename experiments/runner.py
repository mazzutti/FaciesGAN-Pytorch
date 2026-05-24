"""Main execution runner for experiments."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

import utils
from config import CheckpointFilenames
from datasets.dataset import PyramidsDataset
from datasets.utils import build_conditioning_pyramids
from device import device_manager
from enums import ExperimentVariant
from log import format_time
from main import get_arguments as get_main_arguments
from options import ExperimentOptions, TrainingOptions

from .cli import build_training_args, get_arguments
from .embeddings import (
    ExperimentCache,
    compute_shared_embeddings,
    load_shared_embeddings,
    save_shared_embeddings,
)
from .generation import generate_variant
from .plotting import plot_method_all_variants, plot_sample_grid
from .training import find_last_completed_scale, read_completed_epochs, train_variant

logger = logging.getLogger(__name__)


@dataclass
class PlotData:
    """Encapsulates data required for a specific comparison plot."""

    kind: str
    data_dict: dict[str, list[np.ndarray]]
    real_np: np.ndarray | None


def _has_loadable_scale(model_path: str, max_scan: int = 64) -> bool:
    """Return True when at least one scale folder contains CheckpointFilenames.NOISE_AMP."""
    model_root = Path(model_path)
    for scale in range(max_scan):
        amp_path = model_root / str(scale) / CheckpointFilenames.NOISE_AMP
        if amp_path.is_file():
            return True
    return False


def main() -> None:
    parser = get_arguments()
    args = parser.parse_args(namespace=ExperimentOptions())

    if not getattr(args, "enable_logging", False):
        logging.disable(logging.CRITICAL)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
    )

    base_output = args.output_path
    nproc = args.nproc_per_node

    active_variants = list(ExperimentVariant)
    if args.variants:
        valid_ids = [ev.id for ev in ExperimentVariant]
        for v in args.variants:
            if v not in valid_ids:
                parser.error(f"Invalid variant: {v}. Must be one of {valid_ids}")
        active_variants = [ev for ev in ExperimentVariant if ev.id in args.variants]

    # Map variant name -> model path (either from training or --model-paths)
    model_paths: dict[str, str] = {}
    any_retrained = False

    total_start = time.time()

    if args.skip_training:
        if not args.model_paths:
            variant_names = ", ".join(ev.id for ev in active_variants)
            parser.error(
                f"--skip-training requires --model-paths with {len(active_variants)} paths "
                f"({variant_names})."
            )
        if len(args.model_paths) != len(active_variants):
            variant_names = ", ".join(ev.id for ev in active_variants)
            parser.error(
                f"Expected {len(active_variants)} model paths matching the active variants ({variant_names}), "
                f"but got {len(args.model_paths)} paths."
            )
        for ev, path in zip(active_variants, args.model_paths):
            if not Path(path).is_dir():
                parser.error(f"Model path does not exist: {path}")
            if not _has_loadable_scale(path):
                parser.error(
                    f"Model path has no loadable scale checkpoints (missing {CheckpointFilenames.NOISE_AMP}): {path}"
                )
            model_paths[ev.id] = path
        print("Skipping training, using provided model paths.")
    else:
        # ── Train active variants ──
        print("=" * 70)
        print("FACIESGAN CONDITIONING-ABLATION EXPERIMENTS")
        print("=" * 70)
        print(
            f"Device: GPU {args.gpu_device} for generation/plotting (DDP training runs in subprocesses with {nproc} GPUs)"
        )
        print(f"compile_backend: {'ON' if args.compile_backend else 'OFF'}")
        print(f"Variants: {', '.join(ev.id for ev in active_variants)}")
        print(f"Output: {base_output}")
        print("=" * 70 + "\n")

        for ev in active_variants:
            name = ev.id
            variant_output = str(Path(base_output) / name)
            utils.create_dirs(variant_output)
            wells_flag = "ON" if ev.value.use_wells else "OFF"
            seismic_flag = "ON" if ev.value.use_seismic else "OFF"

            # Check for existing checkpoint
            last_done = find_last_completed_scale(variant_output)
            # 1. Determine base start point (group-aware for parallel scales)
            group_size = max(1, int(args.num_parallel_scales))
            if last_done == -1:
                # No existing training found
                effective_start_scale = 0
                effective_start_epoch = 0
            else:
                # Training exists up to last_done. For parallel-scale training,
                # resume from the BEGINNING of the active scale group.
                completed_in_last = read_completed_epochs(variant_output, last_done)
                current_group_start = (last_done // group_size) * group_size
                next_group_start = current_group_start + group_size

                if completed_in_last >= args.num_iter:
                    # Current group is finished.
                    if last_done >= args.stop_scale:
                        # ALL scales finished
                        effective_start_scale = last_done
                        effective_start_epoch = completed_in_last
                    else:
                        # Move to the next group start.
                        effective_start_scale = next_group_start
                        effective_start_epoch = 0
                else:
                    # Current group is in progress.
                    effective_start_scale = current_group_start
                    effective_start_epoch = completed_in_last

            # 2. Skip if fully trained
            if (
                effective_start_scale >= args.stop_scale
                and effective_start_epoch >= args.num_iter
            ):
                model_paths[name] = variant_output
                print("\n" + "─" * 60)
                print(
                    f"Skipping variant: {name} (fully trained - scale {effective_start_scale}, epoch {effective_start_epoch}/{args.num_iter})"
                )
                print(variant_output)
                print("─" * 60)
                continue

            # 3. Resume logic
            resume_scale = effective_start_scale
            resume_epoch = effective_start_epoch

            print("\n" + "─" * 60)
            print(f"Training variant: {name}")
            print(f"wells={wells_flag}  seismic={seismic_flag}")

            if resume_epoch > 0:
                print(f"Resuming from scale {resume_scale}, epoch {resume_epoch}")
            elif resume_scale > 0:
                print(
                    f"Starting from scale {resume_scale} (scales 0-{resume_scale - 1} already done)"
                )
            else:
                print("Starting training from scratch (scale 0, epoch 0)")

            print(
                f"DDP: {nproc} GPUs  compile_backend: {'ON' if args.compile_backend else 'OFF'}"
            )
            print("─" * 60)

            variant_args = build_training_args(
                args, ev.value, variant_output, start_scale=resume_scale
            )

            variant_start = time.time()
            train_variant(variant_args, nproc)
            any_retrained = True
            elapsed = format_time(int(time.time() - variant_start))

            # --output-fullpath places artifacts directly in variant_output
            model_paths[name] = variant_output
            print(f"Training complete ({elapsed}) -> {variant_output}")

            # ── Load base options & dataset (needed for plots and embeddings) ──
            device_manager.initialize(gpu_id=args.gpu_device, use_cpu=args.use_cpu)
    _first_ev = active_variants[0]
    first_model = model_paths[_first_ev.id]
    _base_args = build_training_args(args, _first_ev.value, first_model)
    _base_opts = get_main_arguments().parse_args(
        _base_args, namespace=TrainingOptions()
    )
    _base_opts.rec = False
    # Force loading of all real conditioning data types for comparison grids and embedding manifolds
    _base_opts.use_wells = True
    _base_opts.use_seismic = True
    _base_opts.use_rock_physics = True

    _dataset = PyramidsDataset(_base_opts)
    # NOTE: We no longer subset the dataset here. By keeping the full dataset,
    # we can use absolute indices (from seen_indices) to retrieve the correct
    # ground-truth samples for comparison plots and embeddings.

    # ── Load seen indices from the first trained variant's checkpoint ──
    seen_indices: list[int] = []
    # Scan scales from finest to coarsest to find the most recent seen_indices
    last_scale = find_last_completed_scale(first_model)
    for s in range(last_scale, -1, -1):
        ckpt_path = Path(first_model) / str(s) / CheckpointFilenames.EPOCH_CKPT
        if ckpt_path.exists():
            try:
                from training.checkpoint import Checkpoint

                ckpt = Checkpoint.load(str(ckpt_path))
                seen_indices_raw = ckpt.seen_indices
                if seen_indices_raw:
                    # Type hint for the analyzer to understand the pair indexing
                    raw_list: list[Any] = seen_indices_raw
                    if isinstance(raw_list[0], (list, tuple)):
                        # New format: [[rel0, orig0], [rel1, orig1], ...]
                        # We use the absolute (orig) indices for generation to ensure
                        # we fetch the correct conditioning from the full dataset.
                        well_choices = [int(p[1]) for p in raw_list]
                    else:
                        # Old format: [rel0, rel1, ...]
                        well_choices = [int(idx) for idx in raw_list]

                    seen_indices = well_choices
                    print(
                        f"\n[INFO] Loaded {len(seen_indices)} seen indices from checkpoint."
                    )
                    break
            except Exception:
                logger.warning(
                    "Could not load seen indices from %s", ckpt_path, exc_info=True
                )

    # Pre-build pyramids for generation once
    wells_pyramid, seismic_pyramid = build_conditioning_pyramids(_base_opts)

    # Move pyramids to device
    wells_pyramid = device_manager.to_device(wells_pyramid)
    seismic_pyramid = device_manager.to_device(seismic_pyramid)

    # ── Initialize data containers ──
    all_facies: dict[str, list[np.ndarray]] = {}
    all_ip: dict[str, list[np.ndarray]] = {}
    all_is: dict[str, list[np.ndarray]] = {}
    all_vpvs: dict[str, list[np.ndarray]] = {}
    all_seismic: dict[str, list[np.ndarray]] = {}
    all_mask_indexes: dict[str, torch.Tensor] = {}
    shared: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}

    # ── Try to load cached embeddings ──
    # Reuse only when no variant was retrained (all skipped); if any model
    # was retrained the old embeddings are stale and must be recomputed.
    cached = (
        load_shared_embeddings(base_output, args.num_iter)
        if not any_retrained
        else None
    )
    if cached is not None:
        shared = cached.shared
        all_facies = cached.all_facies
        all_mask_indexes = cached.all_mask_indexes
        all_ip = cached.all_ip
        all_is = cached.all_is
        all_vpvs = cached.all_vpvs
        all_seismic = cached.all_seismic
        print(f"Loaded cached embeddings for epoch {args.num_iter}")
        # Verify cache completeness (especially for rock physics)
        if not all_is or not all_vpvs:
            print("    [Cache] Incomplete (missing rock physics). Recomputing...")
            cached = None
    if cached is None:
        # ── Generate facies (and rock_physics) from all trained models ──
        print("\n" + "=" * 70)
        print("GENERATING FACIES FROM TRAINED MODELS")
        print("=" * 70 + "\n")

        from models.utils import calculate_channels

        for ev in active_variants:
            name = ev.id
            model_path = model_paths[name]
            gen_output = str(Path(base_output) / name / "generated")

            print(f"Generating from variant: {name}")
            print(f"  model: {model_path}")

            # Calculate noise channels for this specific variant configuration
            v_args = build_training_args(args, ev.value, model_path)
            variant_opts = get_main_arguments().parse_args(
                v_args, namespace=TrainingOptions()
            )
            channels = calculate_channels(variant_opts)

            (
                variant_facies,
                variant_ip,
                variant_is,
                variant_vpvs,
                variant_seismic,
                variant_mi,
            ) = generate_variant(
                model_path=model_path,
                opts=variant_opts,
                how_many=args.how_many,
                wells_pyramid=tuple(wells_pyramid.values()),
                seismic_pyramid=tuple(seismic_pyramid.values()),
                channels=channels,
                gen_output=gen_output,
                seen_indices=seen_indices,
            )
            all_facies[name] = variant_facies
            if variant_ip:
                all_ip[name] = variant_ip
            if variant_is:
                all_is[name] = variant_is
            if variant_vpvs:
                all_vpvs[name] = variant_vpvs
            if variant_seismic:
                all_seismic[name] = variant_seismic
            all_mask_indexes[name] = variant_mi

        if not args.no_embeddings:
            # ── Shared embeddings for all plots ──
            emb_methods = list(args.embedding_methods)
            print(
                f"\nComputing shared embeddings "
                f"({', '.join(m.upper() for m in emb_methods)}) ...",
                flush=True,
            )
            shared = compute_shared_embeddings(
                all_facies, _dataset, methods=emb_methods
            )
            print("    Done initial compute_shared_embeddings.", flush=True)

            # Persist for future resume
            save_shared_embeddings(
                ExperimentCache(
                    shared={},  # Shared embeddings will be computed per-kind below
                    all_facies=all_facies,
                    all_mask_indexes=all_mask_indexes,
                    all_ip=all_ip,
                    all_is=all_is,
                    all_vpvs=all_vpvs,
                    all_seismic=all_seismic,
                ),
                base_output,
                args.num_iter,
            )
            print("    Done initial save_shared_embeddings.", flush=True)
        else:
            shared = {}

    # ── Resolve effective embedding method list ────────────────────────────
    emb_methods = list(args.embedding_methods or ["isomap", "mds", "tsne", "umap"])
    emb_data_kinds: list[str] = list(args.embedding_data or ["facies", "rock_physics"])

    # ── 1. Comparison Grids ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GENERATING COMPARISON GRIDS")
    print("=" * 70)

    import utils as _utils

    # ── 1. Comparison Grids ───────────────────────────────────────────────
    # Split real tensor to avoid clamping categorical facies indices to [-1, 1]
    real_tensor, _, _, real_seismic_tensor = _dataset.get_scale_data(-1)
    norm_range: tuple[float, float] = (
        float(_base_opts.normalization_range[0]),
        float(_base_opts.normalization_range[1]),
    )
    facies_ch = _base_opts.num_facies_channels
    real_facies_tensor = real_tensor[:, :facies_ch, ...]
    real_rp_tensor = real_tensor[:, facies_ch:, ...]

    # Facies (RGB or categorical): no denormalization
    real_facies_np = _utils.torch2np(real_facies_tensor, denormalize=False)
    # Rock Physics (continuous): apply denormalization
    real_rp_np = _utils.torch2np(
        real_rp_tensor, denormalize=True, normalization_range=norm_range
    )
    # Recombine into full numpy array
    real_full_np = np.concatenate([real_facies_np, real_rp_np], axis=-1)

    # Define kinds to plot
    plots = [
        PlotData(
            "facies",
            all_facies,
            (
                real_full_np[..., :facies_ch]
                if real_full_np.shape[-1] >= facies_ch
                else None
            ),
        ),
        PlotData(
            "ip",
            all_ip,
            (
                real_full_np[..., facies_ch]
                if (all_ip and real_full_np.shape[-1] > facies_ch)
                else None
            ),
        ),
        PlotData(
            "is",
            all_is,
            (
                real_full_np[..., facies_ch + 1]
                if (all_is and real_full_np.shape[-1] > facies_ch + 1)
                else None
            ),
        ),
        PlotData(
            "vp_vs",
            all_vpvs,
            (
                real_full_np[..., facies_ch + 2]
                if (all_vpvs and real_full_np.shape[-1] > facies_ch + 2)
                else None
            ),
        ),
        PlotData(
            "seismic",
            all_seismic,
            (
                np.transpose(device_manager.to_numpy(real_seismic_tensor), (0, 2, 3, 1))
                if (
                    all_seismic
                    and real_seismic_tensor is not None
                    and real_seismic_tensor.numel() > 0
                    and len(real_seismic_tensor.shape) == 4
                )
                else None
            ),
        ),
    ]

    for p in plots:
        if not p.data_dict:
            continue
        print("\n" + "-" * 70)
        print(f"Generating {p.kind} comparison grid...")
        plot_sample_grid(
            p.data_dict,
            p.real_np,
            base_output,
            p.kind,
            all_mask_indexes,
            num_samples=args.num_real_facies,
            seed=args.manual_seed,
        )

    # ── 2. Embedding Plots ────────────────────────────────────────────────
    if not args.no_embeddings:
        print("\n" + "=" * 70)
        print("GENERATING EMBEDDING PLOTS")
        print("=" * 70)

        emb_data_map: dict[
            str,
            tuple[
                dict[str, list[np.ndarray]],
                dict[str, bool | int],
            ],
        ] = {
            "facies": (
                all_facies,
                {"rock_physics_only": False, "seismic_only": False},
            ),
            "ip": (
                all_ip,
                {
                    "rock_physics_only": True,
                    "seismic_only": False,
                    "channel_index": facies_ch,
                },
            ),
            "is": (
                all_is,
                {
                    "rock_physics_only": True,
                    "seismic_only": False,
                    "channel_index": facies_ch + 1,
                },
            ),
            "vp_vs": (
                all_vpvs,
                {
                    "rock_physics_only": True,
                    "seismic_only": False,
                    "channel_index": facies_ch + 2,
                },
            ),
            "seismic": (
                all_seismic,
                {"rock_physics_only": False, "seismic_only": True},
            ),
        }

        # Ensure vp_vs is included if rock_physics was requested
        effective_kinds: list[str] = []
        for k in emb_data_kinds:
            if k == "rock_physics":
                effective_kinds.append("ip")
                effective_kinds.append("is")
                effective_kinds.append("vp_vs")
            else:
                effective_kinds.append(k)

        for kind in effective_kinds:
            if kind not in emb_data_map or not emb_data_map[kind][0]:
                continue

            data_dict, emb_args = emb_data_map[kind]
            print(f"\n--- Embedding kind: {kind} ---")
            # RECOMPUTE shared embeddings for each kind to ensure manifolds
            # are specific to the feature space being analyzed.
            # Ensure boolean flags are strictly bool (not int) to satisfy
            # type expectations of compute_shared_embeddings.
            coerced_args = {
                k: (bool(v) if k in ("rock_physics_only", "seismic_only") else v)
                for k, v in emb_args.items()
            }

            # Explicitly cast boolean arguments to bool for type checking
            rock_physics_only = (
                bool(coerced_args.pop("rock_physics_only", False))
                if "rock_physics_only" in coerced_args
                else False
            )
            seismic_only = (
                bool(coerced_args.pop("seismic_only", False))
                if "seismic_only" in coerced_args
                else False
            )

            shared = compute_shared_embeddings(
                data_dict,
                _dataset,
                methods=emb_methods,
                rock_physics_only=rock_physics_only,
                seismic_only=seismic_only,
                **coerced_args,
            )

            for method in emb_methods:
                plot_method_all_variants(
                    shared,
                    method,
                    base_output,
                    args.num_iter,
                    kind,
                    all_mask_indexes,
                    args.embedding_per_facies,
                )

    total_elapsed = format_time(int(time.time() - total_start))
    print("\n" + "=" * 70)
    print(f"ALL EXPERIMENTS COMPLETE  ({total_elapsed})")
    print("=" * 70 + "\n")
    print(f"\nOutputs in: {base_output}")
    for ev in active_variants:
        print(f"  {ev.id}: {model_paths.get(ev.id, 'N/A')}")
