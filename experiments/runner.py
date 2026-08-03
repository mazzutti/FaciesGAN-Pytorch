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
from .generation import generate_variant, generate_uncertainty
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


def load_variant_options(model_path: str, args: ExperimentOptions, ev_value: Any) -> TrainingOptions:
    opts_path = Path(model_path) / "options.json"
    if opts_path.exists():
        import json
        try:
            with open(opts_path, encoding="utf-8") as f:
                opt_dict = json.load(f)
            v_opts = TrainingOptions()
            for k, v in opt_dict.items():
                if isinstance(v, list):
                    v = tuple(v)
                setattr(v_opts, k, v)
            v_opts.gpu_device = args.gpu_device
            v_opts.use_cpu = args.use_cpu
            return v_opts
        except Exception as e:
            print(f"Error loading options.json from {opts_path}: {e}. Falling back to default CLI args.")
    
    v_args = build_training_args(args, ev_value, model_path)
    return get_main_arguments().parse_args(
        v_args, namespace=TrainingOptions()
    )


def main() -> None:
    parser = get_arguments()
    args = parser.parse_args(namespace=ExperimentOptions())
    args.post_process()

    # Ensure num_train_pyramids is large enough to cover the selected uncertainty index
    if args.skip_training:
        args.num_train_pyramids = max(args.num_train_pyramids, args.uncertainty_index + 1)

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

    # Override scale configurations from the first variant's options.json if it exists
    opts_path = Path(first_model) / "options.json"
    if opts_path.exists():
        import json
        try:
            with open(opts_path, encoding="utf-8") as f:
                opt_dict = json.load(f)
            print(f"DEBUG OVERRIDE: loaded from {opts_path.resolve()}", flush=True)
            # Override scale parameters of _base_opts with those from options.json
            for param in ["stop_scale", "start_scale", "min_size", "max_size", "crop_size", "num_train_pyramids"]:
                if param in opt_dict:
                    print(f"  overriding {param}: {getattr(_base_opts, param)} -> {opt_dict[param]}", flush=True)
                    setattr(_base_opts, param, opt_dict[param])
        except Exception as e:
            logger.warning("Could not override base scale options from %s: %s", opts_path, e)

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
    # If the checkpoint seen_indices is incomplete (e.g. due to DDP gathering issues in old runs),
    # reconstruct the deterministic training set to compute the correct test complement.
    if _base_opts.num_train_pyramids < len(_dataset):
        temp_dataset = PyramidsDataset(_base_opts)
        reconstructed_seen = temp_dataset.select_equally_spaced(_base_opts.num_train_pyramids)
        if not seen_indices or set(seen_indices).issubset(set(reconstructed_seen)):
            seen_indices = reconstructed_seen
            print(
                f"[INFO] Reconstructed {len(seen_indices)} deterministic training indices for test complement calculation."
            )

    # Compute test indices as the complement of seen indices
    test_indices = sorted(list(set(range(len(_dataset))) - set(seen_indices)))
    if not test_indices:
        test_indices = sorted(list(range(len(_dataset))))
    print(f"\n[INFO] Computed test indices (complement of seen training indices): {test_indices}\n", flush=True)

    # Create a test-only subset for comparison grids and generation conditioning.
    # _dataset remains the FULL dataset (all 200 samples) for embedding computation.
    _test_dataset = PyramidsDataset(_base_opts)
    _test_dataset.batches = [_test_dataset.batches[i] for i in test_indices]
    _test_dataset.indices = _test_dataset.indices[torch.as_tensor(test_indices, dtype=torch.long)]
    _test_dataset._scale_data_cache.clear()

    # Pre-build pyramids for generation once
    wells_pyramid, seismic_pyramid = build_conditioning_pyramids(_base_opts)

    # Move pyramids to device
    wells_pyramid = device_manager.to_device(wells_pyramid)
    seismic_pyramid = device_manager.to_device(seismic_pyramid)

    print("DEBUG SHAPES:", flush=True)
    print(f"  _base_opts.num_train_pyramids: {_base_opts.num_train_pyramids}", flush=True)
    print(f"  _dataset size (full, for embeddings): {len(_dataset)}", flush=True)
    print(f"  _test_dataset size (test only, for generation/grids): {len(_test_dataset)}", flush=True)
    for k, v in wells_pyramid.items():
        print(f"  wells_pyramid[{k}]: {v.shape}", flush=True)
    for k, v in seismic_pyramid.items():
        print(f"  seismic_pyramid[{k}]: {v.shape}", flush=True)

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
            variant_opts = load_variant_options(model_path, args, ev.value)
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
                seen_indices=test_indices,
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
    real_tensor, _, _, real_seismic_tensor = _test_dataset.get_scale_data(-1)
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
            test_indices=test_indices,
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
                k: (
                    bool(v)
                    if k in ("rock_physics_only", "seismic_only", "zscore_rp")
                    else v
                )
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
                zscore_rp=args.zscore_rp,
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

    # ── 3. Uncertainty Analysis ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RUNNING SUBSURFACE UNCERTAINTY ANALYSIS (1000 REALIZATIONS)")
    print("=" * 70)

    import matplotlib.pyplot as plt
    from models.utils import calculate_channels

    sel_idx = args.uncertainty_index
    num_samples = args.uncertainty_samples

    print(f"Selected conditioning index: {sel_idx}", flush=True)
    print(f"Number of realizations per variant: {num_samples}", flush=True)

    # Validate that uncertainty_index exists in the test set. If not, pick the first test index.
    if sel_idx in test_indices:
        absolute_sel_idx = sel_idx
        print(f"Uncertainty index {sel_idx} is in the test set.", flush=True)
    else:
        absolute_sel_idx = test_indices[0]
        print(f"Uncertainty index {sel_idx} is NOT in the test set. Using first test index {absolute_sel_idx}.", flush=True)

    relative_sel_idx = test_indices.index(absolute_sel_idx)

    if relative_sel_idx < len(real_full_np):
        # Extract ground truth maps
        real_facies_gt = real_full_np[relative_sel_idx, ..., :facies_ch]  # shape (H, W, 3)
        real_ip_gt = real_full_np[relative_sel_idx, ..., facies_ch]       # shape (H, W)

        # real_seismic_tensor is (N_test, 1, H, W). Convert to numpy.
        real_seis_np = device_manager.to_numpy(real_seismic_tensor)
        real_seis_gt = real_seis_np[relative_sel_idx, 0, ...] if real_seis_np.size > 0 else np.zeros_like(real_ip_gt)

        # Also get well mask for indicators
        _, _, real_masks_all, _ = _test_dataset.get_scale_data(-1)
        real_masks_np = device_manager.to_numpy(real_masks_all)
        real_masks_gt = real_masks_np[relative_sel_idx, 0, ...] if real_masks_np.size > 0 else np.zeros_like(real_ip_gt)
        well_cols = np.where(np.sum(real_masks_gt, axis=0) > 0)[0]

        # Number of facies classes
        from config import DomainConfig
        num_classes = DomainConfig.NUM_FACIES

        # Generate and plot uncertainty for each active variant
        for ev in active_variants:
            name = ev.id
            model_path = model_paths[name]

            # Load variant options and channels
            v_opts = load_variant_options(model_path, args, ev.value)
            v_channels = calculate_channels(v_opts)

            # Build variant-specific pyramids matching the model's scale configuration
            print(f"\nBuilding variant-specific pyramids for: {name} ...", flush=True)
            v_wells_pyramid, v_seismic_pyramid = build_conditioning_pyramids(v_opts)
            v_wells_pyramid = device_manager.to_device(v_wells_pyramid)
            v_seismic_pyramid = device_manager.to_device(v_seismic_pyramid)

            print(f"Generating {num_samples} uncertainty realizations for variant: {name} ...", flush=True)
            gen_facies, gen_ip, gen_seis = generate_uncertainty(
                model_path=model_path,
                opts=v_opts,
                how_many=num_samples,
                wells_pyramid=tuple(v_wells_pyramid.values()),
                seismic_pyramid=tuple(v_seismic_pyramid.values()),
                channels=v_channels,
                selected_idx=absolute_sel_idx,
            )

            if len(gen_facies) > 0:
                print(f"Computing uncertainty maps for {name} ...", flush=True)

                # ── Convert all generated facies to class indices ──
                real_facies_idx = utils.rgb_to_facies(real_facies_gt)  # (H, W)

                gen_facies_idxs = np.stack(
                    [utils.rgb_to_facies(gen_facies[i]) for i in range(len(gen_facies))],
                    axis=0,
                )  # (N, H, W)

                # ── 1. Facies: Mode (most frequent class) ──
                from scipy import stats as sp_stats
                facies_mode_result = sp_stats.mode(gen_facies_idxs, axis=0, keepdims=False)
                facies_mode = facies_mode_result.mode.astype(np.int32)  # (H, W)
                facies_mode_rgb = utils.facies_to_rgb(facies_mode).transpose(1, 2, 0)  # (H, W, 3)

                # ── 2. Facies: Shannon Entropy ──
                # Per-pixel class probability: p_k(i,j) = count_k / N
                H, W = real_facies_idx.shape
                class_counts = np.zeros((num_classes, H, W), dtype=np.float32)
                for k in range(num_classes):
                    class_counts[k] = np.mean(gen_facies_idxs == k, axis=0)
                # H = -Σ p_k log(p_k), with 0*log(0) = 0
                eps = 1e-12
                log_probs = np.log(class_counts + eps)
                facies_entropy = -np.sum(class_counts * log_probs, axis=0)  # (H, W)
                # Normalize to [0, 1] by dividing by max possible entropy log(num_classes)
                max_entropy = np.log(num_classes)
                facies_entropy_norm = facies_entropy / max_entropy

                # ── 3. Ip: Mean and Std Dev ──
                gen_ip_arr = np.squeeze(gen_ip)  # (N, H, W)
                ip_mean = np.mean(gen_ip_arr, axis=0)   # (H, W)
                ip_std = np.std(gen_ip_arr, axis=0)      # (H, W)

                # ── 4. Seismic: Mean and Std Dev ──
                gen_seis_arr = np.squeeze(gen_seis)  # (N, H, W)
                seis_mean = np.mean(gen_seis_arr, axis=0)   # (H, W)
                seis_std = np.std(gen_seis_arr, axis=0)      # (H, W)

                # ── Plot 3×3 figure ──
                fig, axes = plt.subplots(3, 3, figsize=(16, 13), squeeze=False)
                fig.patch.set_facecolor("#151b26")

                row_labels = ["Ground Truth", "Mean Realization", "Uncertainty"]
                col_labels = ["Facies", "Acoustic Impedance (Ip)", "Synthetic Seismic"]

                for c, label in enumerate(col_labels):
                    axes[0][c].set_title(label, fontsize=13, fontweight="bold", color="#f0f4f9", pad=10)
                for r, label in enumerate(row_labels):
                    axes[r][0].set_ylabel(label, fontsize=12, fontweight="bold", color="#8c9eb5", rotation=90, labelpad=15)

                # ── Row 0: Ground Truth ──
                gt_facies_rgb = utils.facies_to_rgb(real_facies_idx).transpose(1, 2, 0)
                axes[0][0].imshow(gt_facies_rgb, aspect="auto", interpolation="nearest")

                axes[0][1].imshow(real_ip_gt, aspect="auto", cmap="magma", interpolation="nearest")

                seis_plot = real_seis_gt - np.mean(real_seis_gt)
                p_lo = float(np.percentile(seis_plot, 2))
                p_hi = float(np.percentile(seis_plot, 98))
                max_abs = max(abs(p_lo), abs(p_hi), 1e-6)
                axes[0][2].imshow(seis_plot, aspect="auto", cmap="RdBu", vmin=-max_abs, vmax=max_abs, interpolation="nearest")

                # ── Row 1: Mean Realization ──
                axes[1][0].imshow(facies_mode_rgb, aspect="auto", interpolation="nearest")

                axes[1][1].imshow(ip_mean, aspect="auto", cmap="magma", interpolation="nearest")

                seis_mean_plot = seis_mean - np.mean(seis_mean)
                seis_mean_lo = float(np.percentile(seis_mean_plot, 2))
                seis_mean_hi = float(np.percentile(seis_mean_plot, 98))
                seis_mean_abs = max(abs(seis_mean_lo), abs(seis_mean_hi), 1e-6)
                axes[1][2].imshow(seis_mean_plot, aspect="auto", cmap="RdBu", vmin=-seis_mean_abs, vmax=seis_mean_abs, interpolation="nearest")

                # ── Row 2: Uncertainty ──
                im_ent = axes[2][0].imshow(facies_entropy_norm, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
                cb_ent = fig.colorbar(im_ent, ax=axes[2][0], fraction=0.046, pad=0.04)
                cb_ent.set_label("Normalized Entropy", color="#f0f4f9", fontsize=9)
                cb_ent.ax.yaxis.set_tick_params(color="#f0f4f9")
                plt.setp(cb_ent.ax.yaxis.get_ticklabels(), color="#f0f4f9")

                im_ip = axes[2][1].imshow(ip_std, aspect="auto", cmap="inferno", interpolation="nearest")
                cb_ip = fig.colorbar(im_ip, ax=axes[2][1], fraction=0.046, pad=0.04)
                cb_ip.set_label("Std Dev", color="#f0f4f9", fontsize=9)
                cb_ip.ax.yaxis.set_tick_params(color="#f0f4f9")
                plt.setp(cb_ip.ax.yaxis.get_ticklabels(), color="#f0f4f9")

                im_seis = axes[2][2].imshow(seis_std, aspect="auto", cmap="cividis", interpolation="nearest")
                cb_seis = fig.colorbar(im_seis, ax=axes[2][2], fraction=0.046, pad=0.04)
                cb_seis.set_label("Std Dev", color="#f0f4f9", fontsize=9)
                cb_seis.ax.yaxis.set_tick_params(color="#f0f4f9")
                plt.setp(cb_seis.ax.yaxis.get_ticklabels(), color="#f0f4f9")

                # ── Subtitle labels for Row 2 ──
                axes[2][0].set_title("Shannon Entropy", fontsize=10, color="#c0c8d4", style="italic", pad=4)
                axes[2][1].set_title("Ip Std Deviation", fontsize=10, color="#c0c8d4", style="italic", pad=4)
                axes[2][2].set_title("Seismic Std Deviation", fontsize=10, color="#c0c8d4", style="italic", pad=4)

                # Draw well indicator markers if applicable
                if ev.value.use_wells and len(well_cols) > 0:
                    for col in well_cols:
                        for r in range(3):
                            axes[r][0].plot(col, 5, marker='v', color='red', markersize=5)

                # Clean axis ticks and spines
                for r in range(3):
                    for c in range(3):
                        axes[r][c].set_xticks([])
                        axes[r][c].set_yticks([])
                        for spine in axes[r][c].spines.values():
                            spine.set_visible(False)

                fig.suptitle(
                    f"Uncertainty Overview — {ev.value.label}  (Index {absolute_sel_idx}, {num_samples} Realizations)",
                    fontsize=15, color="#f0f4f9",
                )
                fig.tight_layout()

                out_path = Path(base_output) / name / "uncertainty_overview.png"

                try:
                    from report.utils import save_dual_theme_plot
                    save_dual_theme_plot(fig, axes, out_path, dpi=150)
                except Exception as e:
                    logger.warning(f"Failed to save dual theme plot: {e}")
                    plt.savefig(out_path, dpi=150, bbox_inches="tight")

                plt.close(fig)
                print(f"Saved uncertainty map to: {out_path}", flush=True)

        # Save config specifying the uncertainty parameters used
        try:
            import json
            cfg_path = Path(base_output) / "uncertainty_config.json"
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump({"uncertainty_index": absolute_sel_idx, "uncertainty_samples": num_samples}, f, indent=4)
            print(f"Saved uncertainty config to: {cfg_path}", flush=True)
        except Exception as e:
            logger.warning(f"Failed to save uncertainty config: {e}")
    else:
        print(f"Warning: Selected index {absolute_sel_idx} is out of bounds for the dataset of size {len(real_full_np)}.")

    total_elapsed = format_time(int(time.time() - total_start))
    print("\n" + "=" * 70)
    print(f"ALL EXPERIMENTS COMPLETE  ({total_elapsed})")
    print("=" * 70 + "\n")
    print(f"\nOutputs in: {base_output}")
    for ev in active_variants:
        print(f"  {ev.id}: {model_paths.get(ev.id, 'N/A')}")


if __name__ == "__main__":
    main()
