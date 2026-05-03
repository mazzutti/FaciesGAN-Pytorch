"""Main execution runner for experiments."""

import os
import time
from dataclasses import dataclass

import numpy as np
import torch

import utils
from datasets.data_files import DataFiles
from datasets.dataset import TorchPyramidsDataset
from datasets.utils import build_conditioning_pyramids
from log import format_time
from main import get_arguments as get_main_arguments
from options import TrainingOptions

from .cli import build_training_args, get_arguments
from .constants import ExperimentVariant, VariantConfig
from .embeddings import (
    ExperimentCache,
    compute_shared_embeddings,
    load_shared_embeddings,
    save_shared_embeddings,
)
from .generation import generate_variant
from .plotting import plot_method_all_variants, plot_sample_grid
from .training import find_last_completed_scale, read_completed_epochs, train_variant


@dataclass
class PlotData:
    """Encapsulates data required for a specific comparison plot."""

    kind: str
    data_dict: dict[str, list[np.ndarray]]
    real_np: np.ndarray | None


def main() -> None:
    parser = get_arguments()
    args = parser.parse_args()

    device = utils.resolve_device(args.gpu_device)
    base_output = args.output_path
    nproc = args.nproc_per_node

    # Map variant name -> model path (either from training or --model-paths)
    model_paths: dict[str, str] = {}
    any_retrained = False

    total_start = time.time()

    if args.skip_training:
        if not args.model_paths:
            parser.error(
                "--skip-training requires --model-paths with 4 paths "
                "(wells_seismic, wells_only, seismic_only, unconditional)."
            )
        for variant, path in zip(ExperimentVariant, args.model_paths):
            if not os.path.isdir(path):
                parser.error(f"Model path does not exist: {path}")
            model_paths[variant.id] = path
        print("Skipping training, using provided model paths.")
    else:
        # ── Train all 4 variants ──
        print("=" * 70)
        print("FACIESGAN CONDITIONING-ABLATION EXPERIMENTS")
        print("=" * 70)
        print(f"Device: {device} (DDP with {nproc} GPUs)")
        print(f"compile_backend: {'ON' if args.compile_backend else 'OFF'}")
        print(f"Variants: {', '.join(v.id for v in ExperimentVariant)}")
        print(f"Output:   {base_output}")
        print("=" * 70 + "\n")

        for variant in ExperimentVariant:
            name = variant.id
            v_config: VariantConfig = variant.value
            variant_output = os.path.join(base_output, name)
            os.makedirs(variant_output, exist_ok=True)
            wells_flag = "ON" if v_config.use_wells else "OFF"
            seismic_flag = "ON" if v_config.use_seismic else "OFF"

            # Check for existing checkpoint
            last_done = find_last_completed_scale(variant_output)
            start_epoch_arg = getattr(args, "start_epoch", 0)

            # 1. Determine base start point
            if last_done == -1:
                # No existing training found
                effective_start_scale = 0
                effective_start_epoch = start_epoch_arg
            else:
                # Training exists up to last_done. Check its progress.
                completed_in_last = read_completed_epochs(variant_output, last_done)
                if completed_in_last >= args.num_iter:
                    # Current scale is finished.
                    if last_done >= args.stop_scale:
                        # ALL scales finished
                        effective_start_scale = last_done
                        effective_start_epoch = completed_in_last
                    else:
                        # Move to next scale
                        effective_start_scale = last_done + 1
                        effective_start_epoch = 0
                else:
                    # Current scale is in progress.
                    effective_start_scale = last_done
                    effective_start_epoch = max(start_epoch_arg, completed_in_last)

            # 2. Skip if fully trained
            if (
                effective_start_scale >= args.stop_scale
                and effective_start_epoch >= args.num_iter
            ):
                model_paths[name] = variant_output
                print(f"\n{'─' * 60}")
                print(
                    f"Skipping variant: {name} "
                    f"(fully trained — scale {effective_start_scale}, epoch {effective_start_epoch}/{args.num_iter})"
                )
                print(f"  {variant_output}")
                print(f"{'─' * 60}")
                continue

            # 3. Resume logic
            resume_scale = effective_start_scale
            resume_epoch = effective_start_epoch

            print(f"\n{'─' * 60}")
            print(f"Training variant: {name}")
            print(f"  wells={wells_flag}  seismic={seismic_flag}")

            if resume_epoch > 0:
                print(f"  Resuming from scale {resume_scale}, epoch {resume_epoch}")
            elif resume_scale > 0:
                print(
                    f"  Starting from scale {resume_scale} (scales 0-{resume_scale-1} already done)"
                )
            else:
                print("  Starting training from scratch (scale 0, epoch 0)")

            print(
                f"  DDP: {nproc} GPUs  compile_backend: {'ON' if args.compile_backend else 'OFF'}"
            )
            print(f"{'─' * 60}")

            variant_args = build_training_args(
                args, v_config, variant_output, start_scale=resume_scale
            )

            # Inject start_epoch
            try:
                idx = variant_args.index("--start-epoch")
                variant_args[idx + 1] = str(resume_epoch)
            except ValueError:
                if resume_epoch > 0:
                    variant_args.extend(["--start-epoch", str(resume_epoch)])

            variant_start = time.time()
            train_variant(variant_args, nproc)
            any_retrained = True
            elapsed = format_time(int(time.time() - variant_start))

            # --output-fullpath places artifacts directly in variant_output
            model_paths[name] = variant_output
            print(f"  Training complete ({elapsed}) -> {variant_output}")

    # ── Load base options & dataset (needed for plots and embeddings) ──
    first_model = model_paths[list(ExperimentVariant)[0].id]
    _base_args = build_training_args(args, list(ExperimentVariant)[0].value, first_model)
    _base_opts = get_main_arguments().parse_args(
        _base_args, namespace=TrainingOptions()
    )
    _base_opts.rec = False
    _dataset = TorchPyramidsDataset(_base_opts)

    # Pre-build pyramids for generation once
    wells_pyramid, seismic_pyramid = build_conditioning_pyramids(_base_opts)

    # Move pyramids to device
    wells_pyramid = {k: v.to(device) for k, v in wells_pyramid.items()}
    seismic_pyramid = {k: v.to(device) for k, v in seismic_pyramid.items()}

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
        all_seismic = cached.all_seismic
        print(f"Loaded cached embeddings for epoch {args.num_iter}")
    else:
        # ── Generate facies (and rock_physics) from all trained models ──
        print(f"\n{'=' * 70}")
        print("GENERATING FACIES FROM TRAINED MODELS")
        print(f"{'=' * 70}\n")

        all_facies: dict[str, list[np.ndarray]] = {}
        all_ip: dict[str, list[np.ndarray]] = {}
        all_seismic: dict[str, list[np.ndarray]] = {}
        all_mask_indexes: dict[str, torch.Tensor] = {}

        from models.utils import calculate_noise_channels

        for variant in ExperimentVariant:
            name = variant.id
            v_config = variant.value
            model_path = model_paths[name]
            gen_output = os.path.join(base_output, name, "generated")

            print(f"Generating from variant: {name}")
            print(f"  model: {model_path}")

            # Calculate noise channels for this specific variant configuration
            v_args = build_training_args(args, v_config, model_path)
            variant_opts = get_main_arguments().parse_args(
                v_args, namespace=TrainingOptions()
            )
            v_noise_channels = calculate_noise_channels(variant_opts)

            variant_facies, variant_ip, variant_seismic, variant_mi = generate_variant(
                model_path=model_path,
                opts=_base_opts,
                how_many=args.how_many,
                device=device,
                wells_pyramid=tuple(wells_pyramid.values()),
                seismic_pyramid=tuple(seismic_pyramid.values()),
                noise_channels=v_noise_channels,
                gen_output=gen_output,
            )
            all_facies[name] = variant_facies
            if variant_ip:
                all_ip[name] = variant_ip
            if variant_seismic:
                all_seismic[name] = variant_seismic
            all_mask_indexes[name] = variant_mi

        if not args.no_embeddings:
            # ── Shared embeddings for all plots ──
            emb_methods: list[str] = args.embedding_methods
            print(
                f"\nComputing shared embeddings "
                f"({', '.join(m.upper() for m in emb_methods)}) ...",
                flush=True,
            )
            shared = compute_shared_embeddings(
                all_facies, _dataset, methods=emb_methods
            )

            # Persist for future resume
            cache = ExperimentCache(
                shared=shared,
                all_facies=all_facies,
                all_mask_indexes=all_mask_indexes,
                all_ip=all_ip,
                all_seismic=all_seismic,
            )
            save_shared_embeddings(
                cache,
                base_output,
                args.num_iter,
            )
        else:
            shared = {}

    # ── Resolve effective embedding method list ────────────────────────────
    emb_methods = list(
        getattr(args, "embedding_methods", ["isomap", "mds", "tsne", "umap"])
    )
    emb_data_kinds: list[str] = list(
        getattr(args, "embedding_data", [DataFiles.FACIES.name.lower(), "rock_physics"])
    )

    # ── 1. Comparison Grids ────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("GENERATING COMPARISON GRIDS")
    print(f"{'=' * 70}")

    import utils as _utils

    # Get real data once
    real_tensor, _, _, real_seismic_tensor = _dataset.get_scale_data(-1)
    real_full_np = _utils.torch2np(real_tensor, denormalize=True)
    facies_ch = _base_opts.num_facies_classes

    # Define kinds to plot
    plots = [
        PlotData(DataFiles.FACIES.name.lower(), all_facies, real_full_np[..., :facies_ch]),
        PlotData(DataFiles.Ip.name.lower(), all_ip, real_full_np[..., facies_ch] if all_ip else None),
        PlotData(
            DataFiles.SEISMIC.name.lower(),
            all_seismic,
            (
                np.transpose(real_seismic_tensor.cpu().numpy(), (0, 2, 3, 1))
                if all_seismic
                else None
            ),
        ),
    ]

    for p in plots:
        if not p.data_dict:
            continue
        print(f"\n{'-' * 70}")
        print(f"Generating {p.kind} comparison grid...")
        plot_sample_grid(p.data_dict, p.real_np, base_output, p.kind)

    # ── 2. Embedding Plots ────────────────────────────────────────────────
    if not args.no_embeddings:
        print(f"\n{'=' * 70}")
        print("GENERATING EMBEDDING PLOTS")
        print(f"{'=' * 70}")

        emb_data_map = {
            DataFiles.FACIES.name.lower(): (all_facies, {"rock_physics_only": False, "seismic_only": False}),
            "rock_physics": (
                all_ip,
                {"rock_physics_only": True, "seismic_only": False},
            ),
            DataFiles.SEISMIC.name.lower(): (
                all_seismic,
                {"rock_physics_only": False, "seismic_only": True},
            ),
        }

        for kind in emb_data_kinds:
            if kind not in emb_data_map or not emb_data_map[kind][0]:
                continue

            data_dict, emb_args = emb_data_map[kind]
            plot_kind = DataFiles.Ip.name.lower() if kind == "rock_physics" else kind

            print(f"\n{'-' * 70}")
            print(f"Computing shared {kind} embeddings for plots...", flush=True)

            # Use shared embeddings if available for facies, otherwise compute
            if kind == DataFiles.FACIES.name.lower() and shared:
                current_shared = shared
            else:
                current_shared = compute_shared_embeddings(
                    data_dict, _dataset, methods=emb_methods, **emb_args
                )

            for method in emb_methods:
                plot_method_all_variants(
                    current_shared,
                    method,
                    base_output,
                    args.num_iter,
                    plot_kind,
                    all_mask_indexes=all_mask_indexes,
                    embedding_per_facies=args.embedding_per_facies,
                )

    total_elapsed = format_time(int(time.time() - total_start))
    print(f"\n{'=' * 70}")
    print(f"ALL EXPERIMENTS COMPLETE  ({total_elapsed})")
    print(f"{'=' * 70}\n")
    print(f"\nOutputs in: {base_output}")
    for variant in ExperimentVariant:
        print(f"  {variant.id}: {model_paths.get(variant.id, 'N/A')}")
