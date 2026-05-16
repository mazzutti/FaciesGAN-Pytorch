from argparse import ArgumentParser
from typing import Any
import cli_shared
from enums import VariantConfig


def get_arguments() -> ArgumentParser:
    """Build argument parser for the experiments runner.

    All unrecognised arguments are forwarded to each training run.
    """
    parser = ArgumentParser(
        description="Run conditioning-ablation experiments for FaciesGAN.",
    )

    # Workspace and hardware
    cli_shared.add_device_args(parser)

    # 1. Experiment-specific options
    parser.add_argument(
        "--how-many",
        type=int,
        help="Number of facies to generate per variant (default: 2000).",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and only run generation using existing model paths.",
    )
    parser.add_argument(
        "--model-paths",
        nargs=4,
        metavar=("WELLS_SEISMIC", "WELLS_ONLY", "SEISMIC_ONLY", "UNCONDITIONAL"),
        help=(
            "Explicit model paths for generation-only mode (requires "
            "--skip-training). Provide 4 paths in order."
        ),
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=2,
        help="Number of GPUs for DDP training (default: 2).",
    )

    # 2. Shared training hyper-parameters with experiment-specific defaults
    # Note: we use different defaults for experiments than for standalone main.py
    # to ensure faster runs and larger batches by default.
    experiment_defaults: dict[str, Any] = {
        "num_iter": 2000,
        "num_train_pyramids": 10,
        "batch_size": 50,
        "num_workers": 1,
        "num_parallel_scales": 7,
        "lr_decay": 999999,
        "scale0_noise_amp": 1.5,
        "min_noise_amp": 0.3,
        "gp_interval": 8,
        "gradient_loss_penalty": 0.1,
    }

    cli_shared.add_io_args(parser, defaults=experiment_defaults)
    cli_shared.add_network_args(parser)
    cli_shared.add_optimization_args(parser, defaults=experiment_defaults)
    cli_shared.add_pyramid_args(parser, defaults=experiment_defaults)
    cli_shared.add_physics_args(parser)
    cli_shared.add_gan_loss_args(parser, defaults=experiment_defaults)

    # 3. Experiment-specific scale 0 overrides
    cli_shared.add_scale0_args(parser)

    # 4. Latent space / Metrics options
    _add_latent_space_args(parser)

    # 5. UI/Logging overrides
    cli_shared.add_runtime_args(parser)

    return parser


from options import ExperimentOptions


def build_training_args(
    args: ExperimentOptions, variant: VariantConfig, output: str, start_scale: int = 0
) -> list[str]:
    """Compose the command-line arguments for a single variant training run."""
    cmd: list[str] = [
        "--input-path",
        args.input_path,
        "--output-fullpath",
        output,
        "--num-iter",
        str(args.num_iter),
        "--num-train-pyramids",
        str(args.num_train_pyramids),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--num-parallel-scales",
        str(args.num_parallel_scales),
        "--stop-scale",
        str(args.stop_scale),
        "--start-scale",
        str(start_scale),
        "--checkpoint-interval",
        str(args.checkpoint_interval),
        "--save-interval",
        str(args.save_interval),
        "--discriminator-steps",
        str(args.discriminator_steps),
        "--scale0-disc-steps-multiplier",
        str(args.scale0_disc_steps_multiplier),
        "--scale0-loss-multiplier",
        str(args.scale0_loss_multiplier),
        "--scale0-r1-gamma",
        str(args.scale0_r1_gamma),
        "--scale0-disc-grad-clip",
        str(args.scale0_disc_grad_clip),
        "--scale0-gp-alpha",
        str(args.scale0_gp_alpha),
        "--generator-steps",
        str(args.generator_steps),
        "--rec-facies-loss-penalty",
        str(args.rec_facies_loss_penalty),
        "--gamma",
        str(args.gamma),
        "--lr-g",
        str(args.lr_g),
        "--lr-d",
        str(args.lr_d),
        "--lr-decay",
        str(args.lr_decay),
        "--lr-decay-unit",
        str(args.lr_decay_unit),
        "--lr-patience",
        str(args.lr_patience),
        "--lr-min",
        str(args.lr_min),
        "--lr-smoothing-alpha",
        str(args.lr_smoothing_alpha),
        "--lr-g-factor",
        str(args.lr_g_factor),
        "--scale0-noise-amp",
        str(args.scale0_noise_amp),
        "--scale0-disc-lr-factor",
        str(args.scale0_disc_lr_factor),
        "--min-noise-amp",
        str(args.min_noise_amp),
        "--num-diversity-samples",
        str(args.num_diversity_samples),
        "--diversity-loss-penalty",
        str(args.diversity_loss_penalty),
        "--adversarial-loss-penalty",
        str(args.adversarial_loss_penalty),
        "--well-loss-penalty",
        str(args.well_loss_penalty),
        "--grad-clip-norm",
        str(args.grad_clip_norm),
        "--gradient-loss-penalty",
        str(args.gradient_loss_penalty),
        "--gp-interval",
        str(args.gp_interval),
        "--num-real-facies",
        str(args.num_real_facies),
        "--seismic-stretch-percentile",
        str(args.seismic_stretch_percentile),
        "--num-features",
        str(args.num_feature),
        "--min-num-features",
        str(args.min_num_feature),
        "--kernel-size",
        str(args.kernel_size),
        "--num-layers",
        str(args.num_layer),
        "--stride",
        str(args.stride),
        "--padding-size",
        str(args.padding_size),
        "--beta1",
        str(args.beta1),
        "--amp-dtype",
        str(args.amp_dtype),
        "--normalization-range",
        str(args.normalization_range[0]),
        str(args.normalization_range[1]),
    ]

    if args.manual_seed is not None:
        cmd.extend(["--manual-seed", str(args.manual_seed)])
    if args.scale0_padding_size is not None:
        cmd.extend(["--scale0-padding-size", str(args.scale0_padding_size)])

    if not args.shuffle:
        cmd.append("--no-shuffle")
    if not args.enable_tensorboard:
        cmd.append("--no-tensorboard")
    if not args.enable_plot_outputs:
        cmd.append("--no-plot-outputs")
    if not args.compile_backend:
        cmd.append("--no-compile")
    if args.gradient_checkpointing:
        cmd.append("--gradient-checkpoint")

    # Conditioning flags
    if variant.use_wells:
        cmd.append("--use-wells")
    if variant.use_seismic:
        cmd.append("--use-seismic")

    # Rock physics flags
    if args.use_rock_physics:
        cmd.append("--use-rock-physics")
        if args.vp_vs_robust_range:
            cmd.append("--vp-vs-robust-range")
            cmd.extend(
                [
                    "--vp-vs-robust-percentiles",
                    str(args.vp_vs_robust_percentiles[0]),
                    str(args.vp_vs_robust_percentiles[1]),
                ]
            )
        cmd.extend(
            [
                "--rec-facies-loss-penalty",
                str(args.rec_facies_loss_penalty),
            ]
        )
        cmd.extend(
            [
                "--rec-rock-physics-loss-penalty",
                str(args.rec_rock_physics_loss_penalty),
            ]
        )
        cmd.extend(["--tv-loss-penalty", str(args.tv_loss_penalty)])
        cmd.extend(["--elastic-loss-penalty", str(args.elastic_loss_penalty)])
        cmd.extend(["--seismic-loss-penalty", str(args.seismic_loss_penalty)])
        cmd.extend(["--dz-pixel", str(args.dz_pixel)])
        cmd.extend(["--wavelet-f-peak", str(args.wavelet_f_peak)])
        cmd.extend(["--wavelet-dt", str(args.wavelet_dt)])

    if args.use_gradnorm:
        cmd.append("--use-gradnorm")
        cmd.extend(["--gradnorm-interval", str(args.gradnorm_interval)])
        cmd.extend(["--gradnorm-alpha", str(args.gradnorm_alpha)])
        cmd.extend(["--gradnorm-lr", str(args.gradnorm_lr)])

    return cmd


def _add_latent_space_args(parser: ArgumentParser) -> None:
    """Add latent space and manifold learning arguments to the parser."""
    from enums import EmbeddingMethod

    parser.add_argument(
        "--embedding-methods",
        nargs="+",
        default=[m.value for m in EmbeddingMethod],
        help=f"Dimensionality reduction methods to use (default: {' '.join(m.value for m in EmbeddingMethod)}).",
    )
    parser.add_argument(
        "--embedding-data",
        nargs="+",
        default=["facies", "rock_physics"],
        help="Data types to compute embeddings for (default: facies rock_physics).",
    )
    parser.add_argument(
        "--embedding-per-facies",
        action="store_true",
        help="Generate embedding plots for each unique conditioning crossline.",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Disable all latent space visualization (plots only comparison grids).",
    )
