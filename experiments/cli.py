"""CLI argument parsing for experiments."""

from argparse import ArgumentParser, Namespace

from config import PhysicsConfig
from constants import EmbeddingMethod, VariantConfig
from options import NORMALIZATION_RANGE


def get_arguments() -> ArgumentParser:
    """Build argument parser for the experiments runner.

    All unrecognised arguments are forwarded to each training run.
    """
    parser = ArgumentParser(
        description="Run conditioning-ablation experiments for FaciesGAN.",
    )
    # Required paths
    parser.add_argument(
        "--input-path", required=True, help="Path to the dataset root directory."
    )
    parser.add_argument(
        "--output-path",
        default="outputs/experiments",
        help="Base output directory for all experiment runs.",
    )

    # Generation options (applied after training)
    parser.add_argument(
        "--how-many",
        type=int,
        default=2000,
        help="Number of facies to generate per variant (default: 2000).",
    )

    # Convenience: allow skipping training if models already exist
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

    # Forwarded training hyper-parameters with sensible defaults
    parser.add_argument(
        "--num-iter",
        type=int,
        default=2000,
        help="number of full dataset passes (each pass shuffles independently)",
    )
    parser.add_argument("--num-train-pyramids", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-parallel-scales", type=int, default=7)
    parser.add_argument("--stop-scale", type=int, default=6)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1,
        help="Interval (in epochs) between saving training state checkpoints (default: 1).",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="Interval (in epochs) between saving generated visualizations (default: 100).",
    )
    parser.add_argument("--discriminator-steps", type=int, default=3)
    parser.add_argument(
        "--scale0-disc-steps-multiplier",
        type=int,
        default=1,
        help="Extra D-step multiplier for scale 0 only (default: 1).",
    )
    parser.add_argument(
        "--scale0-loss-multiplier",
        type=float,
        default=1.0,
        help="Extra loss multiplier for rec and rock_physics at scale 0 (default: 1.0).",
    )
    parser.add_argument(
        "--scale0-padding-size",
        type=int,
        default=None,
        dest="scale0_padding_size",
        help=(
            "Discriminator padding size override for scale 0 only. "
            "None (default) means use --padding-size globally."
        ),
    )
    parser.add_argument(
        "--scale0-r1-gamma",
        type=float,
        default=0.0,
        dest="scale0_r1_gamma",
        help=(
            "R1 gradient penalty weight for scale 0 discriminator. "
            "0.0 = disabled (default). Recommended: 10.0."
        ),
    )
    parser.add_argument(
        "--scale0-disc-grad-clip",
        type=float,
        default=0.0,
        dest="scale0_disc_grad_clip",
        help=(
            "Gradient clip norm for scale 0 discriminator parameters. "
            "0.0 = disabled (default). Recommended: 25.0."
        ),
    )
    parser.add_argument(
        "--scale0-gp-alpha",
        type=float,
        default=0.0,
        dest="scale0_gp_alpha",
        help=(
            "GP alpha override for scale 0 discriminator. "
            "0.0 = use global --gradient-loss-penalty (default). "
            "E.g. 50.0 for stronger Lipschitz pressure at scale 0."
        ),
    )
    parser.add_argument(
        "--scale0-disc-lr-factor",
        type=float,
        default=1.0,
        dest="scale0_disc_lr_factor",
        help=(
            "Learning-rate multiplier for the scale 0 discriminator. "
            "1.0 = same as global lr_d (default). E.g. 0.2 to slow D at s0."
        ),
    )
    parser.add_argument("--generator-steps", type=int, default=3)
    parser.add_argument(
        "--rec-facies-loss-penalty",
        type=float,
        dest="rec_facies_loss_penalty",
        default=10.0,
    )
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--lr-g", type=float, default=5e-4)
    parser.add_argument("--lr-d", type=float, default=5e-4)
    parser.add_argument("--lr-decay", type=int, default=999999)
    parser.add_argument(
        "--lr-decay-unit",
        type=str,
        choices=["epoch", "step", "batch"],
        default="epoch",
        help="unit for --lr-decay: 'epoch' resets per batch, 'step' uses global steps, 'batch' decays every N dataset batches (default: epoch)",
    )
    parser.add_argument("--lr-patience", type=int, default=400)
    parser.add_argument("--lr-min", type=float, default=1e-4)
    parser.add_argument("--lr-smoothing-alpha", type=float, default=0.95)
    parser.add_argument("--lr-g-factor", type=float, default=0.8)
    parser.add_argument("--scale0-noise-amp", type=float, default=1.5)
    parser.add_argument("--min-noise-amp", type=float, default=0.3)
    parser.add_argument("--num-diversity-samples", type=int, default=3)
    parser.add_argument("--diversity-loss-penalty", type=float, default=1.0)
    parser.add_argument("--adversarial-loss-penalty", type=float, default=1.0)
    parser.add_argument("--well-loss-penalty", type=float, default=10.0)
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="max gradient norm for generator clipping (default: 1.0; set to 0 to disable)",
    )
    parser.add_argument(
        "--gradient-loss-penalty",
        type=float,
        default=0.1,
        help="Gradient penalty weight (default: 0.1).",
    )
    parser.add_argument(
        "--gp-interval",
        type=int,
        default=8,
        help="Compute gradient penalty every N discriminator steps (default: 8).",
    )

    parser.add_argument("--manual-seed", type=int, default=None)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=2,
        help="Number of GPUs for DDP training (default: 2).",
    )
    parser.add_argument(
        "--no-shuffle", action="store_true", help="disable dataset shuffling"
    )
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument(
        "--no-plot-outputs",
        action="store_true",
        help="Disable PNG sample plots during training.",
    )
    parser.add_argument(
        "--seismic-stretch-percentile",
        type=int,
        choices=[95, 98, 99],
        default=98,
        help=(
            "Percentile for TensorBoard seismic contrast stretch (display-only). "
            "Allowed: 95, 98, 99 (default: 98)."
        ),
    )
    parser.add_argument(
        "--rec-skip-per-scale",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Skip rec_facies loss for scale s during the first s*N epochs in parallel "
            "training. Default: 0 (disabled)."
        ),
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile for the generator and discriminator.",
    )
    parser.add_argument(
        "--compile-backend",
        action="store_true",
        help="Enable torch.compile for the generator and discriminator.",
    )
    parser.add_argument(
        "--no-triton-kernels",
        action="store_true",
        help="Disable custom Triton kernels and use PyTorch fallbacks.",
    )
    parser.add_argument(
        "--triton-kernels",
        action="store_true",
        help="Enable custom Triton kernels (enabled by default).",
    )
    parser.add_argument(
        "--use-wells",
        action="store_true",
        help="Accept the conditioning flag used by main.py (retained for compatibility).",
    )
    parser.add_argument(
        "--use-seismic",
        action="store_true",
        help="Accept the conditioning flag used by main.py (retained for compatibility).",
    )
    parser.add_argument(
        "--use-rock-physics",
        action="store_true",
        dest="use_rock_physics",
        help="Train with rock physics volumes (Ip, Is, Vp/Vs) as additional output channels.",
    )
    parser.add_argument(
        "--vp-vs-robust-range",
        action="store_true",
        dest="vp_vs_robust_range",
        help="Use percentile-based robust normalization range for VP/VS pyramids.",
    )
    parser.add_argument(
        "--vp-vs-robust-percentiles",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        dest="vp_vs_robust_percentiles",
        default=(1.0, 99.0),
        help="Percentiles [LOW HIGH] used when --vp-vs-robust-range is enabled (default: 1 99).",
    )
    parser.add_argument(
        "--rock-physics-loss-penalty",
        type=float,
        dest="rock_physics_loss_penalty",
        default=1.0,
        help="Extra loss multiplier for rock physics reconstruction (default: 1.0).",
    )
    parser.add_argument(
        "--rec-rock-physics-loss-penalty",
        type=float,
        dest="rec_rock_physics_loss_penalty",
        default=1.0,
        help="Extra loss multiplier for rock-physics reconstruction on the generated volume (default: 1.0).",
    )
    parser.add_argument(
        "--tv-loss-penalty",
        type=float,
        dest="tv_loss_penalty",
        default=1.0,
        help="Scalar multiplier for the total-variation smoothness loss (default: 1.0).",
    )
    parser.add_argument(
        "--elastic-loss-penalty",
        type=float,
        dest="elastic_loss_penalty",
        default=0.1,
        help="Scalar multiplier for the elastic-consistency loss (default: 0.1).",
    )
    parser.add_argument(
        "--seismic-loss-penalty",
        type=float,
        dest="seismic_loss_penalty",
        default=0.1,
        help="Scalar multiplier for the seismic physics loss (default: 0.1).",
    )
    parser.add_argument(
        "--dz-pixel",
        type=float,
        dest="dz_pixel",
        default=5.0,
        help="Vertical resolution in meters per pixel (default: 5.0).",
    )
    parser.add_argument(
        "--wavelet-f-peak",
        type=float,
        dest="wavelet_f_peak",
        default=PhysicsConfig.WAVELET_F_PEAK,
        help=f"Peak frequency for the default Ricker wavelet (default: {PhysicsConfig.WAVELET_F_PEAK}).",
    )
    parser.add_argument(
        "--wavelet-dt",
        type=float,
        dest="wavelet_dt",
        default=PhysicsConfig.WAVELET_DT,
        help=f"Sampling interval of the wavelet in seconds (default: {PhysicsConfig.WAVELET_DT}).",
    )
    parser.add_argument(
        "--normalization-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        dest="normalization_range",
        default=NORMALIZATION_RANGE,
        help="Normalization range [MIN MAX] used for padding midpoint defaults.",
    )

    # Latent space / Metrics options
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

    return parser


def build_training_args(
    args: Namespace, variant: VariantConfig, output: str, start_scale: int = 0
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
        "--seismic-stretch-percentile",
        str(args.seismic_stretch_percentile),
        "--rec-skip-per-scale",
        str(args.rec_skip_per_scale),
        "--normalization-range",
        str(args.normalization_range[0]),
        str(args.normalization_range[1]),
    ]

    if args.manual_seed is not None:
        cmd.extend(["--manual-seed", str(args.manual_seed)])
    if args.scale0_padding_size is not None:
        cmd.extend(["--scale0-padding-size", str(args.scale0_padding_size)])
    if args.no_shuffle:
        cmd.append("--no-shuffle")
    if args.no_tensorboard:
        cmd.append("--no-tensorboard")
    if args.no_plot_outputs:
        cmd.append("--no-plot-outputs")
    if args.no_compile:
        cmd.append("--no-compile")
    if args.compile_backend:
        cmd.append("--compile-backend")
    if args.triton_kernels:
        cmd.append("--triton-kernels")
    if args.no_triton_kernels:
        cmd.append("--no-triton-kernels")

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
        cmd.extend(["--rock-physics-loss-penalty", str(args.rock_physics_loss_penalty)])
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

    return cmd
