"""Shared CLI argument parsing helpers for FaciesGAN.

This module provides reusable functions to add standard argument groups (I/O,
network, optimization, pyramid, physics) to an argparse.ArgumentParser.
Using these helpers ensures consistency across different entry points (training,
experiments, generation) and reduces code duplication.
"""

import os
from argparse import ArgumentParser
from typing import Any, Optional, cast

from config import DirectoryConfig, PhysicsConfig
from enums import AmpDtype, SchedulerType, TimeUnit
from options import NORMALIZATION_RANGE


def add_device_args(parser: ArgumentParser) -> None:
    """Add hardware and reproducibility arguments."""
    parser.add_argument("--use-cpu", action="store_true", help="use cpu")
    parser.add_argument("--gpu-device", type=int, help="which GPU to use", default=0)
    parser.add_argument("--manual-seed", type=int, help="manual seed", default=None)


def add_io_args(
    parser: ArgumentParser, defaults: Optional[dict[str, Any]] = None
) -> None:
    """Add input/output and dataset configuration arguments."""
    d = defaults or {}
    parser.add_argument("--input-path", help="input facie path", required=True)
    parser.add_argument(
        "--output-path",
        help="output folder path",
        default=d.get("output_path", DirectoryConfig.OUTPUTS),
    )
    parser.add_argument(
        "--output-fullpath",
        help="Set exact output path (overrides automatic timestamp prefix).",
        default=None,
    )
    parser.add_argument(
        "--num-facies",
        type=int,
        dest="num_facies",
        help="number of one-hot encoded facies classes (channels)",
        default=d.get("num_facies", 3),
    )
    parser.add_argument(
        "--img-color-range",
        type=int,
        nargs=2,
        help="range of values in the input facie",
        default=d.get("img_color_range", [0, 255]),
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        help="crop size to train the facie",
        default=d.get("crop_size", 256),
    )
    parser.add_argument(
        "--batch-size",
        default=d.get("batch_size", 1),
        type=int,
        help="Total batch size - e.g: num_gpus = 2, batch_size = 128 then, effectively, 64",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        help="Interval (in units specified by --time-unit) between saving generated outputs and plots (default: 100).",
        default=d.get("save_interval", 100),
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        help="Interval (in units specified by --time-unit) between saving training state checkpoints for resume (default: 1).",
        default=d.get("checkpoint_interval", 1),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number of workers for data loading (0 = main process only). "
        "Defaults to min(4, cpu_count//2) for parallel data prep.",
        default=cast(
            int, d.get("num_workers", min(4, max(1, (os.cpu_count() or 1) // 2)))
        ),
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        help="Number of batches loaded in advance by each worker (default: 4).",
        default=d.get("prefetch_factor", 4),
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_false",
        dest="shuffle",
        default=d.get("shuffle", True),
        help="Disable dataset shuffling (useful for reproducible debugging).",
    )


def add_network_args(parser: ArgumentParser) -> None:
    """Add network architecture and feature configuration arguments."""
    parser.add_argument(
        "--num-features",
        dest="num_feature",
        type=int,
        help="initial number of features in each layer",
        default=32,
    )
    parser.add_argument(
        "--min-num-features",
        dest="min_num_feature",
        type=int,
        help="minimal number of features in each layer",
        default=32,
    )
    parser.add_argument("--kernel-size", type=int, help="kernel size", default=3)
    parser.add_argument(
        "--num-layers",
        dest="num_layer",
        type=int,
        help="number of layers in each scale",
        default=5,
    )
    parser.add_argument("--stride", type=int, help="stride", default=1)
    parser.add_argument(
        "--normalization-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        dest="normalization_range",
        help="Normalization range [MIN MAX] used to derive default padding midpoint.",
        default=NORMALIZATION_RANGE,
    )
    parser.add_argument("--padding-size", type=int, help="net pad size", default=0)


def add_optimization_args(
    parser: ArgumentParser, defaults: Optional[dict[str, Any]] = None
) -> None:
    """Add optimizer and training dynamic arguments."""
    d = defaults or {}
    parser.add_argument(
        "--num-iter",
        type=int,
        default=d.get("num_iter", 2000),
        help="number of full dataset passes (each pass shuffles the dataset independently)",
    )
    parser.add_argument(
        "--lr-d-factor",
        type=float,
        dest="lr_d_factor",
        help="scheduler decay factor (multiplier) for discriminator StepLR",
        default=d.get("lr_d_factor", 0.9),
    )
    parser.add_argument(
        "--lr-g",
        type=float,
        default=d.get("lr_g", 5e-4),
        help="learning rate, default=5e-4",
    )
    parser.add_argument(
        "--lr-d",
        type=float,
        default=d.get("lr_d", 5e-4),
        help="learning rate, default=5e-4",
    )
    parser.add_argument(
        "--lr-decay",
        type=int,
        default=d.get("lr_decay", 100),
        help="learning rate decay interval (in units specified by --time-unit, used by discriminator StepLR)",
    )
    parser.add_argument(
        "--time-unit",
        type=str,
        choices=[u.value for u in TimeUnit],
        default=d.get("time_unit", TimeUnit.STEP),
        help="unit for all training intervals (lr-decay, lr-patience, save-interval, checkpoint-interval): "
        "'epoch' measures in epochs, 'step'/'batch' measures in optimization steps/iterations (default: step)",
    )
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=d.get("lr_patience", 400),
        help="ReduceLROnPlateau patience: training units (specified by --time-unit) with no improvement before reducing generator LR (default: 400)",
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=d.get("lr_min", 1e-4),
        help="minimum learning rate for generator ReduceLROnPlateau (default: 1e-4)",
    )
    parser.add_argument(
        "--lr-smoothing-alpha",
        type=float,
        default=d.get("lr_smoothing_alpha", 0.95),
        help="EMA smoothing factor for generator loss fed to ReduceLROnPlateau (0=no smoothing, 0.99=very smooth, default: 0.95)",
    )
    parser.add_argument(
        "--lr-g-factor",
        type=float,
        default=d.get("lr_g_factor", 0.8),
        help="Factor by which the generator LR is reduced on plateau (default: 0.8)",
    )
    parser.add_argument(
        "--scheduler-g",
        type=str,
        choices=[s.value for s in SchedulerType],
        default=d.get("scheduler_g", SchedulerType.PLATEAU),
        help="learning rate scheduler type for generator (default: plateau)",
    )
    parser.add_argument(
        "--scheduler-d",
        type=str,
        choices=[s.value for s in SchedulerType],
        default=d.get("scheduler_d", SchedulerType.STEP),
        help="learning rate scheduler type for discriminator (default: step)",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=d.get("beta1", 0.5),
        help="beta1 for adam. default=0.5",
    )
    parser.add_argument(
        "--generator-steps",
        type=int,
        help="Generator inner steps",
        default=d.get("generator_steps", 3),
    )
    parser.add_argument(
        "--use-gradnorm",
        action="store_true",
        default=d.get("use_gradnorm", False),
        help="Use GradNorm to dynamically balance multi-task generator losses.",
    )
    parser.add_argument(
        "--gradnorm-interval",
        type=int,
        default=d.get("gradnorm_interval", 16),
        help="Update loss weights via GradNorm every N generator steps.",
    )
    parser.add_argument(
        "--gradnorm-alpha",
        type=float,
        default=d.get("gradnorm_alpha", 0.15),
        help="GradNorm asymmetry parameter (restoring force strength).",
    )
    parser.add_argument(
        "--gradnorm-lr",
        type=float,
        default=d.get("gradnorm_lr", 0.0005),
        help="GradNorm optimizer learning rate.",
    )
    parser.add_argument(
        "--discriminator-steps",
        type=int,
        help="Discriminator inner steps",
        default=d.get("discriminator_steps", 3),
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=d.get("grad_clip_norm", 1.0),
        help="max gradient norm for generator clipping (default: 1.0; set to 0 to disable)",
    )
    parser.add_argument(
        "--amp-dtype",
        type=str,
        choices=[u.value for u in AmpDtype],
        default=d.get("amp_dtype", AmpDtype.BF16),
        help=(
            "AMP compute dtype for CUDA autocast. "
            "Use bf16 on Ampere+ for improved stability/perf tradeoff."
        ),
    )
    parser.add_argument(
        "--rec",
        action="store_true",
        help="Enable rock physics conditioning ablation study mode.",
    )


def add_pyramid_args(
    parser: ArgumentParser, defaults: Optional[dict[str, Any]] = None
) -> None:
    """Add multi-scale pyramid and parallel training arguments."""
    d = defaults or {}
    parser.add_argument(
        "--stop-scale", type=int, help="stop scale", default=d.get("stop_scale", 6)
    )
    parser.add_argument(
        "--min-size",
        type=int,
        dest="min_size",
        help="minimum image size at the coarsest pyramid scale (default: 12)",
        default=d.get("min_size", 12),
    )
    parser.add_argument(
        "--start-scale",
        type=int,
        help="scale to start/resume training from (default: 0)",
        default=d.get("start_scale", 0),
    )
    parser.add_argument(
        "--num-parallel-scales",
        type=int,
        help="Number of scales to train in parallel (default: 2)",
        default=d.get("num_parallel_scales", 2),
    )
    parser.add_argument(
        "--noise-channels",
        type=int,
        help="number of noise channels to generate per scale",
        default=d.get("noise_channels", 3),
    )
    parser.add_argument(
        "--noise-amp",
        type=float,
        help="adaptive noise cont weight",
        default=d.get("noise_amp", 0.1),
    )
    parser.add_argument(
        "--min-noise-amp",
        type=float,
        help="minimum noise amplitude floor for diversity",
        default=d.get("min_noise_amp", 0.1),
    )
    parser.add_argument(
        "--scale0-noise-amp",
        type=float,
        help="noise amplitude at scale 0 (controls structural diversity)",
        default=d.get("scale0_noise_amp", 1.0),
    )
    parser.add_argument(
        "--num-train-pyramids",
        type=int,
        help="Number of train pyramids to use in the FaciesGAN training",
        default=d.get("num_train_pyramids", 200),
    )


def add_physics_args(parser: ArgumentParser) -> None:
    """Add well, seismic, and rock physics consistency arguments."""
    parser.add_argument(
        "--use-wells",
        action="store_true",
        help="enable using wells during data loading (filter by --wells-mask-columns if set)",
    )
    parser.add_argument(
        "--wells-mask-columns",
        type=int,
        help="list of well indices to train the model from",
        nargs="+",
        default=tuple(),
    )
    parser.add_argument(
        "--well-loss-penalty",
        type=float,
        help="weight multiplier for well/mask reconstruction loss",
        default=10.0,
    )
    parser.add_argument(
        "--use-seismic",
        action="store_true",
        help="enable using seismic data during data loading",
    )
    parser.add_argument(
        "--use-rock-physics",
        action="store_true",
        dest="use_rock_physics",
        help="Train with rock physics volumes (Ip, Is, Vp/Vs) as additional output channels.",
    )
    parser.add_argument(
        "--no-ip",
        action="store_false",
        dest="use_ip",
        default=True,
        help="Disable Acoustic Impedance (Ip) output channel.",
    )
    parser.add_argument(
        "--no-is",
        action="store_false",
        dest="use_is",
        default=True,
        help="Disable Shear Impedance (Is) output channel.",
    )
    parser.add_argument(
        "--no-vpvs",
        action="store_false",
        dest="use_vpvs",
        default=True,
        help="Disable Vp/Vs output channel.",
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
        default=1.0,
        help="Vertical resolution in meters per pixel (default: 1.0). For Scale 6 (256x256) matching the real seismic, set to ~0.75 to prevent wavelet thinning and aliasing.",
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
        "--wavelet-length",
        type=float,
        dest="wavelet_length",
        default=0.128,
        help="Total length of the wavelet in seconds (default: 0.128).",
    )
    parser.add_argument(
        "--vp-vs-robust-range",
        action="store_true",
        help="Use robust min/max (percentiles) for Vp/Vs normalization.",
    )
    parser.add_argument(
        "--vp-vs-robust-percentiles",
        type=float,
        nargs=2,
        default=(1.0, 99.0),
        help="Percentiles for robust Vp/Vs range (default: 1.0 99.0).",
    )


def add_scale0_args(
    parser: ArgumentParser, defaults: Optional[dict[str, Any]] = None
) -> None:
    """Add discriminator and loss overrides for the first pyramid scale (s0)."""
    d = defaults or {}
    parser.add_argument(
        "--scale0-disc-steps-multiplier",
        type=int,
        default=d.get("scale0_disc_steps_multiplier", 1),
        help="Extra D-step multiplier for scale 0 only (default: 1).",
    )
    parser.add_argument(
        "--scale0-loss-multiplier",
        type=float,
        default=d.get("scale0_loss_multiplier", 1.0),
        help="Extra loss multiplier for rec and rock_physics at scale 0 (default: 1.0).",
    )
    parser.add_argument(
        "--scale0-padding-size",
        type=int,
        default=d.get("scale0_padding_size", None),
        dest="scale0_padding_size",
        help="Discriminator padding size override for scale 0 only.",
    )
    parser.add_argument(
        "--scale0-r1-gamma",
        type=float,
        default=d.get("scale0_r1_gamma", 0.0),
        dest="scale0_r1_gamma",
        help="R1 gradient penalty weight for scale 0 discriminator (default: 0.0).",
    )
    parser.add_argument(
        "--scale0-disc-grad-clip",
        type=float,
        default=d.get("scale0_disc_grad_clip", 0.0),
        dest="scale0_disc_grad_clip",
        help="Gradient clip norm for scale 0 discriminator parameters (default: 0.0).",
    )
    parser.add_argument(
        "--scale0-gradient-loss-penalty",
        type=float,
        default=d.get("scale0_gradient_loss_penalty", 0.0),
        dest="scale0_gradient_loss_penalty",
        help="Gradient penalty override for scale 0 discriminator (default: 0.0).",
    )
    parser.add_argument(
        "--scale0-disc-lr-factor",
        "--scale0_disc_lr_factor",
        type=float,
        default=d.get("scale0_disc_lr_factor", 1.0),
        dest="scale0_disc_lr_factor",
        help="Learning-rate multiplier for the scale 0 discriminator (default: 1.0).",
    )


def add_gan_loss_args(
    parser: ArgumentParser, defaults: Optional[dict[str, Any]] = None
) -> None:
    """Add GAN loss and regularization arguments."""
    d = defaults or {}
    parser.add_argument(
        "--gradient-loss-penalty",
        type=float,
        help="gradient penalty weight",
        default=d.get("gradient_loss_penalty", 10.0),
    )
    parser.add_argument(
        "--rec-facies-loss-penalty",
        type=float,
        dest="rec_facies_loss_penalty",
        help="reconstruction loss weight",
        default=d.get("rec_facies_loss_penalty", 10.0),
    )
    parser.add_argument(
        "--gp-interval",
        type=int,
        help="Gradient penalty lazy regularization interval (default 16).",
        default=d.get("gp_interval", 16),
    )
    parser.add_argument(
        "--num-diversity-samples",
        type=int,
        help="number of diverse samples per G-step (default: 3)",
        default=d.get("num_diversity_samples", 3),
    )
    parser.add_argument(
        "--diversity-loss-penalty",
        type=float,
        dest="diversity_loss_penalty",
        help="multiplier for generator diversity loss (default: 1.0)",
        default=d.get("diversity_loss_penalty", 1.0),
    )
    parser.add_argument(
        "--adversarial-loss-penalty",
        type=float,
        help="multiplier for generator adversarial loss (default: 1.0)",
        default=d.get("adversarial_loss_penalty", 1.0),
    )


def add_runtime_args(parser: ArgumentParser) -> None:
    """Add runtime control arguments (logging, visualization, compilation)."""
    parser.add_argument(
        "--enable-logging",
        action="store_true",
        dest="enable_logging",
        default=False,
        help="Enable Python logging and stdout/stderr teeing (default: disabled).",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_false",
        dest="enable_tensorboard",
        default=True,
        help="Disable TensorBoard logging during training (default: enabled).",
    )
    parser.add_argument(
        "--no-plot-outputs",
        action="store_false",
        dest="enable_plot_outputs",
        default=True,
        help="Disable generated output visualizations during training (default: enabled).",
    )
    parser.add_argument(
        "--no-compile",
        action="store_false",
        dest="compile_backend",
        default=True,
        help="Disable torch.compile (default: enabled).",
    )
    parser.add_argument(
        "--gradient-checkpoint",
        action="store_true",
        dest="gradient_checkpointing",
        help="Enable activation checkpointing on generator blocks to save memory.",
    )
    parser.add_argument(
        "--num-real-facies",
        type=int,
        help="Number of real facies in grid plot",
        default=5,
    )
    parser.add_argument(
        "--num-generated-per-real",
        type=int,
        help="Number of generated facies per real in grid plot",
        default=5,
    )
    parser.add_argument(
        "--seismic-stretch-percentile",
        type=int,
        choices=[95, 98, 99],
        default=98,
        help="Percentile for TensorBoard seismic contrast stretch (default: 98).",
    )
