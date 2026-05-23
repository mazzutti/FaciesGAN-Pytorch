"""Typed option namespaces used by CLI entry points and scripts.

This module provides :class:`TrainingOptions` and :class:`ResumeOptions`,
lightweight dataclass-backed namespaces that mirror the command-line
arguments used throughout the project. They may be passed as the
``namespace`` argument to ``argparse.ArgumentParser.parse_args``.
"""

import argparse
import os
from typing import Any
from dataclasses import dataclass

from config import DirectoryConfig, DomainConfig
from enums import EmbeddingMethod, FeatureKey, SchedulerType, TimeUnit

NORMALIZATION_RANGE: tuple[float, float] = (-1.0, 1.0)


# noinspection PyMissingConstructor
@dataclass
class TrainingOptions(argparse.Namespace):
    """Namespace-like object holding training options with explicit defaults.

    This class mirrors the command-line arguments and provides defaults when
    instantiated without parameters. It is safe to pass an instance to
    `argparse.ArgumentParser.parse_args(namespace=...)`.
    """

    def __init__(
        self,
        rec_facies_loss_penalty: float = 10,
        batch_size: int = 1,
        beta1: float = 0.5,
        crop_size: int = 256,
        discriminator_steps: int = 3,
        scale0_disc_steps_multiplier: int = 1,
        scale0_loss_multiplier: float = 1.0,
        num_facies_channels: int = DomainConfig.NUM_FACIES_CHANNELS,
        lr_d_factor: float = 0.9,
        scheduler_g: str = SchedulerType.PLATEAU,
        scheduler_d: str = SchedulerType.STEP,
        generator_steps: int = 3,
        gpu_device: int = 0,
        input_path: str = DirectoryConfig.DATA,
        kernel_size: int = 3,
        gradient_loss_penalty: float = 10.0,
        lr_d: float = 5e-04,
        lr_decay: int = 1000,
        lr_g: float = 5e-04,
        manual_seed: int | None = None,
        max_size: int = 1024,
        min_num_feature: int = 32,
        min_size: int = 12,
        noise_amp: float = 0.1,
        min_noise_amp: float = 0.1,
        scale0_noise_amp: float = 1.0,
        well_loss_penalty: float = 10.0,
        grad_clip_norm: float = 1.0,
        diversity_loss_penalty: float = 1.0,
        adversarial_loss_penalty: float = 1.0,
        num_diversity_samples: int = 3,
        num_feature: int = 32,
        num_generated_per_real: int = 5,
        num_iter: int = 2000,
        num_layer: int = 5,
        noise_channels: int = DomainConfig.NOISE_CHANNELS,
        num_real_facies: int = 5,
        num_train_pyramids: int = 200,
        num_parallel_scales: int = 2,
        num_workers: int = min(4, max(1, (os.cpu_count() or 1) // 2)),
        prefetch_factor: int = 4,
        output_path: str = DirectoryConfig.OUTPUTS,
        normalization_range: tuple[float, ...] = NORMALIZATION_RANGE,
        padding_size: int = 0,
        regen_npy_gz: bool = False,
        save_interval: int = 100,
        checkpoint_interval: int = 1,
        start_scale: int = 0,
        stride: int = 1,
        stop_scale: int = 6,
        use_cpu: bool = False,
        use_wells: bool = False,
        use_seismic: bool = False,
        use_rock_physics: bool = False,
        vp_vs_robust_range: bool = False,
        vp_vs_robust_percentiles: tuple[float, float] = (1.0, 99.0),
        wells_mask_columns: tuple[int, ...] = (),
        enable_tensorboard: bool = True,
        enable_plot_outputs: bool = True,
        shuffle: bool = True,
        gp_interval: int = 16,
        gradient_checkpointing: bool = False,
        amp_dtype: str = "bf16",
        rec_rock_physics_loss_penalty: float = 10.0,
        tv_loss_penalty: float = 1e-4,
        elastic_loss_penalty: float = 0.1,
        seismic_loss_penalty: float = 0.1,
        dz_pixel: float = 1.0,
        wavelet_f_peak: float = 26.0,
        wavelet_dt: float = 0.001,
        wavelet_length: float = 0.128,
        lr_patience: int = 400,
        lr_min: float = 1e-4,
        lr_smoothing_alpha: float = 0.95,
        lr_g_factor: float = 0.8,
        time_unit: str = TimeUnit.STEP,
        compile_backend: bool = True,
        seismic_stretch_percentile: int = 98,
        scale0_padding_size: int | None = None,
        scale0_r1_gamma: float = 0.0,
        scale0_disc_grad_clip: float = 0.0,
        scale0_gradient_loss_penalty: float = 0.0,
        scale0_disc_lr_factor: float = 1.0,
        use_gradnorm: bool = False,
        gradnorm_interval: int = 16,
        gradnorm_alpha: float = 0.15,
        gradnorm_lr: float = 0.0005,
    ) -> None:
        """Create a TrainingOptions namespace with defaults for training.

        Parameters
        ----------
        rec_facies_loss_penalty : float, optional
            Weight for facies reconstruction loss (Dice Loss) used by the model. Default
            is 10.
        batch_size : int, optional
            Number of samples per batch. Default is 1.
        beta1 : float, optional
            Beta1 parameter for the Adam optimizer. Default is 0.5.
        crop_size : int, optional
            Size to crop input facies for training. Default is 256.
        discriminator_steps : int, optional
            Number of discriminator steps per training iteration. Default is 3.
        scale0_disc_steps_multiplier : int, optional
            Multiplier for extra discriminator steps at scale 0 only. E.g. 2
            runs twice as many D-steps at the coarsest scale. Default is 1.
        num_facies_channels : int, optional
            Number of facies output channels (e.g. 3 for RGB). Default is 3.
            The facies classes are typically:
            0: Floodplain, 1: Point bar, 2: Channel, 3: Boundary.
        lr_d_factor : float, optional
            Learning-rate scheduler decay factor (multiplier) for discriminator. Default is 0.9.
        scheduler_g : str, optional
            Learning-rate scheduler type for generator ('plateau', 'step', 'none'). Default is 'plateau'.
        scheduler_d : str, optional
            Learning-rate scheduler type for discriminator ('plateau', 'step', 'none'). Default is 'step'.
        generator_steps : int, optional
            Number of generator steps per training iteration. Default is 3.
        gpu_device : int, optional
            GPU device id to use when CUDA is available. Default is 0.
        input_path : str, optional
            Path to the dataset root directory. Default is "data/."
        kernel_size : int, optional
            Convolution kernel size used across the networks. Default is 3.
        gradient_loss_penalty : float, optional
            Gradient penalty weight for discriminator regularization. Default
            is 10.0.
        lr_d : float, optional
            Learning rate for the discriminator optimizer. Default is 5e-05.
        lr_decay : int, optional
            Number of epochs before the learning rate scheduler decays. Default
            is 1000.
        lr_g : float, optional
            Learning rate for the generator optimizer. Default is 5e-05.
        manual_seed : int or None, optional
            Optional random seed for reproducibility. Default is None.
        max_size : int, optional
            Maximum image size used in scale generation. Default is 1024.
        min_num_feature : int, optional
            Minimum number of features in network layers. Default is 32.
        min_size : int, optional
            Minimum size at the coarsest pyramid scale. Default is 12.
        noise_amp : float, optional
            Base amplitude used to scale adaptive noise. Default is 0.1.
        min_noise_amp : float, optional
            Minimum noise amplitude floor for diversity. Default is 0.1.
        scale0_noise_amp : float, optional
            Noise amplitude at scale 0 (controls structural diversity). Default is 1.0.
        diversity_loss_penalty : float, optional
            Scalar multiplier for the generator diversity loss. Default is 1.0.
        adversarial_loss_penalty : float, optional
            Scalar multiplier applied to the generator adversarial loss
            (``-E[D(fake)]``). Default is 1.0.
        num_diversity_samples : int, optional
            Number of noise samples to generate per real example when computing
            the diversity loss. Default is 3.
        num_feature : int, optional
            Base number of features in the first network layer. Default is 32.
        num_generated_per_real : int, optional
            How many generated facies to produce per real example. Default is 5.
        num_iter : int, optional
            Number of full passes through the training dataset. Each pass
            shuffles the dataset independently. Default is 2000.
        num_layer : int, optional
            Number of layers per block/scale. Default is 5.
        num_real_facies : int, optional
            Number of real facies used when composing result grids. Default is 5.
        output_path : str, optional
            Output directory for checkpoints and outputs. Default is "outputs/."
        padding_size : int, optional
            Padding size applied in network layers. Default is 0.
        regen_npy_gz : bool, optional
            If True, regenerate the npy.gz files from the input data. Default
            is False.
        save_interval : int, optional
            Interval (in epochs) between saving generated outputs. Default is
            100.
        start_scale : int, optional
            Starting scale index for training. Default is 0.
        stride : int, optional
            Convolution stride used across the networks. Default is 1.
        stop_scale : int, optional
            Final scale index (number of pyramid levels - 1). Default is 6.
        use_cpu : bool, optional
            Force CPU even if CUDA is available. Default is False.
        use_wells : bool, optional
            If True, enable loading/using well data (filter dataset by `wells`). Default is False.
        use_seismic : bool, optional
            If True, enable loading/using seismic data during training. Default is False.
        use_rock_physics : bool, optional
            If True, use Ip, Is, and Vp/Vs data as continuous outputs. Default is False.
        enable_tensorboard : bool, optional
            Enable TensorBoard logging during training. Default is True.
        enable_plot_outputs : bool, optional
            Enable saving generated output visualizations (facies and rock physics) during training. Default is True.
        tv_loss_penalty : float, optional
            Scalar multiplier for the total variation loss (smoothness) applied to rock physics. Default is 1e-4.
        elastic_loss_penalty : float, optional
            Scalar multiplier for the elastic consistency loss (MSE between Ip/Is and VpVs). Default is 1.0.
        seismic_loss_penalty : float, optional
            Scalar multiplier for the geophysical seismic loss (MSE between synthetic and real seismic). Default is 1e-4.
        dz_pixel : float, optional
            Vertical resolution of the target scale in meters per pixel. Default is 1.0.
            Note: The real seismic dataset is convolved sample-by-sample (effectively
            dt = 1.0 ms). To match its wavelet width and avoid checkerboard artifacts at
            Scale 6 (256x256), dz_pixel should be set to ~0.75 m (equivalent to 1.595 m
            for the original 120-pixel grid). Setting this too high (e.g. 5.0) causes
            the wavelet to become 3.13x too thin in pixel space, triggering spatial aliasing.
        wavelet_f_peak : float, optional
            Peak frequency of the Ricker wavelet in Hz. Default is 8.0.
        wavelet_dt : float, optional
            Sampling interval of the wavelet in seconds. Default is 0.001.
        wavelet_length : float, optional
            Total length of the wavelet in seconds. Default is 0.128.
        seismic_stretch_percentile : int, optional
            Percentile for TensorBoard seismic contrast stretch (display-only).
            Allowed: 95, 98, 99 (default: 98).

        Notes
        -----
        All parameters set here are attached as attributes on the resulting
        `TrainingOptions` instance so `argparse` can populate them when used
        as the `namespace=` for `ArgumentParser.parse_args`.
        """
        # Assign attributes (alphabetical by attribute name)
        self.rec_facies_loss_penalty = rec_facies_loss_penalty
        self.batch_size = batch_size
        self.beta1 = beta1
        self.crop_size = crop_size
        self.discriminator_steps = discriminator_steps
        self.scale0_disc_steps_multiplier = scale0_disc_steps_multiplier
        self.scale0_loss_multiplier = scale0_loss_multiplier
        self.num_facies_channels = num_facies_channels
        self.lr_d_factor = lr_d_factor
        self.scheduler_g = scheduler_g
        self.scheduler_d = scheduler_d
        self.generator_steps = generator_steps
        self.gpu_device = gpu_device
        self.input_path = input_path
        self.kernel_size = kernel_size
        self.gradient_loss_penalty = gradient_loss_penalty
        self.lr_d = lr_d
        self.lr_decay = lr_decay
        self.lr_g = lr_g
        self.manual_seed = manual_seed
        self.max_size = max_size
        self.min_num_feature = min_num_feature
        self.min_size = min_size
        self.noise_amp = noise_amp
        self.min_noise_amp = min_noise_amp
        self.scale0_noise_amp = scale0_noise_amp
        self.well_loss_penalty = well_loss_penalty
        self.grad_clip_norm = grad_clip_norm
        self.diversity_loss_penalty = diversity_loss_penalty
        self.adversarial_loss_penalty = adversarial_loss_penalty
        self.num_diversity_samples = num_diversity_samples
        self.num_feature = num_feature
        self.num_generated_per_real = num_generated_per_real
        self.num_iter = num_iter
        self.num_layer = num_layer
        self.num_real_facies = num_real_facies
        self.num_train_pyramids = num_train_pyramids
        self.num_parallel_scales = num_parallel_scales
        self.noise_channels = noise_channels
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.output_path = output_path
        self.normalization_range = tuple(normalization_range)
        self.padding_size = padding_size
        self.regen_npy_gz = regen_npy_gz
        self.save_interval = save_interval
        self.checkpoint_interval = checkpoint_interval
        self.start_scale = start_scale
        self.stride = stride
        self.stop_scale = stop_scale
        self.use_cpu = use_cpu
        self.use_wells = use_wells
        self.use_seismic = use_seismic
        self.use_rock_physics = use_rock_physics
        self.vp_vs_robust_range = vp_vs_robust_range
        self.vp_vs_robust_percentiles = tuple(vp_vs_robust_percentiles)
        self.wells_mask_columns = wells_mask_columns
        self.enable_tensorboard = enable_tensorboard
        self.enable_plot_outputs = enable_plot_outputs
        self.shuffle = shuffle
        self.gp_interval = gp_interval
        self.gradient_checkpointing = gradient_checkpointing
        self.amp_dtype = amp_dtype
        self.rec_rock_physics_loss_penalty = rec_rock_physics_loss_penalty
        self.tv_loss_penalty = tv_loss_penalty
        self.elastic_loss_penalty = elastic_loss_penalty
        self.seismic_loss_penalty = seismic_loss_penalty
        self.dz_pixel = dz_pixel
        self.wavelet_f_peak = wavelet_f_peak
        self.wavelet_dt = wavelet_dt
        self.wavelet_length = wavelet_length
        self.layers_per_scale = num_layer
        self.lr_patience = lr_patience
        self.lr_min = lr_min
        self.lr_smoothing_alpha = lr_smoothing_alpha
        self.lr_g_factor = lr_g_factor
        self.time_unit = time_unit
        self.compile_backend = compile_backend
        self.seismic_stretch_percentile = seismic_stretch_percentile
        # Discriminator padding override for scale 0 only (None = use global padding_size).
        self.scale0_padding_size = scale0_padding_size
        # R1 gradient penalty weight for scale 0 discriminator (0 = disabled).
        self.scale0_r1_gamma = scale0_r1_gamma
        # Gradient clip norm for scale 0 discriminator parameters (0.0 = disabled).
        self.scale0_disc_grad_clip = scale0_disc_grad_clip
        # GP override for scale 0 discriminator (0.0 = use global gradient_loss_penalty).
        self.scale0_gradient_loss_penalty = scale0_gradient_loss_penalty
        # Learning-rate multiplier for the scale 0 discriminator (1.0 = same as global lr_d).
        # Values < 1 (e.g. 0.2) slow down D at s0 so G can keep up.
        self.scale0_disc_lr_factor = scale0_disc_lr_factor
        self.use_gradnorm = use_gradnorm
        self.gradnorm_interval = gradnorm_interval
        self.gradnorm_alpha = gradnorm_alpha
        self.gradnorm_lr = gradnorm_lr


@dataclass
class ExperimentOptions(TrainingOptions):
    """Namespace for ablation experiment runner arguments.

    This class extends :class:`TrainingOptions` with experiment-specific
    parameters such as sample counts, embedding methods, and evaluation
    flags. It provides a typed interface for orchestrating multi-variant
    training and generation studies.
    """

    def __init__(
        self,
        how_many: int = 2000,
        skip_training: bool = False,
        model_paths: list[str] | None = None,
        nproc_per_node: int = 2,
        embedding_methods: list[str] = [
            EmbeddingMethod.ISOMAP,
            EmbeddingMethod.MDS,
            EmbeddingMethod.TSNE,
            EmbeddingMethod.UMAP,
        ],
        embedding_data: list[str] = [
            FeatureKey.FACIES,
            FeatureKey.ROCK_PHYSICS,
            FeatureKey.SEISMIC,
        ],
        embedding_per_facies: bool = False,
        no_embeddings: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize ExperimentOptions, forwarding training args to parent.

        Parameters
        ----------
        how_many : int, optional
            Number of facies to generate per variant during evaluation.
            Default is 2000.
        skip_training : bool, optional
            If True, skip the training phase and only run generation using
            existing model paths. Default is False.
        model_paths : list of str, optional
            Explicit model paths (one per variant) to use when `skip_training`
            is True. Default is None.
        nproc_per_node : int, optional
            Number of processes (GPUs) per node to use for DDP training.
            Default is 2.
        embedding_methods : list of str, optional
            Dimensionality reduction methods (e.g., 'isomap', 'tsne') for latent
            space visualization. Default is ["isomap", "mds", "tsne", "umap"].
        embedding_data : list of str, optional
            Data types (e.g., 'facies', 'rock_physics', 'seismic') to include in the
            embedding analysis. Default is ["facies", "rock_physics", "seismic"].
        embedding_per_facies : bool, optional
            If True, generate separate embedding plots for each unique
            conditioning crossline. Default is False.
        no_embeddings : bool, optional
            If True, disable all manifold learning and latent space
            visualizations. Default is False.
        **kwargs : Any
            Additional training arguments passed to the :class:`TrainingOptions`
            constructor.
        """
        super().__init__(**kwargs)
        self.how_many = how_many
        self.skip_training = skip_training
        self.model_paths = model_paths
        self.nproc_per_node = nproc_per_node
        self.embedding_methods = embedding_methods
        self.embedding_data = embedding_data
        self.embedding_per_facies = embedding_per_facies
        self.no_embeddings = no_embeddings


# noinspection PyMissingConstructor
class ResumeOptions(argparse.Namespace):
    """Namespace-like object holding resume script options.

    This mirrors the command-line arguments used by `resume.py` and provides
    explicit defaults. An instance is safe to pass to
    `argparse.ArgumentParser.parse_args(namespace=...)`.
    """

    def __init__(
        self,
        fine_tuning: bool = False,
        checkpoint_path: str = "",
        num_iter: int | None = None,
        start_scale: int = 0,
    ) -> None:
        """Initialize ResumeOptions.

        Parameters
        ----------
        fine_tuning : bool
            If True, resume script will perform fine-tuning of the models.
        checkpoint_path : str
            Path to the checkpoint directory to resume training from.
            This is typically required when resuming.
        num_iter : int or None
            Number of iterations (epochs) to run when fine-tuning. If
            `fine_tuning` is True, this should be provided by the caller.
        start_scale : int
            Starting scale index for resuming/fine-tuning (default 0).
        """
        self.fine_tuning = fine_tuning
        self.checkpoint_path = checkpoint_path
        self.num_iter = num_iter
        self.start_scale = start_scale
