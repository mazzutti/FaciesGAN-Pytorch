"""Base trainer abstraction for different training backends.

This module provides an abstract :class:`Trainer` base that defines the
minimal interface and a couple of shared utilities used by concrete
trainers such as :class:`training.trainer.TorchTrainer`.

Keep this class lightweight: it only initialises common configuration
fields and exposes abstract methods concrete trainers must implement.
"""

from __future__ import annotations

import math
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Iterator, cast

import torch
from tensorboardX import SummaryWriter  # type: ignore
from tqdm import tqdm  # type: ignore[import]

import utils


class DummyProgress:
    def update(self, n: int = 1) -> None:
        pass

    def set_description(self, desc: str) -> None:
        pass

    def write(self, s: str) -> None:
        print(s)

    def close(self) -> None:
        pass


import log
from config import OUTPUT_FACIES_PATH
from datasets import IDataLoader, PyramidsBatch, TorchPyramidsDataset
from metrics import DiscriminatorMetrics, GeneratorMetrics, ScaleMetrics
from models.facies_gan import TorchFaciesGAN
from models.utils import calculate_channels
from options import TrainingOptions
from tensorboard_visualizer import SeismicPhysicsInfo, TensorBoardVisualizer


class Trainer(ABC):
    """Abstract base class for training runners.

    Subclasses must implement :meth:`train` and :meth:`train_scales`.
    The constructor initialises a small set of commonly-used attributes
    from the provided :class:`TrainingOptions` instance.
    """

    model: TorchFaciesGAN

    def __init__(
        self,
        options: TrainingOptions,
        fine_tuning: bool = False,
        checkpoint_path: str = ".checkpoints",
    ) -> None:
        self.options: TrainingOptions = options
        self.fine_tuning: bool = fine_tuning
        self.checkpoint_path: str = checkpoint_path

        # Distributed training flag — only rank-0 should perform I/O.
        # Subclasses (e.g. TorchTrainer) may set this before calling super().
        if not hasattr(self, "_is_main_process"):
            self._is_main_process: bool = True

        # Common training parameters (conservative subset)
        self.start_scale: int = options.start_scale
        self.start_epoch: int = getattr(options, "start_epoch", 0)
        self.stop_scale: int = options.stop_scale
        self.output_path: str = options.output_path
        self.num_iter: int = 1  # inner per-batch epoch loop always runs once
        self.save_interval: int = options.save_interval
        self.num_parallel_scales: int = options.num_parallel_scales
        # Total dataset passes == the user's --num-iter value.
        self._num_passes: int = options.num_iter
        self._current_pass_idx: int = 0

        # How often to flush TensorBoard scalars (epochs).  Writing every
        # epoch triggers a GPU→CPU sync per scale; batching to every N epochs
        # reduces that overhead by ~N× with minimal loss of resolution.
        self._tb_log_interval: int = 10

        self.batch_size: int = (
            options.batch_size
            if (options.batch_size < options.num_train_pyramids)
            else options.num_train_pyramids
        )

        # Feature flags
        self.enable_tensorboard: bool = options.enable_tensorboard
        self.enable_plot_outputs: bool = options.enable_plot_outputs

        # Placeholder containers commonly used by concrete trainers
        self.visualizer: TensorBoardVisualizer | None = None

        # Determine channel counts using centralized utility
        channel_info = calculate_channels(options)
        self.total_output_channels = channel_info["generator_out"]
        from models.utils import calculate_noise_channels

        self.noise_channels = calculate_noise_channels(options)

        self.num_real_facies: int = options.num_real_facies
        self.num_generated_per_real: int = options.num_generated_per_real
        self.wells_mask_columns: tuple[int, ...] = options.wells_mask_columns

        # Optimizer configuration (default values from options)
        self.lr_g: float = options.lr_g
        self.lr_d: float = options.lr_d
        self.beta1: float = options.beta1
        self.lr_decay: int = options.lr_decay
        self.gamma: float = options.gamma

        # Model parameters
        self.zero_padding: int = options.num_layer * math.floor(options.kernel_size / 2)
        self.noise_amp: float = options.noise_amp
        self.min_noise_amp: float = options.min_noise_amp
        self.scale0_noise_amp: float = options.scale0_noise_amp

        # Initialize dataset and data loader
        dataset, scales = self.init_dataset()
        self.dataset: TorchPyramidsDataset = dataset
        self.num_of_batchs: int = len(self.dataset) // self.batch_size
        self.scales: tuple[tuple[int, ...], ...] = scales
        self.data_loader: IDataLoader = self.create_dataloader()

        if self._is_main_process:
            print(f"DataLoader num_workers: {self.data_loader.num_workers}")

        self.model: TorchFaciesGAN = self.create_model()
        self.model.shapes = list(self.scales)

        # learning rate decay unit: 'epoch' (per-batch, reset each batch)
        # or 'step' (global optimisation steps, not reset between batches)
        self.lr_decay_unit: str = getattr(options, "lr_decay_unit", "epoch")

        # learning rate gamma
        self.gamma = options.gamma

        # generator optimizers
        self.generator_optimizers: dict[int, Any] = {}

        # discriminator optimizers
        self.discriminator_optimizers: dict[int, Any] = {}

        # generator schedulers
        self.generator_schedulers: dict[int, Any] = {}

        # discriminator schedulers
        self.discriminator_schedulers: dict[int, Any] = {}

        # Cached indices gathered at the end of an epoch for visualization.
        self._seen_indices_for_save: list[int] = []

        if self._is_main_process:
            self._print_facie_shapes_table()

        # Initialize TensorBoard visualizer if enabled
        self.enable_tensorboard = options.enable_tensorboard
        self.enable_plot_outputs = options.enable_plot_outputs
        if self.enable_tensorboard and self._is_main_process:
            viz_path = os.path.join(self.output_path, "training_visualizations")
            log_dir = os.path.join(self.output_path, "tensorboard_logs")
            dataset_info = f"{len(self.dataset)} pyramids, {self.batch_size} batch size"
            if len(options.wells_mask_columns) > 0:
                dataset_info += f", wells: {options.wells_mask_columns}"

            _purge = self.start_epoch if self.start_epoch > 0 else None

            # Collect physics info for TensorBoard seismic modeling
            physics_info = None
            if options.use_rock_physics:
                physics_info = SeismicPhysicsInfo(
                    ip_min=cast(torch.Tensor, self.model.ip_min),
                    ip_max=cast(torch.Tensor, self.model.ip_max),
                    seis_min=cast(torch.Tensor, self.model.seis_min),
                    seis_max=cast(torch.Tensor, self.model.seis_max),
                    vp_min=cast(torch.Tensor, self.model.vp_min),
                    vp_max=cast(torch.Tensor, self.model.vp_max),
                    vp_ref=cast(torch.Tensor, self.model.vp_ref),
                    rho_mean=cast(torch.Tensor, self.model.rho_mean),
                    dz_pyramid=cast(torch.Tensor, self.model.dz_pyramid),
                    wavelet_t=self.model.wavelet_t,
                    dt_wavelet=options.wavelet_dt,
                )

            self.visualizer = TensorBoardVisualizer(
                num_scales=self.stop_scale - self.start_scale + 1,
                output_dir=viz_path,
                log_dir=log_dir,
                update_interval=1,
                image_log_interval=options.save_interval,
                dataset_info=dataset_info,
                purge_step=_purge,
                num_facies=options.num_facies_classes,
                has_rp=options.use_rock_physics,
                physics_info=physics_info,
            )
            print(f"📊 TensorBoard logging enabled")
            print(f"   logdir: {log_dir}")
            print(f"   URL: http://localhost:6006")
        else:
            self.visualizer = None  # type: ignore
            if self._is_main_process:
                print("📊 TensorBoard logging disabled")

    @abstractmethod
    def create_model(self) -> TorchFaciesGAN:
        """Create the model used by the trainer.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement create_model")

    @abstractmethod
    def create_dataloader(self) -> IDataLoader:
        """Create the data loader used by the trainer.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement create_dataloader")

    @abstractmethod
    def init_dataset(
        self,
    ) -> tuple[TorchPyramidsDataset, tuple[tuple[int, ...], ...]]:
        """Initialize the dataset used by the trainer.

        Returns
        -------
        tuple[TorchPyramidsDataset, tuple[tuple[int, ...], ...]]
            A tuple containing the dataset instance and the scales list used
            by the dataset.


        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement init_dataset")

    def _ddp_barrier(self) -> None:
        """Synchronize DDP ranks.  No-op for non-distributed training.

        Overridden by framework-specific trainers (e.g.
        :class:`TorchTrainer`) to call ``dist.barrier()``.
        """

    @abstractmethod
    def generate_visualization_samples(
        self,
        scales: tuple[int, ...],
        indexes: torch.Tensor,
        wells_pyramid: dict[int, torch.Tensor] = {},
        seismic_pyramid: dict[int, torch.Tensor] = {},
    ) -> tuple[torch.Tensor, ...]:
        """Generate fixed samples for visualization at specified scales.

        Parameters
        ----------
        scales : tuple[int, ...]
            Tuple of scale indices to generate samples for.
        indexes : list[int]
            List of batch sample indices.
        wells_pyramid : dict[int, torch.Tensor], optional
            Dictionary of well-conditioning tensors for all scales.
        seismic_pyramid : dict[int, torch.Tensor], optional
            Dictionary of seismic-conditioning tensors for all scales.

        Returns
        -------
        tuple[torch.Tensor, ...]
            A tuple mapping scale indices to generated facies tensors
            for visualization.
        """
        raise NotImplementedError(
            "Subclasses must implement generate_visualization_samples"
        )

    @abstractmethod
    def compute_rec_input(
        self,
        scale: int,
        indexes: torch.Tensor,
        facies_pyramid: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the reconstruction input tensor for a specific scale.

        This method upsamples the reconstruction from the previous scale
        to match the spatial dimensions of the current scale's real facies.

        Parameters
        ----------
        scale : int
            Current pyramid scale index.
        indexes : list[int]
            Batch sample indices.
        facies_pyramid : dict[int, torch.Tensor]
            Dictionary of real facies data for all scales.

        Returns
        -------
        torch.Tensor
            The upsampled reconstruction input tensor for the current scale.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement compute_rec_input")

    @abstractmethod
    def init_rec_noise_and_amp(
        self,
        scale: int,
        indexes: torch.Tensor,
        real: torch.Tensor,
        wells_pyramid: dict[int, torch.Tensor] = {},
        seismic_pyramid: dict[int, torch.Tensor] = {},
    ) -> None:
        """Initialize reconstruction noise and noise amplitude for a specific scale.

        Parameters
        ----------
        scale : int
            Current pyramid scale index.
        indexes : list[int]
            Batch sample indices.
        real : torch.Tensor
            Real facies tensor for the current scale.
        wells_pyramid : dict[int, torch.Tensor], optional
            Dictionary of well-conditioning tensors for all scales.
        seismic_pyramid : dict[int, torch.Tensor], optional
            Dictionary of seismic-conditioning tensors for all scales.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement init_rec_noise_and_amp")

    @abstractmethod
    def create_batch_iterator(
        self, loader: IDataLoader, scales: tuple[int, ...]
    ) -> Iterator[PyramidsBatch | None]:
        """Create an iterator that yields batches for training.

        Subclasses must implement this to define how data is fetched and
        prepared (e.g., using a prefetcher or standard iteration).

        Parameters
        ----------
        loader : IDataLoader
            The data loader to iterate over.
        scales : tuple[int, ...]
            Tuple of scale indices being trained.

        Yields
        ------
        DictBatch[torch.Tensor] | None
            The prepared batch for training, or None if no more batches are available.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement create_batch_iterator")

    @abstractmethod
    def setup_optimizers(self, scales: tuple[int, ...]) -> None:
        """Setup optimizers and schedulers for all scales.

        Parameters
        ----------
        scales : tuple[int, ...]
            Tuple of scale indices to setup optimizers for.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement setup_optimizers")

    @abstractmethod
    def reset_schedulers(self, scales: tuple[int, ...]) -> None:
        """Reset LR schedulers to their initial state for a new batch.

        Called at the start of each DataLoader batch so that every batch
        trains with the same LR schedule (e.g. decay at epoch 500, 1000).
        """
        raise NotImplementedError("Subclasses must implement reset_schedulers")

    def collect_seen_indices(self) -> list[int]:
        """Return the dataset indices seen by the current trainer iteration.

        Subclasses that use a prefetcher can override this to gather the
        indices across distributed ranks. The default implementation returns
        an empty list.
        """
        return []

    def _warmup_compile_traces(
        self,
        scales: tuple[int, ...],
        indexes: torch.Tensor,
        facies_pyramid: dict[int, torch.Tensor],
        rec_in_pyramid: dict[int, torch.Tensor],
        wells_pyramid: dict[int, torch.Tensor],
        seismic_pyramid: dict[int, torch.Tensor],
    ) -> None:
        """Hook for subclasses to warm up compiled code paths.

        The default implementation is a no-op.  ``TorchTrainer`` overrides
        this to run dummy forwards that force ``torch.compile`` to cache
        specializations for recovery-loss and diversity-N>1 branches
        before the training loop needs them.
        """

    def train_scales(
        self,
        scales: tuple[int, ...],
        writers: dict[int, SummaryWriter],
        scale_paths: dict[int, str],
        outputs_paths: dict[int, str],
        batch_id: int,
        progress: tqdm[Any],
        facies_pyramid: dict[int, torch.Tensor] | None = None,
        wells_pyramid: dict[int, torch.Tensor] | None = None,
        masks_pyramid: dict[int, torch.Tensor] | None = None,
        seismic_pyramid: dict[int, torch.Tensor] | None = None,
        start_epoch: int = 0,
        batch: PyramidsBatch | None = None,
    ) -> ScaleMetrics:
        """Process a single optimization step for the provided scales."""
        if batch is not None:
            indexes, facies_pyramid, wells_pyramid, masks_pyramid, seismic_pyramid = (
                batch
            )
        else:
            # Fallback if batch is not provided (should not happen with prefetcher)
            first_scale = min(facies_pyramid) if facies_pyramid else 0
            actual_batch = facies_pyramid[first_scale].shape[0] if facies_pyramid else 0
            indexes = torch.arange(
                actual_batch,
                device=facies_pyramid[first_scale].device if facies_pyramid else "cpu",
            )

        if facies_pyramid is None:
            raise ValueError("facies_pyramid must be provided or part of batch")
        if wells_pyramid is None:
            wells_pyramid = {}
        if masks_pyramid is None:
            masks_pyramid = {}
        if seismic_pyramid is None:
            seismic_pyramid = {}

        # Derive indexes from the actual batch size
        # indexes is now a torch.Tensor provided by the batch unpacking above

        rec_in_pyramid: dict[int, torch.Tensor] = {}
        # 1. Ensure rec_noise/amp are initialized for all scales up to max(scales).
        # The generator needs noise tensors for ALL scales (0..scale) to perform
        # a progressive forward pass.
        max_scale = max(scales)
        for s in range(max_scale + 1):
            if len(self.model.rec_noise) <= s:
                self.init_rec_noise_and_amp(
                    s,
                    indexes,
                    facies_pyramid[s],
                    wells_pyramid,
                    seismic_pyramid,
                )

        # 2. Compute reconstruction inputs for the active scales
        for scale in scales:
            rec_in_pyramid[scale] = self.compute_rec_input(
                scale, indexes, facies_pyramid
            )

        epoch = start_epoch

        # Let the model know the current epoch
        self.model._current_epoch = epoch  # type: ignore
        global_step = epoch * self._total_batches + batch_id

        generated_samples: tuple[torch.Tensor, ...] = ()
        scale_metrics = self.optimization_step(
            indexes,
            facies_pyramid,
            rec_in_pyramid,
            wells_pyramid,
            masks_pyramid,
            seismic_pyramid,
        )

        # Visualization logic
        _is_last_step = (
            epoch == self._num_passes - 1 and batch_id == self._total_batches - 1
        )
        _is_viz_epoch = global_step % 200 == 0 or _is_last_step
        if self._is_main_process and _is_viz_epoch:
            generated_samples = self.generate_visualization_samples(
                scales,
                indexes,
                wells_pyramid,
                seismic_pyramid,
            )

        self.handle_epoch_end(
            scales=scales,
            epoch=epoch,
            scale_metrics=scale_metrics,
            generated_samples=generated_samples,
            real_seismic=seismic_pyramid,
            writers=writers,
            outputs_paths=outputs_paths,
            progress=progress,
        )

        _needs_barrier = _is_viz_epoch or (
            self.enable_plot_outputs
            and (epoch % self.save_interval == 0 or epoch == self._num_passes - 1)
            and (epoch != 0 or global_step == 0 or _is_last_step)
            and batch_id == self._total_batches - 1
        )
        if _needs_barrier:
            self._ddp_barrier()

        return scale_metrics

    def optimization_step(
        self,
        indexes: torch.Tensor,
        facies_pyramid: dict[int, torch.Tensor],
        rec_in_pyramid: dict[int, torch.Tensor],
        wells_pyramid: dict[int, torch.Tensor] = {},
        masks_pyramid: dict[int, torch.Tensor] = {},
        seismic_pyramid: dict[int, torch.Tensor] = {},
    ) -> ScaleMetrics:
        """Perform a single optimization step for the model."""
        return self.model(
            self.generator_optimizers,
            self.discriminator_optimizers,
            indexes,
            facies_pyramid,
            rec_in_pyramid,
            wells_pyramid,
            masks_pyramid,
            seismic_pyramid,
        )

    @abstractmethod
    def save_optimizers(
        self,
        scale_path: str,
        generator_optimizer: Any,
        discriminator_optimizer: Any,
        generator_scheduler: Any,
        discriminator_scheduler: Any,
    ) -> None:
        raise NotImplementedError("Subclasses must implement save_optimizers")

    @abstractmethod
    def save_epoch_checkpoint(
        self,
        scales: tuple[int, ...],
        scale_paths: dict[int, str],
        epoch: int,
        batch_id: int,
    ) -> None:
        """Save a mid-training checkpoint so training can resume from *epoch*.

        The checkpoint must capture model weights, optimizer/scheduler
        states and the noise amplitudes / reconstruction noise for the
        scales currently being trained.

        Parameters
        ----------
        scales : tuple[int, ...]
            Scale indices being trained in the current group.
        scale_paths : dict[int, str]
            Per-scale output directory paths.
        epoch : int
            The *next* epoch to run (i.e. training completed up to
            ``epoch - 1``).
        batch_id : int
            Current batch index within the DataLoader iteration.
        """
        raise NotImplementedError("Subclasses must implement save_epoch_checkpoint")

    @abstractmethod
    def load_epoch_checkpoint(
        self,
        scales: tuple[int, ...],
        scale_paths: dict[int, str],
    ) -> tuple[int, int]:
        """Restore a previously saved epoch checkpoint.

        Parameters
        ----------
        scales : tuple[int, ...]
            Scale indices being trained in the current group.
        scale_paths : dict[int, str]
            Per-scale output directory paths (used to locate the checkpoint
            file).

        Returns
        -------
        tuple[int, int]
            ``(start_epoch, resume_batch_id)`` — the epoch and batch id
            to resume from.  Returns ``(0, 0)`` when no checkpoint is
            found.
        """
        raise NotImplementedError("Subclasses must implement load_epoch_checkpoint")

    def load(self, path: str, until_scale: int | None = None) -> None:
        """Load saved models and set the starting scale for training.

        Parameters
        ----------
        path : str
            Path to the directory containing model checkpoint files.
        until_scale : int | None, optional
            Load models up to and including this scale. If None, loads all
            available scales. Defaults to None.
        """
        self.start_scale = self.model.load(
            path,
            load_shapes=False,
            until_scale=until_scale,
            load_discriminator=True,
        )

    @abstractmethod
    def load_model(self, scale: int) -> None:
        """Load generator and discriminator state dicts for a specific scale.

        Parameters
        ----------
        scale : int
            Scale index to load the model for.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement load_model")

    def load_optimizers(
        self,
        scale: int,
        scale_path: str,
        generator_optimizer: Any,
        discriminator_optimizer: Any,
        generator_scheduler: Any,
        discriminator_scheduler: Any,
    ) -> None:
        """Load optimizer and scheduler state dicts from disk using project filename
        constants from :mod:`config`.

        Parameters
        ----------
        scale : int
            Scale index to load the optimizers for.
        scale_path : str
            Path to the scale directory where optimizers are saved.
        generator_optimizer : Any
            Generator optimizer instance to load state into.
        discriminator_optimizer : Any
            Discriminator optimizer instance to load state into.
        generator_scheduler : Any
            Generator learning rate scheduler instance to load state into.
        discriminator_scheduler : Any
            Discriminator learning rate scheduler instance to load state into.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement load_optimizers")

    def save_generated_outputs(
        self,
        scales: tuple[int, ...],
        epoch: int,
        batch_id: int,
        outputs_path: dict[int, str],
    ) -> None:
        """Persist generated facies for a given scale.

        Concrete trainers that can generate and save facies (e.g. torch)
        should implement this method. The base implementation is a
        no-op / hook and may be overridden.

        Parameters
        ----------
        scales : tuple[int, ...]
            Tuple of scale indices to save generated facies for.
        epoch : int
            Current epoch index.
        batch_id : int
            Current batch index.
        outputs_path : dict[int, str]
            Dictionary of paths to the outputs directories for the current scales.
        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement save_generated_outputs")

    def handle_epoch_end(
        self,
        scales: tuple[int, ...],
        epoch: int,
        scale_metrics: ScaleMetrics,
        generated_samples: tuple[torch.Tensor, ...],
        writers: dict[int, SummaryWriter],  # type: ignore
        outputs_paths: dict[int, str],
        progress: "tqdm[Any]",  # type: ignore
        real_seismic: dict[int, torch.Tensor] | None = None,
    ) -> None:
        """Shared end-of-epoch bookkeeping for trainers.

        This consolidates visualization updates, metric printing,
        TensorBoard logging, optional facies saving and scheduler steps.

        Parameters
        ----------
        scales : tuple[int, ...]
            Tuple of scale indices being trained.
        epoch : int
            Current epoch index.
        scale_metrics : ScaleMetrics
            Collected metrics for the current epoch.
        generated_samples : tuple[torch.Tensor, ...]
            Tuple of generated facies samples for visualization.
        writers : dict[int, SummaryWriter]
            Dictionary of per-scale TensorBoard writers.
        outputs_paths : dict[int, str]
            Dictionary of per-scale outputs directory paths.
        progress : tqdm[Any]
            Progress bar instance to update.
        """
        # Use a globally unique step so that different DataLoader
        # batches do not overwrite each other's TensorBoard entries.
        global_step = epoch * self._total_batches + self._current_batch_id
        samples_processed = self.batch_size * global_step
        _is_last_step = (
            epoch == self._num_passes - 1
            and self._current_batch_id == self._total_batches - 1
        )

        # ── All ranks participation ──────────────────────────────────
        # Check if we need to gather indices for plotting (all ranks must agree
        # to avoid hanging during the collective all_gather).
        _should_gather = (
            (epoch % self.save_interval == 0 or epoch == self._num_passes - 1)
            and (epoch != 0 or self._num_passes == 1)
            and (self._current_batch_id == self._total_batches - 1)
        )
        if _should_gather:
            self._seen_indices_for_save = self.collect_seen_indices()

        # ── Rank-0 only I/O ─────────────────────────────────────────
        if self._is_main_process:
            # Visualizer update
            if self.enable_tensorboard and self.visualizer:
                self.visualizer.update(  # type: ignore
                    global_step,
                    scale_metrics,
                    generated_samples,
                    samples_processed,
                    scales=scales,
                    real_seismic=real_seismic,
                )

            # Print formatted metrics table every 100 batches, at the end of the epoch, or at the very start.
            _print_table = (global_step % 100 == 0) or (
                self._current_batch_id == self._total_batches - 1
            )
            if _print_table:
                self._print_metrics_table(scales, scale_metrics)

            # Save to TensorBoard and log per-scale (only when TB is
            # enabled — each call does a GPU→CPU sync via torch.stack().tolist();
            # throttled to every _tb_log_interval global steps to reduce host syncs).
            _log_tb = global_step % self._tb_log_interval == 0 or _is_last_step
            if self.enable_tensorboard and _log_tb:
                for scale in scales:
                    g = scale_metrics.generator[scale]
                    d = scale_metrics.discriminator[scale]
                    self.log_epoch(writers[scale], epoch, g, d, global_step)  # type: ignore
                    # Log learning rates per scale.
                    # ReduceLROnPlateau doesn't have get_last_lr(); read
                    # the optimizer param group directly.
                    lr_g = self.generator_optimizers[scale].param_groups[0]["lr"]  # type: ignore[index]
                    lr_d = self.discriminator_schedulers[scale].get_last_lr()[0]  # type: ignore[union-attr]
                    writers[scale].add_scalar("LearningRate/generator", lr_g, global_step)  # type: ignore
                    writers[scale].add_scalar("LearningRate/discriminator", lr_d, global_step)  # type: ignore

            # Log when learning rate decays (at every lr_decay interval,
            # before schedulers_step advances the count).
            _decay_counter = (
                global_step
                if self.lr_decay_unit == "step"
                else self._current_batch_id if self.lr_decay_unit == "batch" else epoch
            )
            if (
                self.lr_decay > 0
                and _decay_counter > 0
                and _decay_counter % self.lr_decay == 0
            ):
                lr_d_before = self.discriminator_schedulers[scales[0]].get_last_lr()[0]  # type: ignore[union-attr]
                lr_d_after = lr_d_before * self.gamma  # type: ignore[operator]
                _unit_label = (
                    f"step {global_step}"
                    if self.lr_decay_unit == "step"
                    else (
                        f"batch {self._current_batch_id}"
                        if self.lr_decay_unit == "batch"
                        else f"epoch {epoch}"
                    )
                )
                progress.write(  # type: ignore
                    f"\n  ⚡ LR decay at {_unit_label}: "
                    f"lr_d {lr_d_before:.2e} → {lr_d_after:.2e} "
                    f"(gamma={self.gamma})"
                )

            # Only rank 0 saves outputs at configured intervals
            if _should_gather:
                self.save_generated_outputs(
                    scales, epoch, self._current_batch_id, outputs_paths
                )

        # ── All ranks ─────────────────────────────────────────────────
        # Step schedulers — generator uses smoothed loss via
        # ReduceLROnPlateau, discriminator uses StepLR (epoch or step based).
        self.schedulers_step(scales, scale_metrics)

    def schedulers_step(
        self,
        scales: tuple[int, ...],
        scale_metrics: ScaleMetrics | None = None,
    ) -> None:
        """Step the learning-rate schedulers for the provided scales.

        The generator scheduler is :class:`ReduceLROnPlateau` and receives
        the EMA-smoothed generator total loss.  The discriminator scheduler
        remains a plain :class:`StepLR`.

        Parameters
        ----------
        scales : tuple[int, ...]
            Tuple of scale indices to step the schedulers for.
        scale_metrics : ScaleMetrics, optional
            Current epoch metrics.  When provided the smoothed generator
            loss is fed to ``ReduceLROnPlateau.step(metric)``.
        """
        from metrics import MetricSmoother

        if not hasattr(self, "_g_loss_smoothers"):
            self._g_loss_smoothers: dict[int, MetricSmoother] = {}

        alpha = getattr(self.options, "lr_smoothing_alpha", 0.9)

        for scale in scales:
            # Generator: ReduceLROnPlateau with smoothed loss
            if scale_metrics is not None and scale in scale_metrics.generator:
                g = scale_metrics.generator[scale]
                raw_loss = g.total.item()  # type: ignore[union-attr]
                if scale not in self._g_loss_smoothers:
                    self._g_loss_smoothers[scale] = MetricSmoother(alpha=alpha)
                smoothed = self._g_loss_smoothers[scale].update(raw_loss)
                self.generator_schedulers[scale].step(smoothed)  # type: ignore[arg-type]
            else:
                # Fallback for subclasses that don't pass metrics
                self.generator_schedulers[scale].step()  # type: ignore[arg-type]
            # Discriminator: plain StepLR
            self.discriminator_schedulers[scale].step()

    def _release_accelerator_memory(self) -> None:
        """Release unused accelerator (GPU) memory back to the OS.

        Calls the caching allocator to release unused blocks.
        ``empty_cache()`` already triggers an implicit device sync, so
        an explicit ``synchronize()`` is unnecessary.  GC is skipped
        because this is only called between scale groups and the
        allocator handles freed tensors without a Python GC pass.
        """
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def train(self) -> None:
        """Train the FaciesGAN model with parallel scale training.

        Trains multiple pyramid scales simultaneously in groups. Processes
        scales in batches of num_parallel_scales at a time.

        When running under DDP only the main process (``_is_main_process``)
        performs I/O (directory creation, model saving, TensorBoard logging,
        progress bar updates).  All ranks participate in model init,
        optimizer setup and training iterations.
        """
        start_train_time = time.time()

        # Train scales in parallel groups
        scale = self.start_scale
        while scale <= self.stop_scale:
            # Determine how many scales to train in this parallel group
            num_scales_in_group = min(
                self.num_parallel_scales, self.stop_scale - scale + 1
            )

            scales_to_train = tuple(range(scale, scale + num_scales_in_group))
            if self._is_main_process:
                print(f"\n{'='*60}")
                print(f"Training scales {scales_to_train} in parallel")
                print(f"{'='*60}\n")

            group_start_time = time.time()

            # Initialize all scales in the group (all ranks)
            self.model.init_scales(scale, num_scales_in_group)

            # Freeze generator blocks from previous groups so backward()
            # does not allocate gradient tensors on them.  The forward
            # pass still runs through these scales (progressive
            # synthesis), but without requires_grad the autograd engine
            # skips gradient storage, saving significant CUDA memory.
            self.model.freeze_generator_scales(scales_to_train)

            # Discard rec_noise for scales outside the current group to
            # free CUDA memory.  Only noise for the active scales (and
            # the scales needed by the progressive forward pass) is
            # kept — the rest is regenerated if needed later.
            self.model.trim_rec_noise(min(scales_to_train))

            # Prune optimizers/schedulers from previous groups to free
            # Adam state buffers and avoid training frozen scales.
            for s in list(self.generator_optimizers):
                if s not in scales_to_train:
                    del self.generator_optimizers[s]
                    del self.generator_schedulers[s]
            for s in list(self.discriminator_optimizers):
                if s not in scales_to_train:
                    del self.discriminator_optimizers[s]
                    del self.discriminator_schedulers[s]

            # Setup optimizers for active scales (all ranks)
            self.setup_optimizers(scales_to_train)

            # Create directories for all scales (use dict to map scale -> path)
            scale_paths: dict[int, str] = {
                s: os.path.join(self.output_path, str(s)) for s in scales_to_train
            }
            outputs_paths: dict[int, str] = {
                s: os.path.join(scale_paths[s], OUTPUT_FACIES_PATH)
                for s in scales_to_train
            }

            # Only main process creates directories, writers
            writers: dict[int, SummaryWriter] = {}  # type: ignore
            if self._is_main_process:
                for s in scales_to_train:
                    utils.create_dirs(scale_paths[s])
                    utils.create_dirs(outputs_paths[s])

            if self.fine_tuning:
                for s in scales_to_train:
                    self.load_model(s)

            # ── Epoch-level resume ──────────────────────────────────
            # When start_epoch > 0 for the *first* scale group, load the
            # epoch checkpoint (model weights + optimiser/scheduler states)
            # and figure out which batch to resume from.
            resume_epoch: int = 0
            resume_batch_id: int = 0
            # base_epoch: the epoch that the current scale group should start
            # from.  When resuming mid-batch, batches after the resumed
            # one already completed up to the previous full run's epoch
            # count (recorded in completed_epoch.txt).
            base_epoch: int = self.start_epoch
            if self.start_epoch > 0:
                resume_epoch, resume_batch_id = self.load_epoch_checkpoint(
                    scales_to_train,
                    scale_paths,
                )
                if resume_epoch > 0:
                    base_epoch = resume_epoch
                    if self._is_main_process:
                        print(
                            f"Epoch checkpoint loaded: resuming from batch {resume_batch_id}, "
                            f"epoch {resume_epoch}"
                        )
                elif self._is_main_process:
                    print(
                        f"Resume requested (start_epoch={self.start_epoch}) but no epoch checkpoint found. "
                        "Starting scale group from scratch."
                    )

                if resume_batch_id > 0:
                    # Optional: Verify or override base_epoch from completed_epoch.txt
                    from config import COMPLETED_EPOCH_FILE

                    meta_path = os.path.join(
                        scale_paths[min(scales_to_train)], COMPLETED_EPOCH_FILE
                    )
                    if os.path.isfile(meta_path):
                        with open(meta_path) as f:
                            # The file contains the last FULLY completed epoch.
                            # We want to start the next iteration at base_epoch + 1
                            # if the checkpoint and file are consistent.
                            last_full_epoch = int(f.read().strip())
                            base_epoch = max(base_epoch, last_full_epoch + 1)

            # Create per-scale TensorBoard writers AFTER resume logic so
            # purge_step can use the correct global_step value.
            if self._is_main_process and self.enable_tensorboard:
                if resume_epoch > 0:
                    _purge: int | None = resume_batch_id * self.num_iter + resume_epoch
                else:
                    _purge = None
                writers = {
                    s: SummaryWriter(log_dir=scale_paths[s], purge_step=_purge)
                    for s in scales_to_train
                }

            # Progress bar for all batches and epochs in this group
            total_batches = len(self.data_loader)
            progress_total = self._num_passes * total_batches
            progress = tqdm(
                total=progress_total,
                position=0,
                disable=not self._is_main_process,
                delay=30,
            )

            # Use create_batch_iterator to allow subclasses to inject prefetching logic
            max_scale = max(scales_to_train)
            all_scales = tuple(range(max_scale + 1))

            # Outer loop: Epochs (Passes)
            for epoch in range(base_epoch, self._num_passes):
                self._current_pass_idx = epoch
                sampler = getattr(self.data_loader, "sampler", None)
                if sampler is not None and hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)
                # Inner loop: Batches
                batch_iterator = self.create_batch_iterator(
                    self.data_loader, all_scales
                )
                for batch_id, batch in enumerate(batch_iterator):
                    # Skip batches already processed before the resume point
                    if epoch == base_epoch and batch_id < resume_batch_id:
                        continue

                    # Reset LR schedulers so each batch trains with the same schedule
                    if self.lr_decay_unit not in ("step", "batch"):
                        self.reset_schedulers(scales_to_train)

                    self._total_batches = total_batches
                    self._current_batch_id = batch_id

                    # For each batch, we run the model logic for the current epoch
                    # NOTE: train_scales now only handles ONE epoch per call
                    self.train_scales(
                        scales=scales_to_train,
                        writers=writers,
                        scale_paths=scale_paths,
                        outputs_paths=outputs_paths,
                        batch_id=batch_id,
                        progress=progress,
                        facies_pyramid=None,
                        wells_pyramid=None,
                        masks_pyramid=None,
                        seismic_pyramid=None,
                        start_epoch=epoch,
                        batch=batch,  # Pass the batch directly
                    )

                    if self._is_main_process:
                        progress.set_description(
                            f"Epoch {epoch}/{self._num_passes}, Batch {batch_id}/{total_batches}"
                        )
                        progress.update(1)

                if self._is_main_process:
                    # ── Save Epoch Progress (Periodic or Final) ───────
                    interval = getattr(self.options, "checkpoint_interval", 1)
                    is_final_epoch = epoch == self._num_passes - 1
                    should_save = (
                        interval > 0 and epoch % interval == interval - 1
                    ) or is_final_epoch

                    if should_save:
                        for s in scales_to_train:
                            self.model.save_scale(s, scale_paths[s])
                            self.save_optimizers(
                                scale_paths[s],
                                self.generator_optimizers[s],
                                self.discriminator_optimizers[s],
                                self.generator_schedulers[s],
                                self.discriminator_schedulers[s],
                            )
                            # Record the actual last epoch completed
                            from config import COMPLETED_EPOCH_FILE

                            meta_path = os.path.join(
                                scale_paths[s], COMPLETED_EPOCH_FILE
                            )
                            with open(meta_path, "w") as f:
                                f.write(str(epoch + 1))

                        # Save the monolithic epoch checkpoint
                        self.save_epoch_checkpoint(
                            scales_to_train,
                            scale_paths,
                            epoch + 1,
                            0,
                        )

            if self._is_main_process:
                progress.close()  # type: ignore

            # Flush any pending background plot jobs
            if self._is_main_process and self.enable_plot_outputs:
                try:
                    from background_workers import BackgroundWorker

                    BackgroundWorker().wait_pending()
                except Exception:
                    pass

            # Synchronise so non-zero ranks wait for rank 0 to finish
            # saving before proceeding to the next scale group (or to
            # DDP teardown at the end of training).
            self._ddp_barrier()

            # Close writers (rank 0 only)
            for writer in writers.values():
                writer.close()

            # Release GPU memory cached by the allocator between groups
            self._release_accelerator_memory()

            group_end_time = time.time()
            elapsed = log.format_time(int(group_end_time - group_start_time))
            if self._is_main_process:
                print(f"\nScales {scales_to_train} training time: {elapsed}")

            scale += num_scales_in_group

        end_train_time = time.time()
        if self._is_main_process:
            print(
                "\nTotal training time:",
                log.format_time(int(end_train_time - start_train_time)),
            )

        # Close TensorBoard writer
        if self.enable_tensorboard and self.visualizer:
            self.visualizer.close()
        if self._is_main_process:
            print("\n✅ Training complete!")
        if self.enable_tensorboard:
            print("\n📊 View outputs in TensorBoard (if still running)")

        # Final DDP barrier: ensure all ranks have finished training
        # before returning so the caller can safely tear down the
        # process group without one rank still doing I/O.
        self._ddp_barrier()

    def log_epoch(
        self,
        writer: SummaryWriter,  # type: ignore
        epoch: int,
        generator_metrics: GeneratorMetrics,
        discriminator_metrics: DiscriminatorMetrics,
        global_step: int | None = None,
    ) -> None:
        """Log training metrics for the current epoch to TensorBoard and console.

        Parameters
        ----------
        writer : SummaryWriter
            Per-scale TensorBoard writer to record scalars.
        epoch : int
            Current epoch index (0-based).
        generator_metrics : GeneratorMetrics
            Dataclass carrying tensor-valued generator losses for the scale.
        discriminator_metrics : DiscriminatorMetrics
            Dataclass carrying tensor-valued discriminator losses for the scale.

        Notes
        -----
        Metric dataclass fields are tensor scalars; this function batch-converts
        them to Python floats via ``torch.stack().tolist()`` (single GPU sync)
        before writing to TensorBoard or formatting for display.
        """
        g = generator_metrics
        d = discriminator_metrics

        vals = self._get_metric_floats(g, d)
        (
            g_total,
            g_fake,
            g_facies_rec,
            g_well,
            g_div,
            g_rec_rock_physics,
            g_tv,
            g_elastic,
            g_physics,
        ) = vals[:9]
        d_total, d_real, d_fake, d_gp = vals[9:]

        global_step = global_step if global_step is not None else epoch

        writer.add_scalar("G/Total", g_total, global_step)
        writer.add_scalar("G/Adv", g_fake, global_step)
        writer.add_scalar("G/Facies_Rec", g_facies_rec, global_step)
        writer.add_scalar("G/Well", g_well, global_step)
        writer.add_scalar("G/Diversity", g_div, global_step)
        writer.add_scalar("G/Rec_Rock_Physics", g_rec_rock_physics, global_step)
        writer.add_scalar("G/TV_Smoothness", g_tv, global_step)
        writer.add_scalar("G/Elastic", g_elastic, global_step)
        writer.add_scalar("G/Physics_Seismic", g_physics, global_step)
        writer.add_scalar("D/Total", d_total, global_step)
        writer.add_scalar("D/Real", d_real, global_step)
        writer.add_scalar("D/Fake", d_fake, global_step)
        writer.add_scalar("D/GP", d_gp, global_step)

        step = global_step

        # Log to TensorBoard - discriminator losses
        writer.add_scalar("Loss/train/discriminator/real", -d_real, step)  # type: ignore
        writer.add_scalar("Loss/train/discriminator/fake", d_fake, step)  # type: ignore
        writer.add_scalar(  # type: ignore
            "Loss/train/discriminator/gradient_penalty", d_gp, step
        )
        writer.add_scalar("Loss/train/discriminator", d_total, step)  # type: ignore

        # Log to TensorBoard - generator losses
        writer.add_scalar("Loss/train/generator/adversarial", g_fake, step)  # type: ignore
        writer.add_scalar("Loss/train/generator/facies_rec", g_facies_rec, step)  # type: ignore
        writer.add_scalar("Loss/train/generator/well_constraint", g_well, step)  # type: ignore
        writer.add_scalar("Loss/train/generator/diversity", g_div, step)  # type: ignore
        writer.add_scalar("Loss/train/generator/rec_rock_physics", g_rec_rock_physics, step)  # type: ignore
        writer.add_scalar("Loss/train/generator/tv_smoothness", g_tv, step)  # type: ignore
        writer.add_scalar("Loss/train/generator", g_total, step)  # type: ignore

    def _print_metrics_table(
        self, scales: tuple[int, ...], scale_metrics: ScaleMetrics
    ) -> None:
        """Print formatted ASCII tables for Generator and Discriminator metrics."""

        from metrics import MetricSmoother as _MS

        # Lazily initialise one smoother per scale per metric.
        _sm_alpha = getattr(self.options, "lr_smoothing_alpha", 0.9)
        if not hasattr(self, "_table_smoothers"):
            self._table_smoothers: dict[int, list[_MS]] = {}

        for scale in scales:
            if scale not in self._table_smoothers:
                # 9 G-metrics + 4 D-metrics = 13 total
                self._table_smoothers[scale] = [_MS(alpha=_sm_alpha) for _ in range(13)]

        lines: list[str] = [""]
        lines.append("")
        # --- Generator Table ---
        lines.append("  Generator Metrics:")
        lines.append("  ┌" + "─" * 124 + "┐")
        lines.append(
            (
                f"  │ {'Scale':^5} │ {'G_total':^10} │ {'G_adv':^10} │ {'G_fa_rec':^10} │ "
                f"{'G_well':^10} │ {'G_div':^10} │ {'G_rp_rec':^10} │ {'G_tv':^10} │ {'G_el':^10} │ {'G_ph':^10} │"
            )
        )
        lines.append("  ├" + "─" * 124 + "┤")

        cached_v: list[list[float]] = []

        for scale in scales:
            g = scale_metrics.generator[scale]
            d = scale_metrics.discriminator[scale]
            raw = self._get_metric_floats(g, d)

            sm = self._table_smoothers[scale]
            v = [sm[i].update(raw[i]) for i in range(13)]
            cached_v.append(v)

            lines.append(
                (
                    f"  │ {scale:^5d} │ {v[0]:>10.4f} │ {v[1]:>10.4f} │ {v[2]:>10.4f} │ "
                    f"{v[3]:>10.4f} │ {v[4]:>10.4f} │ {v[5]:>10.4f} │ {v[6]:>10.4f} │ {v[7]:>10.4f} │ {v[8]:>10.4f} │"
                )
            )
        lines.append("  └" + "─" * 124 + "┘")

        # --- Discriminator Table ---
        lines.append("  Discriminator Metrics:")
        lines.append("  ┌" + "─" * 59 + "┐")
        lines.append(
            (
                f"  │ {'Scale':^5} │ {'D_total':^10} │ {'D_real':^10} │ "
                f"{'D_fake':^10} │ {'D_gp':^10} │"
            )
        )
        lines.append("  ├" + "─" * 59 + "┤")

        for i, scale in enumerate(scales):
            v = cached_v[i]
            lines.append(
                (
                    f"  │ {scale:^5d} │ {v[9]:>10.4f} │ {v[10]:>10.4f} │ "
                    f"{v[11]:>10.4f} │ {v[12]:>10.4f} │"
                )
            )
        lines.append("  └" + "─" * 59 + "┘")
        print("\n".join(lines), flush=True)

    def _print_facie_shapes_table(self) -> None:
        """Print an ASCII table of the generated facies dimensions across scales."""
        lines = ["Generated data shapes:"]
        lines.append("╔══════════╦══════════╦══════════╦══════════╦══════════╗")
        lines.append(
            "║ {:^8} ║ {:^8} ║ {:^8} ║ {:^8} ║ {:^8} ║".format(
                "Batch", "In Ch", "Out Ch", "Height", "Width"
            )
        )
        lines.append("╠══════════╬══════════╬══════════╬══════════╬══════════╣")
        for shape in self.scales:
            lines.append(
                "║ {:^8} ║ {:^8} ║ {:^8} ║ {:^8} ║ {:^8} ║".format(
                    shape[0],
                    self.noise_channels,
                    self.total_output_channels,
                    shape[2],
                    shape[3],
                )
            )
        lines.append("╚══════════╩══════════╩══════════╩══════════╩══════════╝")
        print("\n".join(lines), flush=True)

    def _get_metric_floats(
        self, g: GeneratorMetrics, d: DiscriminatorMetrics
    ) -> list[float]:
        """Convert metric tensors to a flat list of Python floats (single sync)."""
        import torch as _t

        return _t.stack([*g.as_tuple(), *d.as_tuple()]).tolist()  # type: ignore[arg-type]
        return _t.stack([*g.as_tuple(), *d.as_tuple()]).tolist()  # type: ignore[arg-type]
