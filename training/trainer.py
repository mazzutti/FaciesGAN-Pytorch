"""Trainer for multiscale FaciesGAN.

Trains multiple pyramid scales simultaneously in parallel groups.
Each scale keeps its own discriminator and optimizer while the shared
generator is managed by the central :class:`models.facies_gan.FaciesGAN`
instance attached to this trainer.

Notes
-----
- For efficiency the trainer typically uses a single data batch per group
  of scales (the DataLoader yields batches of pyramids and a group consumes
  one batch to train all its scales in parallel).
- The trainer stores per-scale reconstruction noise and noise amplitudes in
  the model's ``rec_noise`` and ``noise_amp`` lists respectively.
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from collections.abc import Iterator
from typing import IO, Any, cast

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter  # type: ignore
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import background_workers as bw
import log
import utils
from apex_utils import FusedAdam
from config import CheckpointFilenames, ExperimentPaths
from datasets.data_prefetcher import DataPrefetcher, gather_seen_indices
from datasets.dataset import PyramidsDataset
from device import device_manager
from enums import DeviceType, TimeUnit
from models import utils as model_utils
from models.facies_gan import FaciesGAN, unwrap_ddp
from models.utils import ChannelKey
from options import TrainingOptions
from tensorboard_visualizer import TensorBoardVisualizer
from training.checkpoint import Checkpoint, ScaleCheckpoint
from training.metrics import (
    DiscriminatorMetrics,
    GeneratorMetrics,
    IterableMetrics,
    MetricSmoother,
    ScaleMetrics,
    compute_masked_loss,
)
from typedefs import Batch, IDataLoader, PyramidsBatch, RawBatch


class DummyProgress:
    def update(self, n: int = 1) -> None:
        pass

    def set_description(self, desc: str) -> None:
        pass

    @staticmethod
    def write(s: str) -> None:
        print(s)

    def close(self) -> None:
        pass


# noinspection PyDefaultArgument,PyBroadException
class Trainer:
    """Trainer for multiscale progressive FaciesGAN.

    Manages simultaneous training of multiple pyramid scales by grouping
    scales and training each group in parallel. Each scale keeps its own
    discriminator and optimizer while the shared generator is exposed via
    the :class:`models.facies_gan.FaciesGAN` instance attached to this
    trainer.

    Parameters
    ----------
    options : TrainingOptions
        Training configuration containing hyperparameters and paths.
    fine_tuning : bool, optional
        Whether to load and fine-tune from existing checkpoints.
    checkpoint_path : str, optional
        Base path used to load/save per-scale checkpoints.

    Attributes
    ----------
    model : FaciesGAN
        The multiscale model instance managed by the trainer.

    Notes
    -----
    - The trainer updates ``self.model.rec_noise`` and ``self.model.noise_amp``
      as part of noise initialization (see :meth:`init_rec_noise_and_amp`).
    - Conditioning tensors (wells/seismic) are expected channels-last when
      prepared and returned by :meth:`TorchDataPrefetcher`.
    """

    model: FaciesGAN

    def __init__(
        self,
        options: TrainingOptions,
        fine_tuning: bool = False,
        checkpoint_path: str = ".checkpoints",
    ) -> None:
        """Create a Trainer instance and prepare datasets, model and logging.

        Parameters
        ----------
        options : TrainingOptions
            Training options with hyperparameters and paths.
        fine_tuning : bool, optional
            Whether to attempt to load existing checkpoints, by default False.
        checkpoint_path : str, optional
            Base path for checkpoint files, by default ".checkpoints/".
        """

        # ── Shared initialisation ──────────────────────────────────────
        self.options: TrainingOptions = options
        self.fine_tuning: bool = fine_tuning
        self.checkpoint_path: str = checkpoint_path

        self.start_scale: int = options.start_scale
        self.stop_scale: int = options.stop_scale
        self.output_path: str = options.output_path
        self.num_iter: int = 1  # inner per-batch epoch loop always runs once
        self.save_interval: int = options.save_interval
        self.num_parallel_scales: int = options.num_parallel_scales
        # Total dataset passes == the user's --num-iter value.
        self._num_passes: int = options.num_iter
        self._current_pass_idx: int = 0

        # How often to flush TensorBoard scalars (epochs).
        self._tb_log_interval: int = 10

        self.batch_size: int = (
            options.batch_size
            if (options.batch_size < options.num_train_pyramids)
            else options.num_train_pyramids
        )

        self.enable_tensorboard: bool = options.enable_tensorboard
        self.enable_plot_outputs: bool = options.enable_plot_outputs
        self.visualizer: TensorBoardVisualizer | None = None

        self.channels = model_utils.calculate_channels(options)
        self.noise_channels = self.channels[ChannelKey.NOISE]
        self.total_output_channels = self.channels[ChannelKey.GENERATOR_OUT]

        self.num_real_facies: int = options.num_real_facies
        self.num_generated_per_real: int = options.num_generated_per_real
        self.wells_mask_columns: tuple[int, ...] = options.wells_mask_columns

        self.lr_g: float = options.lr_g
        self.lr_d: float = options.lr_d
        self.beta1: float = options.beta1
        self.lr_decay: int = options.lr_decay
        self.gamma: float = options.gamma

        self.residual_padding: int = options.num_layer * math.floor(
            options.kernel_size / 2
        )

        dataset, scales = self.init_dataset()
        self.dataset: PyramidsDataset = dataset
        self.num_of_batchs: int = len(self.dataset) // self.batch_size
        self.scales: tuple[tuple[int, ...], ...] = scales
        self.data_loader: IDataLoader = self.create_dataloader()

        if device_manager.is_main_process:
            print(f"DataLoader num_workers: {self.data_loader.num_workers}")

        self.model: FaciesGAN = self.create_model()
        self.model.shapes = self.scales

        # Noise calibration buffers from model (registered in FaciesGAN)
        # Use .item() if a float is strictly required in the training loop.
        self.noise_amp = self.model.noise_amp
        self.min_noise_amp = self.model.min_noise_amp
        self.scale0_noise_amp = self.model.scale0_noise_amp

        # Time unit for scheduling and intervals: 'epoch' or 'step'/'batch'
        self.time_unit: str = getattr(options, "time_unit", TimeUnit.STEP)
        self.gamma = options.gamma

        self.generator_optimizers: dict[int, torch.optim.Optimizer] = {}
        self.discriminator_optimizers: dict[int, torch.optim.Optimizer] = {}
        self.generator_schedulers: dict[int, torch.optim.lr_scheduler.LRScheduler] = {}
        self.discriminator_schedulers: dict[
            int, torch.optim.lr_scheduler.LRScheduler
        ] = {}

        self._seen_indices_for_save: list[tuple[int, ...]] = []

        if device_manager.is_main_process:
            self._print_facie_shapes_table()

        self.enable_tensorboard = options.enable_tensorboard
        self.enable_plot_outputs = options.enable_plot_outputs
        if self.enable_tensorboard and device_manager.is_main_process:
            viz_path = os.path.join(self.output_path, "training_visualizations")
            log_dir = os.path.join(self.output_path, "tensorboard_logs")
            dataset_info = f"{len(self.dataset)} pyramids, {self.batch_size} batch size"
            if len(options.wells_mask_columns) > 0:
                dataset_info += f", wells: {options.wells_mask_columns}"

            _purge: int | None = None

            self.visualizer = TensorBoardVisualizer(
                num_scales=self.stop_scale - self.start_scale + 1,
                output_dir=viz_path,
                log_dir=log_dir,
                update_interval=self._tb_log_interval,
                image_log_interval=options.save_interval,
                dataset_info=dataset_info,
                purge_step=_purge,
                num_facies=options.num_facies_channels,
                has_rp=options.use_rock_physics,
                physics_state=(
                    self.model.physics_state if options.use_rock_physics else None
                ),
                normalization_range=(
                    float(options.normalization_range[0]),
                    float(options.normalization_range[1]),
                ),
                seismic_stretch_percentile=options.seismic_stretch_percentile,
            )
            print("📊 TensorBoard logging enabled")
            print(f"   logdir: {log_dir}")
            print("   URL: http://localhost:6006")
        else:
            self.visualizer = None  # type: ignore
            if device_manager.is_main_process:
                print("📊 TensorBoard logging disabled")

        # ── Trainer-specific fields ────────────────────────────────────
        self._ckpt_thread: threading.Thread | None = None
        self._g_loss_smoother: dict[int, MetricSmoother] = {}
        self._compile_warmed_up_scales: set[tuple[int, ...]] = set()
        self._batch_prefetcher: DataPrefetcher | None = None

    def _is_time_to_act(
        self, epoch: int, batch_id: int, interval: int, is_final: bool
    ) -> bool:
        """Check if the current iteration/epoch matches the interval under the configured TimeUnit.

        Parameters
        ----------
        epoch : int
            Current epoch (0-indexed).
        batch_id : int
            Current batch index (0-indexed).
        interval : int
            The configuration interval parameter.
        is_final : bool
            True if this is the final step/epoch of training.
        """
        if interval <= 0:
            return is_final

        if self.time_unit == TimeUnit.EPOCH:
            # Epoch-based: only check at the end of an epoch
            is_epoch_end = batch_id == self._total_batches - 1
            if not is_epoch_end:
                return False
            completed_epochs = epoch + 1
            return (completed_epochs % interval == 0) or is_final
        else:
            # Step/Batch-based: check at every iteration
            completed_steps = epoch * self._total_batches + batch_id + 1
            return (completed_steps % interval == 0) or is_final

    def _warmup_compile_traces(
        self, scales: tuple[int, ...], batch: PyramidsBatch
    ) -> None:
        """Trigger one-time JIT compilations for active scales before training."""
        if not self.model.use_compile or scales in self._compile_warmed_up_scales:
            return

        self._compile_warmed_up_scales.add(scales)
        if device_manager.is_main_process:
            print(
                f"  [warmup] Warming up JIT compilation traces for scales {scales}..."
            )

        # Run a single optimization step (with no grad update) to trace the model.
        # We use a clone of the optimizers to avoid side effects if possible,
        # but since we are just warming up, we can also just do it carefully.
        with torch.no_grad():
            self.model.eval()
            try:
                (
                    indexes,
                    facies_pyramid,
                    wells_pyramid,
                    masks_pyramid,
                    seismic_pyramid,
                ) = batch
                rec_in_pyramid: dict[int, torch.Tensor] = {}
                for s in range(max(scales) + 1):
                    if len(self.model.rec_noise) <= s:
                        self.init_rec_noise_and_amp(
                            s,
                            indexes,
                            facies_pyramid[s],
                            wells_pyramid or {},
                            seismic_pyramid or {},
                        )
                for scale in scales:
                    rec_in_pyramid[scale] = self.compute_rec_input(
                        scale, indexes, facies_pyramid
                    )

                # Trigger generator traces
                self.model.generator(
                    self.model.get_pyramid_noise(
                        max(scales), indexes, wells_pyramid, seismic_pyramid
                    ),
                    self.model.noise_amps,
                    stop_scale=max(scales),
                )

                # Trigger discriminator traces
                for scale in scales:
                    self.model.discriminator.discs[scale](facies_pyramid[scale])

                # Trigger a dummy call to compute_generator_metrics to satisfy linting
                # and ensure masks_pyramid is functionally traced (if compiled in future).
                if scales and 0 in masks_pyramid:
                    with torch.no_grad():
                        _ = compute_masked_loss(
                            facies_pyramid[0],
                            facies_pyramid[0],
                            wells_pyramid.get(0),
                            masks_pyramid.get(0),
                            self.options,
                        )

                # Trigger training-mode forward and backward JIT traces with CUDAGraphs
                self.model.train()
                with torch.enable_grad():
                    for scale in scales:
                        # Generator train trace
                        noises_tr = self.model.get_pyramid_noise(
                            scale, indexes, wells_pyramid, seismic_pyramid
                        )
                        amps_tr = self.model.noise_amps[:scale+1]
                        fake_tr = self.model.generator(noises_tr, amps_tr, stop_scale=scale)

                        # Discriminator train trace
                        scores_fake = self.model.discriminator.discs[scale](fake_tr)
                        scores_real = self.model.discriminator.discs[scale](facies_pyramid[scale])

                        # Backward trace
                        loss_tr = (scores_fake.mean() + scores_real.mean()) * 0.0
                        loss_tr.backward()

                self.model.generator.zero_grad(set_to_none=True)
                self.model.discriminator.zero_grad(set_to_none=True)
            finally:
                self.model.train()

        if device_manager.is_main_process:
            # Force the compilation bar to finish cleanly if it hasn't reached 100% yet
            unwrap_ddp(self.model).finish_compile_progress()  # type: ignore
            print("  [warmup] JIT compilation traces completed.\n")

    def _ddp_barrier(self) -> None:
        """Synchronize DDP ranks via NCCL barrier."""
        if device_manager.is_distributed and dist.is_initialized():
            dist.barrier()  # type: ignore

    def create_dataloader(self) -> IDataLoader:
        sampler: DistributedSampler[RawBatch] | None = None
        do_shuffle = getattr(self.options, "shuffle", True)
        shuffle = False
        if device_manager.is_distributed:
            world_size = dist.get_world_size()
            self.batch_size = max(1, self.batch_size // world_size)
            sampler = DistributedSampler(
                self.dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=do_shuffle,
            )
        else:
            shuffle = do_shuffle

        has_workers = self.options.num_workers > 0
        return DataLoader[RawBatch](
            self.dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.options.num_workers,
            pin_memory=device_manager.is_cuda,
            persistent_workers=has_workers,
            prefetch_factor=self.options.prefetch_factor if has_workers else None,
            drop_last=False,
            timeout=120 if has_workers else 0,
        )

    def create_model(self) -> FaciesGAN:
        """Instantiate and return the :class:`FaciesGAN` configured
        with the trainer options and device.

        Returns
        -------
        FaciesGAN
            The initialized model instance.
        """
        return FaciesGAN(self.options, self.channels)

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
        indexes : tuple[int, ...]
            Tuple of batch sample indices.
        wells_pyramid : dict[int, torch.Tensor], optional
            Dictionary of well-conditioning tensors per scale.
        seismic_pyramid : dict[int, torch.Tensor], optional
            Dictionary of seismic-conditioning tensors per scale.

        Returns
        -------
        tuple[torch.Tensor, ...]
            A tuple of generated facies tensors for visualization, one per scale.
        """
        with torch.inference_mode():
            return tuple(
                self.model.generate_fake(
                    self.model.get_pyramid_noise(
                        scale, indexes, wells_pyramid, seismic_pyramid
                    ),
                    scale,
                )
                for scale in scales
            )

    def compute_rec_input(
        self, scale: int, indexes: torch.Tensor, facies_pyramid: dict[int, torch.Tensor]
    ) -> torch.Tensor:
        real = facies_pyramid[scale]
        if scale == 0:
            return torch.zeros_like(real).to(device_manager.device)

        prev_facies = facies_pyramid[scale - 1]
        # Only index if prev_facies contains the whole dataset (unlikely in training,
        # but matches robust logic elsewhere). In normal training, it is already
        # the batch aligned with `indexes`.
        if prev_facies.shape[0] != indexes.shape[0]:
            prev_facies = prev_facies[indexes]

        return model_utils.interpolate(
            prev_facies, cast(tuple[int, int], tuple(real.shape[2:]))
        ).to(device_manager.device)

    def _build_z_rec_for_positions(
        self,
        scale: int,
        real: torch.Tensor,
        positions: list[int],
        wells_pyramid: dict[int, torch.Tensor],
        seismic_pyramid: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Build padded z_rec for the given positional indices within the batch."""
        batch_real = real[positions]
        pad_value = float(self.model.padding_value)

        # Scale 0 has no lower-scale spatial prior, so its reconstruction
        # noise must be entirely random — no conditioning channels.
        # This matches the original git HEAD behaviour where scale == 0
        # generated a pure-noise tensor of width self.noise_channels.
        if scale == 0:
            z_rec = model_utils.generate_noise(
                (self.noise_channels, *batch_real.shape[2:]),
                num_samp=len(positions),
            )
            return F.pad(z_rec, [self.residual_padding] * 4, value=pad_value)

        num_cond_channels = 0
        if len(wells_pyramid) > 0:
            num_cond_channels += self.options.num_facies_channels
        if len(seismic_pyramid) > 0:
            num_cond_channels += 1

        noise_ch = self.noise_channels - num_cond_channels
        z_rec = model_utils.generate_noise(
            (noise_ch, *batch_real.shape[2:]),
            num_samp=len(positions),
        )

        to_concat = [z_rec]
        if len(wells_pyramid) > 0:
            to_concat.append(wells_pyramid[scale][positions].to(device_manager.device))
        if len(seismic_pyramid) > 0:
            to_concat.append(
                seismic_pyramid[scale][positions].to(device_manager.device)
            )

        if len(to_concat) > 1:
            z_rec = torch.cat(to_concat, dim=1)

        return F.pad(z_rec, [self.residual_padding] * 4, value=pad_value)

    # noinspection PyDefaultArgument
    def init_rec_noise_and_amp(
        self,
        scale: int,
        indexes: torch.Tensor,
        real: torch.Tensor,
        wells_pyramid: dict[int, torch.Tensor] = {},
        seismic_pyramid: dict[int, torch.Tensor] = {},
    ) -> None:
        """Initialize per-scale reconstruction noise and amplitudes.

        Parameters
        ----------
        scale : int
            Scale index.
        indexes : list[int]
            Batch sample indices.
        real : torch.Tensor
            Real facies data at the current scale.
        wells_pyramid : dict[int, torch.Tensor], optional
            Conditioning well data.
        seismic_pyramid : dict[int, torch.Tensor], optional
            Conditioning seismic data.
        """
        if len(self.model.rec_noise) >= scale + 1:
            if self.model.rec_noise[scale].shape[0] == real.shape[0]:
                return
            # If batch size mismatch (e.g. resume with different DDP config),
            # we must re-initialize this scale's noise.
            z_rec = self._build_z_rec_for_positions(
                scale, real, list(range(real.shape[0])), wells_pyramid, seismic_pyramid
            )
            self.model.rec_noise[scale] = z_rec
            return

        actual_batch = real.shape[0]
        z_rec = self._build_z_rec_for_positions(
            scale, real, list(range(actual_batch)), wells_pyramid, seismic_pyramid
        )
        self.model.rec_noise.append(z_rec)

        with torch.no_grad():
            fake = self.model.generator(
                self.model.get_pyramid_noise(
                    scale, indexes, wells_pyramid, seismic_pyramid, rec=True
                ),
                self.model.noise_amps + [torch.tensor(0.0, device=real.device)],
                stop_scale=scale,
            ).clone()

        use_rp = self.options.use_rock_physics
        res_real = model_utils.split_facies_rp(
            real, self.options.num_facies_channels, has_rp=use_rp
        )
        real_facies = res_real[model_utils.SplitKey.FACIES]

        res_fake = model_utils.split_facies_rp(
            fake, self.options.num_facies_channels, has_rp=use_rp
        )
        fake_facies = res_fake[model_utils.SplitKey.FACIES]

        assert real_facies is not None and fake_facies is not None
        rmse = torch.sqrt(F.mse_loss(fake_facies, real_facies))

        # ── Amplitude: scale 0 uses a dedicated multiplier and no min-clamp ─
        if scale == 0:
            amp = self.scale0_noise_amp * rmse
            if len(self.model.noise_amps) <= scale:
                self.model.noise_amps.append(amp)
            else:
                self.model.noise_amps[scale] = amp
            return
        else:
            amp = torch.max(self.noise_amp * rmse, self.min_noise_amp)

            if scale < len(self.model.noise_amps):
                self.model.noise_amps[scale] = (amp + self.model.noise_amps[scale]) / 2
            else:
                self.model.noise_amps.append(amp)

    def init_dataset(self) -> tuple[PyramidsDataset, tuple[tuple[int, ...], ...]]:
        dataset = PyramidsDataset(self.options, include_index=True)
        if len(self.options.wells_mask_columns) > 0:
            sel = [int(i) for i in self.options.wells_mask_columns]
            dataset.batches = [dataset.batches[i] for i in sel]
            dataset.indices = dataset.indices[torch.as_tensor(sel, dtype=torch.long)]
        elif self.options.num_train_pyramids < len(dataset):
            indexes = torch.randperm(len(dataset))[: self.options.num_train_pyramids]
            dataset.batches = [dataset.batches[i] for i in indexes]
            dataset.indices = dataset.indices[indexes]

        return dataset, dataset.scales

    def load_model(self, scale: int) -> None:
        """Load generator and discriminator state dicts for a specific scale.

        Parameters
        ----------
        scale : int
            Scale index to load models for.
        """
        try:
            generator_path = os.path.join(
                str(self.checkpoint_path), str(scale), CheckpointFilenames.GENERATOR
            )
            discriminator_path = os.path.join(
                str(self.checkpoint_path), str(scale), CheckpointFilenames.DISCRIMINATOR
            )

            gen = unwrap_ddp(self.model.generator.gens[scale])
            gen.load_state_dict(model_utils.load(generator_path))
            disc = unwrap_ddp(self.model.discriminator.discs[scale])
            disc.load_state_dict(model_utils.load(discriminator_path))
        except (FileNotFoundError, RuntimeError, KeyError, ValueError, OSError) as e:
            print(f"Error loading models from {self.checkpoint_path}/{scale}: {e}")
            raise

    def load_optimizers(
        self,
        scale: int,
        scale_path: str,
        generator_optimizer: optim.Optimizer,
        discriminator_optimizer: optim.Optimizer,
        generator_scheduler: optim.lr_scheduler.LRScheduler,
        discriminator_scheduler: optim.lr_scheduler.LRScheduler,
    ) -> None:
        """Load optimizer and scheduler state dictionaries from checkpoint.

        If any checkpoint files are missing or incompatible a warning is
        printed and the trainer continues without restoring those states.

        Parameters
        ----------
        scale : int
            Scale index.
        scale_path : str
            Path to the directory containing checkpoints for this scale.
        generator_optimizer : optim.Optimizer
            The generator optimizer to load state into.
        discriminator_optimizer : optim.Optimizer
            The discriminator optimizer to load state into.
        generator_scheduler : optim.lr_scheduler.LRScheduler
            The generator scheduler to load state into.
        discriminator_scheduler : optim.lr_scheduler.LRScheduler
            The discriminator scheduler to load state into.
        """
        try:
            generator_optimizer.load_state_dict(
                model_utils.load(os.path.join(scale_path, CheckpointFilenames.OPT_G))
            )
            discriminator_optimizer.load_state_dict(
                model_utils.load(os.path.join(scale_path, CheckpointFilenames.OPT_D))
            )
            generator_scheduler.load_state_dict(
                model_utils.load(os.path.join(scale_path, CheckpointFilenames.SCH_G))
            )
            discriminator_scheduler.load_state_dict(
                model_utils.load(os.path.join(scale_path, CheckpointFilenames.SCH_D))
            )
        except (FileNotFoundError, RuntimeError, KeyError, ValueError, OSError) as e:
            print(f"Warning: Could not load optimizers for scale {scale}: {e}")

    def create_batch_iterator(
        self, loader: IDataLoader, scales: tuple[int, ...]
    ) -> Iterator[PyramidsBatch | None]:
        prefetcher = DataPrefetcher(loader, scale_indices=scales)
        self._batch_prefetcher = prefetcher
        return iter(prefetcher)

    def save_generated_outputs(
        self,
        scales: tuple[int, ...],
        epoch: int,
        batch_id: int,
        outputs_path: dict[int, str],
    ) -> None:
        if not self.enable_plot_outputs or self._batch_prefetcher is None:
            return

        sample_batch = self._sample_seen_batch()
        if sample_batch is None:
            return

        facies, wells, masks, seismic = sample_batch

        facies_pyramid = self._to_pyramid(facies)
        wells_pyramid = self._to_pyramid(wells)
        masks_pyramid = self._to_pyramid(masks)
        seismic_pyramid = self._to_pyramid(seismic)

        use_rock_physics = getattr(self.options, "use_rock_physics", False)
        num_facies_ch = self.options.num_facies_channels

        norm_range: tuple[float, float] = (
            float(self.options.normalization_range[0]),
            float(self.options.normalization_range[1]),
        )

        for scale in scales:
            real_facies = facies_pyramid.get(scale)
            if real_facies is None:
                continue

            sample_count = real_facies.shape[0]
            if sample_count == 0:
                continue

            tiled_indexes = torch.arange(
                sample_count, device=device_manager.device
            ).repeat_interleave(self.num_generated_per_real)

            noises = self.model.get_pyramid_noise(
                scale, tiled_indexes, wells_pyramid, seismic_pyramid
            )

            with torch.no_grad():
                generated_facies = self.model.generator(
                    noises, self.model.noise_amps[: scale + 1], stop_scale=scale
                ).clamp(-1, 1)

            facies_tensor = generated_facies.reshape(
                sample_count, self.num_generated_per_real, *generated_facies.shape[1:]
            )
            real_facies_tensor = real_facies

            facies_cpu = device_manager.to_cpu(facies_tensor, non_blocking=True)
            real_cpu = device_manager.to_cpu(real_facies_tensor, non_blocking=True)

            # Synchronize with the non_blocking detach/to('cpu') transfers
            # that were initiated for facies and real.
            device_manager.synchronize()

            masks_cpu: torch.Tensor | None = None
            if scale in masks_pyramid:
                masks_cpu = device_manager.to_cpu(
                    masks_pyramid[scale], non_blocking=True
                )

            res_gen = model_utils.split_facies_rp(
                facies_cpu, num_facies_ch, has_rp=use_rock_physics
            )
            facies_only_cpu, rp_cpu = res_gen["facies"], res_gen["rock_physics"]

            res_real = model_utils.split_facies_rp(
                real_cpu, num_facies_ch, has_rp=use_rock_physics
            )
            real_facies_only_cpu, real_rp_cpu = (
                res_real["facies"],
                res_real["rock_physics"],
            )

            if facies_only_cpu is None or real_facies_only_cpu is None:
                continue

            masks_np = (
                utils.torch2np(masks_cpu, normalization_range=norm_range)
                if masks_cpu is not None
                else None
            )
            out_dir = outputs_path.get(scale)
            if out_dir is None:
                continue

            bw.submit_plot_generated_outputs(
                utils.torch2np(
                    facies_only_cpu, denormalize=True, normalization_range=norm_range
                ),
                utils.torch2np(
                    real_facies_only_cpu,
                    denormalize=True,
                    normalization_range=norm_range,
                ),
                scale,
                epoch,
                out_dir,
                masks_np,
                batch_id=batch_id,
                normalization_range=norm_range,
            )

            if use_rock_physics and rp_cpu is not None and real_rp_cpu is not None:
                lo, hi = min(norm_range), max(norm_range)
                ip_path = out_dir.replace(ExperimentPaths.FACIES, ExperimentPaths.IP)
                is_path = out_dir.replace(ExperimentPaths.FACIES, ExperimentPaths.IS)
                vp_vs_path = out_dir.replace(
                    ExperimentPaths.FACIES, ExperimentPaths.VP_VS
                )

                # Ip  (channel 0)
                os.makedirs(ip_path, exist_ok=True)
                bw.submit_plot_generated_outputs(
                    utils.torch2np(
                        rp_cpu[:, :, 0:1, ...].clamp(lo, hi),
                        denormalize=True,
                        normalization_range=norm_range,
                    ),
                    utils.torch2np(
                        real_rp_cpu[:, 0:1, ...].clamp(lo, hi),
                        denormalize=True,
                        normalization_range=norm_range,
                    ),
                    scale,
                    epoch,
                    ip_path,
                    masks_np,
                    batch_id=batch_id,
                    plot_title="Acoustic Impedance (Ip)",
                    normalization_range=norm_range,
                )

                # Is  (channel 1)
                is_path = out_dir.replace(ExperimentPaths.FACIES, ExperimentPaths.IS)
                os.makedirs(is_path, exist_ok=True)
                bw.submit_plot_generated_outputs(
                    utils.torch2np(
                        rp_cpu[:, :, 1:2, ...].clamp(lo, hi),
                        denormalize=True,
                        normalization_range=norm_range,
                    ),
                    utils.torch2np(
                        real_rp_cpu[:, 1:2, ...].clamp(lo, hi),
                        denormalize=True,
                        normalization_range=norm_range,
                    ),
                    scale,
                    epoch,
                    is_path,
                    masks_np,
                    batch_id=batch_id,
                    plot_title="Shear Impedance (Is)",
                    normalization_range=norm_range,
                )

                # Vp/Vs  (channel 2)
                vp_vs_path = out_dir.replace(
                    ExperimentPaths.FACIES, ExperimentPaths.VP_VS
                )
                os.makedirs(vp_vs_path, exist_ok=True)
                try:
                    # Diagnostic print: numeric ranges for VP/VS (generated vs real)
                    gen_vpvs = rp_cpu[:, :, 2:3, ...]
                    real_vpvs = real_rp_cpu[:, 2:3, ...]
                    gen_vpvs_np = device_manager.to_numpy(gen_vpvs)
                    real_vpvs_np = device_manager.to_numpy(real_vpvs)
                    lo_g = float(np.nanmin(cast(np.ndarray, gen_vpvs_np)))
                    hi_g = float(np.nanmax(cast(np.ndarray, gen_vpvs_np)))
                    mean_g = float(np.nanmean(cast(np.ndarray, gen_vpvs_np)))
                    lo_r = float(np.nanmin(cast(np.ndarray, real_vpvs_np)))
                    hi_r = float(np.nanmax(cast(np.ndarray, real_vpvs_np)))
                    mean_r = float(np.nanmean(cast(np.ndarray, real_vpvs_np)))
                    print(
                        f"[diagnostic] Scale_{scale} VP_VS generated range: [{lo_g:.6f}, {hi_g:.6f}] mean={mean_g:.6f}"
                    )
                    print(
                        f"[diagnostic] Scale_{scale} VP_VS real      range: [{lo_r:.6f}, {hi_r:.6f}] mean={mean_r:.6f}"
                    )
                except Exception as _e:
                    print(f"[diagnostic] Could not compute VP/VS numeric summary: {_e}")
                bw.submit_plot_generated_outputs(
                    utils.torch2np(
                        rp_cpu[:, :, 2:3, ...].clamp(lo, hi),
                        denormalize=True,
                        normalization_range=norm_range,
                    ),
                    utils.torch2np(
                        real_rp_cpu[:, 2:3, ...].clamp(lo, hi),
                        denormalize=True,
                        normalization_range=norm_range,
                    ),
                    scale,
                    epoch,
                    vp_vs_path,
                    masks_np,
                    batch_id=batch_id,
                    plot_title="Vp/Vs Ratio",
                    normalization_range=norm_range,
                )

    def setup_optimizers(self, scales: tuple[int, ...]) -> None:
        """Initialize optimizers and schedulers for the given scales.

        Parameters
        ----------
        scales : tuple[int, ...]
            Tuple of scale indices to set up.
        """
        for scale in scales:
            self.discriminator_optimizers[scale] = FusedAdam(
                self.model.discriminator.discs[scale].parameters(),
                lr=self.lr_d,
                betas=(self.beta1, 0.999),
                set_grad_none=True,
            )
            self.discriminator_schedulers[scale] = torch.optim.lr_scheduler.StepLR(
                self.discriminator_optimizers[scale],
                step_size=self.lr_decay,
                gamma=self.gamma,
            )

            self.generator_optimizers[scale] = FusedAdam(
                self.model.generator.gens[scale].parameters(),
                lr=self.lr_g,
                betas=(self.beta1, 0.999),
                set_grad_none=True,
            )

            self.generator_schedulers[scale] = (
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.generator_optimizers[scale],
                    mode="min",
                    factor=self.options.lr_g_factor,
                    patience=self.options.lr_patience,
                    min_lr=self.options.lr_min,
                    threshold=1e-4,
                )
            )

    def reset_schedulers(self, scales: tuple[int, ...]) -> None:
        """Reset LR schedulers for a new batch.

        Parameters
        ----------
        scales : tuple[int, ...]
            Tuple of scale indices to reset.
        """
        for scale in scales:
            # Reset optimizer param group LRs to initial values before
            # creating new schedulers.
            for pg in self.generator_optimizers[scale].param_groups:
                pg["lr"] = self.lr_g
            for pg in self.discriminator_optimizers[scale].param_groups:
                pg["lr"] = self.lr_d

            self.generator_schedulers[scale] = (
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.generator_optimizers[scale],
                    mode="min",
                    factor=self.options.lr_g_factor,
                    patience=self.options.lr_patience,
                    min_lr=self.options.lr_min,
                    threshold=1e-4,
                )
            )
            self.discriminator_schedulers[scale] = torch.optim.lr_scheduler.StepLR(
                self.discriminator_optimizers[scale],
                step_size=self.lr_decay,
                gamma=self.gamma,
            )
            # Reset the smoother for this scale so it starts fresh
            if scale in self._g_loss_smoother:
                self._g_loss_smoother[scale].reset()

    @staticmethod
    def save_optimizers(
        scale_path: str,
        generator_optimizer: optim.Optimizer,
        discriminator_optimizer: optim.Optimizer,
        generator_scheduler: LRScheduler,
        discriminator_scheduler: LRScheduler,
    ) -> None:
        """Save optimizer and scheduler state dicts to disk.

        Parameters
        ----------
        scale_path : str
            Path to the scale directory.
        generator_optimizer : optim.Optimizer
            Generator optimizer.
        discriminator_optimizer : optim.Optimizer
            Discriminator optimizer.
        generator_scheduler : LRScheduler
            Generator scheduler.
        discriminator_scheduler : LRScheduler
            Discriminator scheduler.
        """
        os.makedirs(scale_path, exist_ok=True)
        torch.save(
            generator_optimizer.state_dict(),
            os.path.join(scale_path, CheckpointFilenames.OPT_G),
        )
        torch.save(
            discriminator_optimizer.state_dict(),
            os.path.join(scale_path, CheckpointFilenames.OPT_D),
        )
        torch.save(
            generator_scheduler.state_dict(),
            os.path.join(scale_path, CheckpointFilenames.SCH_G),
        )
        torch.save(
            discriminator_scheduler.state_dict(),
            os.path.join(scale_path, CheckpointFilenames.SCH_D),
        )

    def save_progress(
        self,
        epoch: int,
        batch_id: int,
        total_batches: int,
        scales_to_train: tuple[int, ...],
        scale_paths: dict[int, str],
    ) -> None:
        """Save training progress models, optimizers, and checkpoint."""
        # Determine the next epoch and batch to resume from
        next_batch_id = batch_id + 1
        next_epoch = epoch
        if next_batch_id >= total_batches:
            next_batch_id = 0
            next_epoch = epoch + 1

        for s in scales_to_train:
            self.model.save_scale(s, scale_paths[s])
            self.save_optimizers(
                scale_paths[s],
                self.generator_optimizers[s],
                self.discriminator_optimizers[s],
                self.generator_schedulers[s],
                self.discriminator_schedulers[s],
            )
            # Record the actual next completed epoch
            meta_path = os.path.join(
                scale_paths[s], CheckpointFilenames.COMPLETED_EPOCH
            )
            with open(meta_path, "w") as f:
                f.write(str(next_epoch))

        # Save the monolithic epoch checkpoint
        self.save_epoch_checkpoint(
            scales_to_train, scale_paths, next_epoch, next_batch_id
        )

    def save_epoch_checkpoint(
        self,
        scales: tuple[int, ...],
        scale_paths: dict[int, str],
        epoch: int,
        batch_id: int,
    ) -> None:
        """Save a full training checkpoint (weights + optimizer states).

        Parameters
        ----------
        scales : tuple[int, ...]
            Active scales.
        scale_paths : dict[int, str]
            Paths to scale directories.
        epoch : int
            Current epoch index.
        batch_id : int
            Current batch index.
        """
        scales_info: dict[int, ScaleCheckpoint] = {}
        for s in scales:
            gen = unwrap_ddp(self.model.generator.gens[s])
            disc = unwrap_ddp(self.model.discriminator.discs[s])
            scales_info[s] = ScaleCheckpoint(
                generator=gen.state_dict(),
                discriminator=disc.state_dict(),
                opt_g=self.generator_optimizers[s].state_dict(),
                opt_d=self.discriminator_optimizers[s].state_dict(),
                sch_g=self.generator_schedulers[s].state_dict(),
                sch_d=self.discriminator_schedulers[s].state_dict(),
            )

        checkpoint = Checkpoint(
            epoch=epoch,
            batch_id=batch_id,
            noise_amps=list(self.model.noise_amps),
            disc_step_counter=self.model.disc_step_counter,
            extra_disc_step_counter=self.model.extra_disc_step_counter,
            scales=scales_info,
            grad_scaler_g=self.model.grad_scaler_g.state_dict(),
            rec_noise=self.model.rec_noise,
            rng_state=device_manager.get_rng_state_dict(),
            seen_indices=self._seen_indices_for_save,
        )

        # Save into the first scale's directory (arbitrary but deterministic)
        ckpt_path = os.path.join(
            scale_paths[min(scales)], CheckpointFilenames.EPOCH_CKPT
        )
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        import sys
        print(f"\n  Saving epoch checkpoint at epoch {epoch} (batch {batch_id})...", end="", flush=True)
        try:
            torch.save(checkpoint.to_dict(), ckpt_path)
            print(" done.")
        except Exception as e:
            print(f" failed: {e}", file=sys.stderr)


    def load_epoch_checkpoint(
        self, scales: tuple[int, ...], scale_paths: dict[int, str]
    ) -> tuple[int, int]:
        """Restore training state from a saved epoch checkpoint."""
        anchor_scale = min(scales)
        checkpoint_path = os.path.join(
            scale_paths[anchor_scale], CheckpointFilenames.EPOCH_CKPT
        )

        if not os.path.isfile(checkpoint_path):
            return 0, 0

        # 1. Load Checkpoint
        ckpt = Checkpoint.load(checkpoint_path)

        # 2. Restore Model Weights + Optimizer/Scheduler States
        for s in scales:
            if s in ckpt.scales:
                sd = ckpt.scales[s]
                self.model.load_state_dict_compat(
                    unwrap_ddp(self.model.generator.gens[s]), sd.generator
                )
                self.model.load_state_dict_compat(
                    unwrap_ddp(self.model.discriminator.discs[s]), sd.discriminator
                )
                if s in self.generator_optimizers:
                    try:
                        self.generator_optimizers[s].load_state_dict(sd.opt_g)
                    except ValueError:
                        from tqdm import tqdm

                        tqdm.write(
                            f"  [warn] scale {s} generator optimizer state skipped "
                            "(checkpoint format mismatch — parameter groups changed)"
                        )
                if s in self.discriminator_optimizers:
                    self.discriminator_optimizers[s].load_state_dict(sd.opt_d)
                if s in self.generator_schedulers:
                    self.generator_schedulers[s].load_state_dict(sd.sch_g)
                if s in self.discriminator_schedulers:
                    self.discriminator_schedulers[s].load_state_dict(sd.sch_d)

        # 2.5 Override LR if requested by launch options (handles Resume with new LR)
        for s in scales:
            if s in self.discriminator_optimizers:
                new_lr_d = self.lr_d * 0.01 if s == 0 else self.lr_d
                for param_group in self.discriminator_optimizers[s].param_groups:
                    param_group["lr"] = new_lr_d

                # Update scheduler base_lrs so it doesn't revert on the next step
                if s in self.discriminator_schedulers:
                    self.discriminator_schedulers[s].base_lrs = [new_lr_d] * len(
                        self.discriminator_optimizers[s].param_groups
                    )

                from tqdm import tqdm

                tqdm.write(
                    f"  [info] scale {s} discriminator LR overridden to {new_lr_d}"
                )

            if s in self.generator_optimizers:
                for param_group in self.generator_optimizers[s].param_groups:
                    param_group["lr"] = self.lr_g
                if s in self.generator_schedulers:
                    self.generator_schedulers[s].base_lrs = [self.lr_g] * len(
                        self.generator_optimizers[s].param_groups
                    )

        # 3. Restore Auxiliary Metadata
        if ckpt.noise_amps:
            self.model.noise_amps = [
                a.to(device_manager.device) for a in ckpt.noise_amps
            ]
        self.model.disc_step_counter = ckpt.disc_step_counter
        self.model.extra_disc_step_counter = ckpt.extra_disc_step_counter

        if ckpt.rec_noise:
            self.model.rec_noise = [n.to(device_manager.device) for n in ckpt.rec_noise]

        if ckpt.grad_scaler_g is not None and self.model.use_grad_scaler:
            try:
                self.model.grad_scaler_g.load_state_dict(ckpt.grad_scaler_g)
            except Exception:
                pass

        # 4. Restore RNG States
        device_manager.set_rng_state_dict(ckpt.rng_state)

        # 5. Restore Seen Indices
        if ckpt.seen_indices:
            self._seen_indices_for_save = ckpt.seen_indices

        return ckpt.epoch, ckpt.batch_id

    def _sample_seen_batch(self) -> Batch | None:
        if not self._seen_indices_for_save:
            return None

        sample_count = min(self.num_real_facies, len(self._seen_indices_for_save))
        if sample_count <= 0:
            return None

        # Single CPU transfer instead of N .item() GPU→CPU syncs.
        # Build a plain Python list of ints on CPU to avoid GPU↔CPU syncs
        randperm_cpu = torch.randperm(
            len(self._seen_indices_for_save), device=DeviceType.CPU
        )[:sample_count]
        positions_cpu: list[int] = cast(list[int], randperm_cpu.tolist())  # type: ignore[assignment]
        sampled_items = [
            cast(
                tuple[int, Batch],
                self.dataset[self._seen_indices_for_save[pos][0]],
            )
            for pos in positions_cpu
        ]

        # dataset[idx] returns (index, Batch) when include_index=True
        _, first_batch = sampled_items[0]
        first_facies, first_wells, first_masks, first_seismic = first_batch

        facies = tuple(
            torch.stack([item[1].facies[scale] for item in sampled_items], dim=0)
            for scale in range(len(first_facies))
        )
        wells = tuple(
            torch.stack([item[1].wells[scale] for item in sampled_items], dim=0)
            for scale in range(len(first_wells))
        )
        masks = tuple(
            torch.stack([item[1].masks[scale] for item in sampled_items], dim=0)
            for scale in range(len(first_masks))
        )
        seismic = tuple(
            torch.stack([item[1].seismic[scale] for item in sampled_items], dim=0)
            for scale in range(len(first_seismic))
        )

        return Batch(facies=facies, wells=wells, masks=masks, seismic=seismic)

    def _to_pyramid(
        self, component: tuple[torch.Tensor, ...]
    ) -> dict[int, torch.Tensor]:
        if not component:
            return {}

        return {
            idx: device_manager.to_device(
                component[idx],
                channels_last=True,
                non_blocking=True,
            )
            for idx in range(len(component))
        }

    def optimization_step(
        self,
        indexes: torch.Tensor,
        facies_pyramid: dict[int, torch.Tensor],
        rec_in_pyramid: dict[int, torch.Tensor],
        wells_pyramid: dict[int, torch.Tensor] | None = None,
        masks_pyramid: dict[int, torch.Tensor] | None = None,
        seismic_pyramid: dict[int, torch.Tensor] | None = None,
    ) -> ScaleMetrics | tuple[IterableMetrics, ...]:
        """Perform a single optimization step for the model.

        Parameters
        ----------
        indexes : list[int]
            Batch sample indices.
        facies_pyramid : dict[int, torch.Tensor]
            Dictionary of real facies data for all scales.
        rec_in_pyramid : dict[int, torch.Tensor]
            Dictionary mapping scale -> reconstruction input from previous scale.
        wells_pyramid : dict[int, torch.Tensor]
            Dictionary of well-conditioning tensors for all scales.
        masks_pyramid : dict[int, torch.Tensor]
            Dictionary of masks tensors for all scales.
        seismic_pyramid : dict[int, torch.Tensor]
            Dictionary of seismic-conditioning tensors for all scales.

        Returns
        -------
        ScaleMetrics[torch.Tensor]
            Collected metrics for all scales after the optimization step.
        """
        return cast(
            ScaleMetrics,
            self.model(
                self.generator_optimizers,
                self.discriminator_optimizers,
                indexes,
                facies_pyramid,
                rec_in_pyramid,
                wells_pyramid,
                masks_pyramid,
                seismic_pyramid,
            ),
        )

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
                device=device_manager.device,
            )

        if facies_pyramid is None:
            raise ValueError("facies_pyramid must be provided or part of batch")
        if wells_pyramid is None:
            wells_pyramid = {}
        if masks_pyramid is None:
            masks_pyramid = {}
        if seismic_pyramid is None:
            seismic_pyramid = {}

        # ── Input Logging (Main Process, First Pass Only) ──────────────
        if device_manager.is_main_process and start_epoch == 0 and batch_id == 0:
            self._log_input_stats(
                start_epoch,
                facies_pyramid,
                wells_pyramid,
                masks_pyramid,
                seismic_pyramid,
            )

        # Derive indexes from the actual batch size
        # indexes is now a torch.Tensor provided by the batch unpacking above

        rec_in_pyramid: dict[int, torch.Tensor] = {}
        # 1. Ensure rec_noise/amp are initialized for all scales up to max(scales).
        # The generator needs noise tensors for ALL scales (0..scale) to perform
        # a progressive forward pass.
        max_scale = max(scales)
        for s in range(max_scale + 1):
            if (
                len(self.model.rec_noise) <= s
                or self.model.rec_noise[s].shape[0] != facies_pyramid[s].shape[0]
            ):
                self.init_rec_noise_and_amp(
                    s, indexes, facies_pyramid[s], wells_pyramid, seismic_pyramid
                )

        # 2. Compute reconstruction inputs for the active scales
        for scale in scales:
            rec_in_pyramid[scale] = self.compute_rec_input(
                scale, indexes, facies_pyramid
            )

        epoch = start_epoch

        # Let the model know the current epoch
        self.model.current_epoch = epoch  # type: ignore
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
        if device_manager.is_main_process and _is_viz_epoch:
            generated_samples = self.generate_visualization_samples(
                scales, indexes, wells_pyramid, seismic_pyramid
            )

        self.handle_epoch_end(
            scales=scales,
            epoch=epoch,
            scale_metrics=cast(ScaleMetrics, scale_metrics),
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

        return cast(ScaleMetrics, scale_metrics)

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
            if device_manager.is_main_process:
                print(f"\n{'=' * 60}")
                print(f"Training scales {scales_to_train} in parallel")
                print(f"{'=' * 60}\n")

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
                s: os.path.join(scale_paths[s], ExperimentPaths.FACIES)
                for s in scales_to_train
            }

            # Only main process creates directories, writers
            writers: dict[int, SummaryWriter] = {}  # type: ignore
            if device_manager.is_main_process:
                for s in scales_to_train:
                    utils.create_dirs(scale_paths[s])

                    if self.enable_plot_outputs:
                        utils.create_dirs(outputs_paths[s])
                        # Pre-create rock-physics sub-dirs once so save_generated_outputs
                        # never needs os.makedirs on the hot visualization path.
                        if self.options.use_rock_physics:
                            _out = outputs_paths[s]
                            utils.create_dirs(
                                _out.replace(ExperimentPaths.FACIES, ExperimentPaths.IP)
                            )
                            utils.create_dirs(
                                _out.replace(ExperimentPaths.FACIES, ExperimentPaths.IS)
                            )
                            utils.create_dirs(
                                _out.replace(
                                    ExperimentPaths.FACIES, ExperimentPaths.VP_VS
                                )
                            )

            if self.fine_tuning:
                for s in scales_to_train:
                    self.load_model(s)

            # ── Epoch-level resume ──────────────────────────────────
            # When start_epoch > 0 for the *first* scale group, load the
            # epoch checkpoint (model weights + optimizer/scheduler states)
            # and figure out which batch to resume from.
            resume_epoch: int = 0
            resume_batch_id: int = 0
            # base_epoch: the epoch that unprocessed batches should start
            # from.  When resuming mid-batch, batches after the resumed
            # one already completed up to the previous full run's epoch
            # count (recorded in completed_epoch.txt).  Without this,
            # they would incorrectly restart from epoch 0.
            base_epoch: int = 0

            resume_epoch, resume_batch_id = self.load_epoch_checkpoint(
                scales_to_train, scale_paths
            )

            if resume_epoch > 0:
                base_epoch = resume_epoch
                if device_manager.is_main_process:
                    resume_step = resume_epoch * len(self.data_loader) + resume_batch_id
                    print(
                        f"Epoch checkpoint loaded: resuming from batch {resume_batch_id}, "
                        f"epoch {resume_epoch}, step {resume_step}"
                    )
            if resume_batch_id > 0:
                # Optional: Verify or override base_epoch from completed_epoch.txt

                meta_path = os.path.join(
                    scale_paths[min(scales_to_train)],
                    CheckpointFilenames.COMPLETED_EPOCH,
                )
                if os.path.isfile(meta_path):
                    with open(meta_path) as f:
                        # The file contains the last FULLY completed epoch.
                        # We want to start the next iteration at base_epoch + 1
                        # if the checkpoint and file are consistent.
                        last_full_epoch = int(f.read().strip())
                        base_epoch = max(base_epoch, last_full_epoch + 1)

            # Progress bar for all batches and epochs in this group
            total_batches = len(self.data_loader)

            if device_manager.is_main_process and self.enable_tensorboard:
                if resume_epoch > 0:
                    _purge: int | None = resume_epoch * total_batches + resume_batch_id
                else:
                    _purge = None
                writers = {
                    s: SummaryWriter(log_dir=scale_paths[s], purge_step=_purge)
                    for s in scales_to_train
                }
                if self.visualizer is not None:
                    # Recreate visualizer writer with identical purge point to
                    # avoid duplicated/stale events in the global TB stream.
                    try:
                        self.visualizer.close()
                    except Exception:
                        pass
                    self.visualizer.writer = SummaryWriter(
                        log_dir=os.path.join(self.output_path, "tensorboard_logs"),
                        purge_step=_purge,
                    )
            progress_total = self._num_passes * total_batches
            resume_progress = base_epoch * total_batches + resume_batch_id
            resume_progress = max(0, min(resume_progress, progress_total))
            # Write directly to /dev/tty so the bar bypasses torchrun's
            # subprocess pipes and debugpy's output capture entirely.
            # Falls back to sys.stderr when /dev/tty is not available.
            if device_manager.is_main_process:
                try:
                    _tqdm_file: "IO[str]" = open("/dev/tty", "w")
                except OSError:
                    _tqdm_file = sys.stderr
            else:
                _tqdm_file = sys.stderr
            progress: tqdm[Any] | DummyProgress = DummyProgress()

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

                if epoch == base_epoch:
                    try:
                        # Peek at the first batch
                        first_batch = next(batch_iterator)
                        if first_batch is not None:
                            self._warmup_compile_traces(scales_to_train, first_batch)
                        # Chain first_batch back to the iterator without recreating it or spawning new workers
                        import itertools

                        batch_iterator = itertools.chain([first_batch], batch_iterator)
                    except StopIteration:
                        pass

                # Initialize progress bar only AFTER warmup completes to avoid
                # it overlapping with the [compile] progress bar output.
                if device_manager.is_main_process and isinstance(
                    progress, DummyProgress
                ):
                    print(
                        f"  [data] Initializing batch prefetchers for scales {scales_to_train}..."
                    )
                    progress = tqdm(
                        total=progress_total,
                        initial=resume_progress,
                        position=0,
                        disable=not device_manager.is_main_process,
                        dynamic_ncols=True,
                        file=_tqdm_file,
                    )

                for batch_id, batch in enumerate(batch_iterator):

                    # Skip batches already processed before the resume point
                    if epoch == base_epoch and batch_id < resume_batch_id:
                        continue

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
                        progress=progress,  # type: ignore
                        facies_pyramid=None,
                        wells_pyramid=None,
                        masks_pyramid=None,
                        seismic_pyramid=None,
                        start_epoch=epoch,
                        batch=batch,  # Pass the batch directly
                    )
                    if device_manager.is_main_process:
                        progress.set_description(
                            f"Epoch {epoch + 1}/{self._num_passes}, Batch {batch_id + 1}/{total_batches}"
                        )
                        progress.update(1)

                    # Save checkpoint mid-epoch if step-based and the interval matches!
                    if self.time_unit != TimeUnit.EPOCH:
                        is_final_step = (
                            epoch == self._num_passes - 1
                            and batch_id == total_batches - 1
                        )
                        if self._is_time_to_act(
                            epoch,
                            batch_id,
                            self.options.checkpoint_interval,
                            is_final_step,
                        ):
                            # Synchronize all processes under DDP before main rank performs file IO
                            self._ddp_barrier()
                            if device_manager.is_main_process:
                                self.save_progress(
                                    epoch,
                                    batch_id,
                                    total_batches,
                                    scales_to_train,
                                    scale_paths,
                                )
                            self._ddp_barrier()

                # End of batch loop

                # Gather all seen indices for this epoch across all ranks
                epoch_indices = self.collect_seen_indices()
                if epoch_indices:
                    # Update the global set of seen indices (accumulated across all epochs)
                    current_seen: set[tuple[int, ...]] = set(
                        self._seen_indices_for_save
                    )
                    current_seen.update(epoch_indices)
                    self._seen_indices_for_save = sorted(list(current_seen))

                # End of batch loop

                if device_manager.is_main_process:
                    # ── Save Epoch Progress (Periodic or Final) ───────
                    interval = self.options.checkpoint_interval
                    is_final_epoch = epoch == self._num_passes - 1
                    should_save = self._is_time_to_act(
                        epoch, total_batches - 1, interval, is_final_epoch
                    )

                    if should_save:
                        self.save_progress(
                            epoch,
                            total_batches - 1,
                            total_batches,
                            scales_to_train,
                            scale_paths,
                        )
            # After processing all batches for this group, save models (rank 0 only)
            if device_manager.is_main_process:
                for s in scales_to_train:
                    self.model.save_scale(s, scale_paths[s])
                    self.save_optimizers(
                        scale_paths[s],
                        self.generator_optimizers[s],
                        self.discriminator_optimizers[s],
                        self.generator_schedulers[s],
                        self.discriminator_schedulers[s],
                    )

            # Flush any pending background plot jobs
            if device_manager.is_main_process and self.enable_plot_outputs:
                try:
                    from background_workers import BackgroundWorker

                    BackgroundWorker().wait_pending()
                except Exception:
                    pass

            # Synchronise so non-zero ranks wait for rank 0 to finish
            # saving before proceeding to the next scale group (or to
            # DDP teardown at the end of training).
            self._ddp_barrier()

            # Close progress bar and the /dev/tty handle (if opened)
            if device_manager.is_main_process:
                progress.close()
            if _tqdm_file is not sys.stderr:
                try:
                    _tqdm_file.close()
                except OSError:
                    pass

            # Close writers (rank 0 only)
            for writer in writers.values():
                writer.close()

            # Release GPU memory cached by the allocator between groups
            device_manager.release_accelerator_memory()

            group_end_time = time.time()
            elapsed = log.format_time(int(group_end_time - group_start_time))
            if device_manager.is_main_process:
                print(f"\nScales {scales_to_train} training time: {elapsed}")

            scale += num_scales_in_group

        end_train_time = time.time()
        if device_manager.is_main_process:
            print(
                "\nTotal training time:",
                log.format_time(int(end_train_time - start_train_time)),
            )

        # Close TensorBoard writer
        if self.enable_tensorboard and self.visualizer:
            self.visualizer.close()
        if device_manager.is_main_process:
            print("\n✅ Training complete!")
        if self.enable_tensorboard:
            print("\n📊 View outputs in TensorBoard (if still running)")

        # Final DDP barrier: ensure all ranks have finished training
        # before returning so the caller can safely tear down the
        # process group without one rank still doing I/O.
        self._ddp_barrier()

    def handle_epoch_end(
        self,
        scales: tuple[int, ...],
        epoch: int,
        scale_metrics: ScaleMetrics,
        generated_samples: tuple[torch.Tensor, ...],
        writers: dict[int, SummaryWriter],  # type: ignore
        outputs_paths: dict[int, str],
        progress: "tqdm[Any]",  # type: ignore
        real_seismic: dict[int, torch.Tensor] = {},
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
        real_seismic : dict[int, torch.Tensor], optional
            Optional mapping from scale index to real seismic tensors used for
            visualization and logging. When provided the visualizer may use
            these tensors to compare generated outputs against the real
            seismic data. Defaults to an empty dict.
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
        _should_gather = self._is_time_to_act(
            epoch, self._current_batch_id, self.save_interval, _is_last_step
        )
        if _should_gather:
            self._seen_indices_for_save = self.collect_seen_indices()

        # ── Rank-0 only I/O ─────────────────────────────────────────
        if device_manager.is_main_process:
            _print_table = global_step % 100 == 0
            _log_tb = global_step % self._tb_log_interval == 0 or _is_last_step

            # Pre-compute metric floats once (single GPU→CPU sync per scale)
            # so _print_metrics_table and log_epoch both reuse the same values.
            _cached_floats: dict[int, list[float]] = {}
            if _print_table or (self.enable_tensorboard and _log_tb):
                _cached_floats = {
                    s: self._get_metric_floats(
                        scale_metrics.generator[s], scale_metrics.discriminator[s]
                    )
                    for s in scales
                    if s in scale_metrics.generator and s in scale_metrics.discriminator
                }

            # Visualizer update (pass pre-flattened dict to avoid extra .item() syncs)
            if self.enable_tensorboard and self.visualizer and _log_tb:
                from enums import MetricKey as _LK

                # Layout from _get_metric_floats: [0..8] G, [9..12] D (13 total)
                _flat: dict[int, dict[str, float]] = (
                    {
                        s: {
                            _LK.G_TOTAL: v[0],
                            _LK.G_FAKE: v[1],
                            _LK.G_REC_FACIES: v[2],
                            _LK.G_WELL: v[3],
                            _LK.G_DIV: v[4],
                            _LK.G_REC_ROCK_PHYSICS: v[5],
                            _LK.G_TV: v[6],
                            _LK.G_ELASTIC: v[7],
                            _LK.G_SEISMIC: v[8],
                            _LK.D_TOTAL: v[9],
                            _LK.D_REAL: v[10],
                            _LK.D_FAKE: v[11],
                            _LK.D_GP: v[12],
                        }
                        for s, v in _cached_floats.items()
                    }
                    if _cached_floats
                    else {}
                )
                self.visualizer.update(  # type: ignore
                    global_step,
                    _flat if _flat else scale_metrics,
                    generated_samples,
                    samples_processed,
                    scales=scales,
                    real_seismic=real_seismic,
                    force_update=_is_last_step,
                )

            # Print formatted metrics table every 100 global steps.
            if _print_table:
                self._print_metrics_table(scales, scale_metrics, _cached_floats)

            # Save to TensorBoard and log per-scale (throttled to _tb_log_interval).
            if self.enable_tensorboard and _log_tb:
                for scale in scales:
                    g = scale_metrics.generator[scale]
                    d = scale_metrics.discriminator[scale]
                    self.log_epoch(writers[scale], epoch, g, d, global_step, _cached_floats.get(scale))  # type: ignore
                    # Log learning rates per scale.
                    lr_g = self.generator_optimizers[scale].param_groups[0]["lr"]  # type: ignore[index]
                    lr_d = self.discriminator_schedulers[scale].get_last_lr()[0]  # type: ignore[union-attr]
                    writers[scale].add_scalar("LearningRate/generator", lr_g, global_step)  # type: ignore
                    writers[scale].add_scalar("LearningRate/discriminator", lr_d, global_step)  # type: ignore

            # Log when learning rate decays (at every lr_decay interval,
            # before schedulers_step advances the count).
            _decay_counter = (
                global_step
                if self.time_unit != TimeUnit.EPOCH
                else epoch
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
                    if self.time_unit != TimeUnit.EPOCH
                    else f"epoch {epoch}"
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
        # Step schedulers at configured frequency
        should_step_schedulers = False
        if self.time_unit == TimeUnit.EPOCH:
            if self._current_batch_id == self._total_batches - 1:
                should_step_schedulers = True
        else:
            should_step_schedulers = True

        if should_step_schedulers:
            self.schedulers_step(scales, scale_metrics)

    def schedulers_step(
        self, scales: tuple[int, ...], scale_metrics: ScaleMetrics | None = None
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
        from training.metrics import MetricSmoother

        if not hasattr(self, "_g_loss_smoother"):
            self._g_loss_smoother: dict[int, MetricSmoother] = {}

        alpha = getattr(self.options, "lr_smoothing_alpha", 0.9)

        for scale in scales:
            # Generator: ReduceLROnPlateau with smoothed loss
            if scale_metrics is not None and scale in scale_metrics.generator:
                g = scale_metrics.generator[scale]
                raw_loss = g.total.item()  # type: ignore[union-attr]
                if scale not in self._g_loss_smoother:
                    self._g_loss_smoother[scale] = MetricSmoother(alpha=alpha)
                smoothed = self._g_loss_smoother[scale].update(raw_loss)
                self.generator_schedulers[scale].step(smoothed)  # type: ignore[arg-type]
            else:
                # Fallback for subclasses that don't pass metrics
                self.generator_schedulers[scale].step()  # type: ignore[arg-type]
            # Discriminator: plain StepLR
            self.discriminator_schedulers[scale].step()

    def log_epoch(
        self,
        writer: SummaryWriter,  # type: ignore
        epoch: int,
        generator_metrics: GeneratorMetrics,
        discriminator_metrics: DiscriminatorMetrics,
        global_step: int | None = None,
        pre_floats: list[float] | None = None,
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
        global_step : int | None
            Global optimization step (used as TensorBoard x-axis).
        pre_floats : list[float] | None
            Pre-computed float values from ``_get_metric_floats`` for this scale.
            When provided the GPU→CPU sync is skipped entirely.

        Notes
        -----
        Metric dataclass fields are tensor scalars; this function batch-converts
        them to Python floats via ``torch.stack().tolist()`` (single GPU sync)
        before writing to TensorBoard or formatting for display.
        """
        g = generator_metrics
        d = discriminator_metrics

        vals = self._get_metric_floats(g, d, _cached=pre_floats)
        (
            g_total,
            g_fake,
            g_rec_facies,
            g_well,
            g_div,
            g_rec_rock_physics,
            g_tv,
            g_elastic,
            g_seismic,
        ) = vals[:9]
        d_total, d_real, d_fake, d_gp = vals[9:13]

        global_step = global_step if global_step is not None else epoch

        writer.add_scalar("G/Total", g_total, global_step)  # type: ignore
        writer.add_scalar("G/Adv", g_fake, global_step)  # type: ignore
        writer.add_scalar("G/Rec_Facies", g_rec_facies, global_step)  # type: ignore
        writer.add_scalar("G/Well", g_well, global_step)  # type: ignore
        writer.add_scalar("G/Diversity", g_div, global_step)  # type: ignore
        writer.add_scalar("G/Rec_Rock_Physics", g_rec_rock_physics, global_step)  # type: ignore
        writer.add_scalar("G/TV_Smoothness", g_tv, global_step)  # type: ignore
        writer.add_scalar("G/Elastic", g_elastic, global_step)  # type: ignore
        writer.add_scalar("G/Seismic", g_seismic, global_step)  # type: ignore
        writer.add_scalar("D/Total", d_total, global_step)  # type: ignore
        writer.add_scalar("D/Real", d_real, global_step)  # type: ignore
        writer.add_scalar("D/Fake", d_fake, global_step)  # type: ignore
        writer.add_scalar("D/GP", d_gp, global_step)  # type: ignore

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
        writer.add_scalar("Loss/train/generator/rec_facies", g_rec_facies, step)  # type: ignore
        writer.add_scalar("Loss/train/generator/well_constraint", g_well, step)  # type: ignore
        writer.add_scalar("Loss/train/generator/diversity", g_div, step)  # type: ignore
        writer.add_scalar("Loss/train/generator/rec_rock_physics", g_rec_rock_physics, step)  # type: ignore
        writer.add_scalar("Loss/train/generator/tv_smoothness", g_tv, step)  # type: ignore
        writer.add_scalar("Loss/train/generator/elastic", g_elastic, step)  # type: ignore
        writer.add_scalar("Loss/train/generator/seismic", g_seismic, step)  # type: ignore
        writer.add_scalar("Loss/train/generator", g_total, step)  # type: ignore

    def _print_metrics_table(
        self,
        scales: tuple[int, ...],
        scale_metrics: ScaleMetrics,
        _cached_floats: dict[int, list[float]] | None = None,
    ) -> None:
        """Print formatted ASCII tables for Generator and Discriminator metrics."""

        from training.metrics import MetricSmoother

        # Lazily initialise one smoother per scale per metric.
        _sm_alpha = getattr(self.options, "lr_smoothing_alpha", 0.9)
        if not hasattr(self, "_table_smoother"):
            self._table_smoother: dict[int, list[MetricSmoother]] = {}

        for scale in scales:
            if scale not in self._table_smoother:
                # 9 G-metrics + 4 D-metrics = 13 total
                self._table_smoother[scale] = [
                    MetricSmoother(alpha=_sm_alpha) for _ in range(13)
                ]

        lines: list[str] = [
            "",
            "",
            "  Generator Metrics:",
            "  ┌" + "─" * 124 + "┐",
            (
                f"  │ {'Scale':^5} │ {'G_total':^10} │ {'G_adv':^10} │ {'G_fa_rec':^10} │ "
                f"{'G_well':^10} │ {'G_div':^10} │ {'G_rp_rec':^10} │ {'G_tv':^10} │ {'G_el':^10} │ {'G_seis':^10} │"
            ),
            "  ├" + "─" * 124 + "┤",
        ]
        # --- Generator Table ---

        cached_v: list[list[float]] = []

        for scale in scales:
            g = scale_metrics.generator[scale]
            d = scale_metrics.discriminator[scale]
            raw = self._get_metric_floats(
                g, d, _cached=(_cached_floats or {}).get(scale)
            )

            sm = self._table_smoother[scale]
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
        lines = [
            "Generated data shapes:",
            "╔══════════╦══════════╦══════════╦══════════╦══════════╗",
            "║ {:^8} ║ {:^8} ║ {:^8} ║ {:^8} ║ {:^8} ║".format(
                "Batch", "In Ch", "Out Ch", "Height", "Width"
            ),
            "╠══════════╬══════════╬══════════╬══════════╬══════════╣",
        ]
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

    @staticmethod
    def _get_metric_floats(
        g: GeneratorMetrics, d: DiscriminatorMetrics, _cached: list[float] | None = None
    ) -> list[float]:
        """Convert metric tensors to a flat list of Python floats (single sync).

        Layout:  [0..8] raw G metrics, [9..12] D metrics.  13 total.

        If *_cached* is provided it is returned directly, avoiding a
        redundant GPU→CPU transfer when the caller has already converted
        the values this step.
        """
        if _cached is not None:
            return _cached
        import torch as _t

        # Ensure all tensors are on the same device (CPU) before stacking.
        # This prevents RuntimeError when some metrics are DomainConfig.ZERO_SCALAR (CPU)
        # while others are loss tensors (GPU).
        gd: list[float] = _t.stack(  # type: ignore[call-overload]
            device_manager.to_cpu([*g.as_tuple(), *d.as_tuple()])
        ).tolist()
        return gd

    # noinspection PyAttributeOutsideInit

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
            path, load_shapes=False, until_scale=until_scale, load_discriminator=True
        )

    def collect_seen_indices(self) -> list[tuple[int, ...]]:
        if self._batch_prefetcher is None:
            return []
        return gather_seen_indices(self._batch_prefetcher)

    def _log_input_stats(
        self,
        epoch: int,
        facies_pyramid: dict[int, torch.Tensor],
        wells_pyramid: dict[int, torch.Tensor],
        masks_pyramid: dict[int, torch.Tensor],
        seismic_pyramid: dict[int, torch.Tensor],
    ) -> None:
        """Log input data statistics and facies images for debugging.

        This runs only once at the start of training (Epoch 0, Batch 0) on the
        main process to provide a baseline for data verification in TensorBoard.
        """
        from debug_logger import DebugLogger as _dl

        _nf = self.options.num_facies_channels
        for _s, _t in facies_pyramid.items():
            _dl.get().log_facies_input(
                f"input/facies/s{_s}", _t, epoch, num_facies_channels=_nf
            )
        for _s, _t in wells_pyramid.items():
            _dl.get().log_tensor_channel_stats(
                f"input/wells/s{_s}", "wells", _t, epoch, expected_range=(0.0, 1.0)
            )
        for _s, _t in masks_pyramid.items():
            _dl.get().log_tensor_channel_stats(
                f"input/masks/s{_s}", "masks", _t, epoch, expected_range=(0.0, 1.0)
            )
        for _s, _t in seismic_pyramid.items():
            norm_range: tuple[float, float] = (
                float(self.options.normalization_range[0]),
                float(self.options.normalization_range[1]),
            )
            _dl.get().log_tensor_stats(
                f"input/seismic/s{_s}", "seismic", _t, epoch, expected_range=norm_range
            )
