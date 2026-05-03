"""Parallel trainer for multi-scale FaciesGAN training."""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator, Mapping
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm  # type: ignore[import]

import background_workers as bw
import utils
from apex_utils import FusedAdam
from config import (
    D_FILE,
    G_FILE,
    OPT_D_FILE,
    OPT_G_FILE,
    OUTPUT_FACIES_PATH,
    OUTPUT_IP_PATH,
    OUTPUT_IS_PATH,
    OUTPUT_VP_VS_PATH,
    SCH_D_FILE,
    SCH_G_FILE,
)
from datasets import PyramidsBatch, TorchPyramidsDataset
from datasets.data_prefetcher import TorchDataPrefetcher, gather_seen_indices
from datasets.dataset import TorchPyramidsDataset
from datasets.pyramids_batch import Batch
from metrics import MetricSmoother
from models import utils as torch_utils
from models.facies_gan import TorchFaciesGAN, unwrap_ddp
from options import TrainingOptions
from training.base import Trainer
from utils import torch2np


class TorchTrainer(Trainer):
    """Parallel trainer for multi-scale progressive FaciesGAN training."""

    model: TorchFaciesGAN

    def __init__(
        self,
        options: TrainingOptions,
        fine_tuning: bool = False,
        checkpoint_path: str = ".checkpoints",
        device: torch.device = torch.device("cpu"),
        distributed: bool = False,
    ) -> None:
        self.device: torch.device = device
        self.distributed: bool = distributed
        if distributed:
            self._is_main_process = dist.get_rank() == 0
        else:
            self._is_main_process = True

        super().__init__(options, fine_tuning, checkpoint_path)

        self._ckpt_thread: threading.Thread | None = None
        self._g_loss_smoothers: dict[int, MetricSmoother] = {}
        self._compile_warmed_up_scales: set[tuple[int, ...]] = set()
        self._batch_prefetcher: TorchDataPrefetcher | None = None

        self._total_batches = len(self.data_loader)
        self._current_batch_id = 0

    def create_dataloader(self) -> DataLoader[tuple[int, Batch] | Batch]:
        sampler: DistributedSampler[Batch] | None = None
        do_shuffle = getattr(self.options, "shuffle", True)
        shuffle = False
        if self.distributed:
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
        return DataLoader[tuple[int, Batch] | Batch](
            self.dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.options.num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=has_workers,
            prefetch_factor=2 if has_workers else None,
            drop_last=True,
            timeout=120 if has_workers else 0,
        )

    def create_model(self) -> TorchFaciesGAN:
        return TorchFaciesGAN(
            self.options,
            self.device,
            noise_channels=self.noise_channels,
            use_ddp=self.distributed,
        )

    def init_dataset(self) -> tuple[TorchPyramidsDataset, tuple[tuple[int, ...], ...]]:
        dataset = TorchPyramidsDataset(self.options, include_index=True)
        if len(self.options.wells_mask_columns) > 0:
            sel = [int(i) for i in self.options.wells_mask_columns]
            dataset.batches = [dataset.batches[i] for i in sel]
        elif self.options.num_train_pyramids < len(dataset):
            idxs = torch.randperm(len(dataset))[: self.options.num_train_pyramids]
            dataset.batches = [dataset.batches[i] for i in idxs]

        return dataset, dataset.scales

    def generate_visualization_samples(
        self,
        scales: tuple[int, ...],
        indexes: torch.Tensor,
        wells_pyramid: dict[int, torch.Tensor] | None = None,
        seismic_pyramid: dict[int, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        wells_pyramid = wells_pyramid or {}
        seismic_pyramid = seismic_pyramid or {}
        with torch.no_grad():
            return tuple(
                self.model.generate_fake(
                    self.model.get_pyramid_noise(
                        scale,
                        indexes,
                        wells_pyramid,
                        seismic_pyramid,
                    ),
                    scale,
                )
                for scale in scales
            )

    def compute_rec_input(
        self,
        scale: int,
        indexes: torch.Tensor,
        facies_pyramid: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        real = facies_pyramid[scale]
        if scale == 0:
            return torch.zeros_like(real).to(self.device)

        prev_facies = facies_pyramid[scale - 1]
        # Only index if prev_facies contains the whole dataset (unlikely in training,
        # but matches robust logic elsewhere). In normal training, it is already
        # the batch aligned with `indexes`.
        if prev_facies.shape[0] != indexes.shape[0]:
            prev_facies = prev_facies[indexes]

        return torch_utils.interpolate(
            prev_facies,
            cast(tuple[int, int], tuple(real.shape[2:])),
        ).to(self.device)

    def init_rec_noise_and_amp(
        self,
        scale: int,
        indexes: torch.Tensor,
        real: torch.Tensor,
        wells_pyramid: dict[int, torch.Tensor] = {},
        seismic_pyramid: dict[int, torch.Tensor] = {},
    ) -> None:
        if len(self.model.rec_noise) >= scale + 1:
            return

        actual_batch = real.shape[0]
        if scale == 0:
            z_rec = torch_utils.generate_noise(
                (self.noise_channels, *real.shape[2:]),
                device=self.device,
                num_samp=actual_batch,
            )
            z_rec = F.pad(z_rec, [self.zero_padding] * 4, value=0)
            self.model.rec_noise.append(z_rec)

            with torch.no_grad():
                fake = self.model.generator(
                    self.model.get_pyramid_noise(scale, indexes),
                    [1.0] * (scale + 1),
                    stop_scale=scale,
                )

            use_rp = getattr(self.options, "use_rock_physics", False)
            res_real = torch_utils.split_facies_rp(
                real, self.options.num_facies_classes, has_rp=use_rp
            )
            facies_only = res_real["facies"]
            res_fake = torch_utils.split_facies_rp(
                fake, self.options.num_facies_classes, has_rp=use_rp
            )
            fake_only = res_fake["facies"]

            assert facies_only is not None
            assert fake_only is not None

            rmse = torch.sqrt(F.mse_loss(fake_only, facies_only.to(fake.device)))
            amp = self.scale0_noise_amp * rmse
            if len(self.model.noise_amps) <= scale:
                self.model.noise_amps.append(amp)
            else:
                self.model.noise_amps[scale] = amp
            return

        num_cond_channels = 0
        if len(wells_pyramid) > 0:
            num_cond_channels += self.options.num_facies_classes
        if len(seismic_pyramid) > 0:
            num_cond_channels += 1

        noise_ch = self.noise_channels - num_cond_channels
        z_rec = torch_utils.generate_noise(
            (noise_ch, *real.shape[2:]),
            device=self.device,
            num_samp=actual_batch,
        )

        to_concat = [z_rec]
        if len(wells_pyramid) > 0:
            to_concat.append(wells_pyramid[scale].to(self.device))
        if len(seismic_pyramid) > 0:
            to_concat.append(seismic_pyramid[scale].to(self.device))

        if len(to_concat) > 1:
            z_rec = torch.cat(to_concat, dim=1)

        z_rec = F.pad(z_rec, [self.zero_padding] * 4, value=0)
        self.model.rec_noise.append(z_rec)

        with torch.no_grad():
            fake = self.model.generator(
                self.model.get_pyramid_noise(
                    scale,
                    indexes,
                    wells_pyramid,
                    seismic_pyramid,
                ),
                self.model.noise_amps + [1.0],
                stop_scale=scale,
            )

            use_rp = getattr(self.options, "use_rock_physics", False)
            res_real = torch_utils.split_facies_rp(
                real, self.options.num_facies_classes, has_rp=use_rp
            )
            facies_only = res_real["facies"]
            res_fake = torch_utils.split_facies_rp(
                fake, self.options.num_facies_classes, has_rp=use_rp
            )
            fake_only = res_fake["facies"]
            assert facies_only is not None and fake_only is not None
            rmse = torch.sqrt(F.mse_loss(fake_only, facies_only.to(fake.device)))
            if torch.isnan(rmse) or torch.isinf(rmse):
                print(
                    f"[DEBUG] RMSE is non-finite at scale {scale}: {rmse.item()}. Defaulting to 1e-4."
                )
                rmse = torch.tensor(1e-4, device=rmse.device, dtype=rmse.dtype)

        # Zero-Sync: Perform max logic on GPU
        min_amp_t = torch.tensor(
            self.min_noise_amp, device=self.device, dtype=rmse.dtype
        )
        amp = torch.max(self.noise_amp * rmse, min_amp_t)

        if scale < len(self.model.noise_amps):
            self.model.noise_amps[scale] = (amp + self.model.noise_amps[scale]) / 2
        else:
            self.model.noise_amps.append(amp)

    def create_batch_iterator(
        self, loader: DataLoader[tuple[int, Batch] | Batch], scales: tuple[int, ...]
    ) -> Iterator[PyramidsBatch | None]:
        prefetcher = TorchDataPrefetcher(
            loader, scale_indices=scales, device=self.device
        )
        self._batch_prefetcher = prefetcher
        return iter(prefetcher)

    def collect_seen_indices(self) -> list[int]:
        if self._batch_prefetcher is None:
            return []
        return gather_seen_indices(self._batch_prefetcher)

    def _to_pyramid(
        self, component: tuple[torch.Tensor, ...]
    ) -> dict[int, torch.Tensor]:
        if not component:
            return {}

        return {
            idx: utils.to_device(
                component[idx],
                self.device,
                channels_last=True,
                non_blocking=True,
            )
            for idx in range(len(component))
        }

    def _sample_seen_batch(self) -> Batch | None:
        if not self._seen_indices_for_save:
            return None

        sample_count = min(self.num_real_facies, len(self._seen_indices_for_save))
        if sample_count <= 0:
            return None

        chosen_positions = torch.randperm(len(self._seen_indices_for_save))[
            :sample_count
        ]
        sampled_items = [
            cast(
                tuple[Any, Batch],
                self.dataset[self._seen_indices_for_save[int(position.item())]],
            )
            for position in chosen_positions
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

    def setup_optimizers(self, scales: tuple[int, ...]) -> None:
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

    def save_optimizers(
        self,
        scale_path: str,
        generator_optimizer: Any,
        discriminator_optimizer: Any,
        generator_scheduler: Any,
        discriminator_scheduler: Any,
    ) -> None:
        """Save optimizer and scheduler state dicts to disk."""
        os.makedirs(scale_path, exist_ok=True)
        torch.save(
            generator_optimizer.state_dict(), os.path.join(scale_path, OPT_G_FILE)
        )
        torch.save(
            discriminator_optimizer.state_dict(), os.path.join(scale_path, OPT_D_FILE)
        )
        torch.save(
            generator_scheduler.state_dict(), os.path.join(scale_path, SCH_G_FILE)
        )
        torch.save(
            discriminator_scheduler.state_dict(), os.path.join(scale_path, SCH_D_FILE)
        )

    def load_model(self, scale: int) -> None:
        """Load generator and discriminator state dicts for a specific scale."""
        try:
            generator_path = os.path.join(str(self.checkpoint_path), str(scale), G_FILE)
            discriminator_path = os.path.join(
                str(self.checkpoint_path), str(scale), D_FILE
            )

            gen = unwrap_ddp(self.model.generator.gens[scale])
            gen.load_state_dict(
                torch_utils.load(generator_path, self.device, as_type=Mapping[str, Any])
            )
            disc = unwrap_ddp(self.model.discriminator.discs[scale])
            disc.load_state_dict(
                torch_utils.load(
                    discriminator_path, self.device, as_type=Mapping[str, Any]
                )
            )
        except Exception as e:
            print(f"Error loading models from {self.checkpoint_path}/{scale}: {e}")
            raise

    def reset_schedulers(self, scales: tuple[int, ...]) -> None:
        for scale in scales:
            self.discriminator_schedulers[scale] = torch.optim.lr_scheduler.StepLR(
                self.discriminator_optimizers[scale],
                step_size=self.lr_decay,
                gamma=self.gamma,
            )

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
        num_facies_ch = self.options.num_facies_classes

        for scale in scales:
            real_facies = facies_pyramid.get(scale)
            if real_facies is None:
                continue

            sample_count = real_facies.shape[0]
            if sample_count == 0:
                continue

            tiled_indexes = torch.arange(
                sample_count, device=self.device
            ).repeat_interleave(self.num_generated_per_real)

            noises = self.model.get_pyramid_noise(
                scale, tiled_indexes, wells_pyramid, seismic_pyramid
            )

            with torch.no_grad():
                generated_facies = self.model.generator(
                    noises, self.model.noise_amps[: scale + 1], stop_scale=scale
                ).clamp(-1, 1)

            facies_tensor = generated_facies.reshape(
                sample_count,
                self.num_generated_per_real,
                *generated_facies.shape[1:],
            )
            real_facies_tensor = real_facies

            facies_cpu = facies_tensor.detach().to("cpu", non_blocking=True)
            real_cpu = real_facies_tensor.detach().to("cpu", non_blocking=True)

            masks_cpu = None
            if scale in masks_pyramid:
                masks_cpu = masks_pyramid[scale].detach().to("cpu", non_blocking=True)

            if self.device.type == "cuda":
                torch.cuda.current_stream().synchronize()

            res_gen = torch_utils.split_facies_rp(
                facies_cpu, num_facies_ch, has_rp=use_rock_physics
            )
            facies_only_cpu, rp_cpu = res_gen["facies"], res_gen["rock_physics"]

            res_real = torch_utils.split_facies_rp(
                real_cpu, num_facies_ch, has_rp=use_rock_physics
            )
            real_facies_only_cpu, real_rp_cpu = (
                res_real["facies"],
                res_real["rock_physics"],
            )

            if facies_only_cpu is None or real_facies_only_cpu is None:
                continue

            masks_np = torch2np(masks_cpu) if masks_cpu is not None else None
            out_dir = outputs_path.get(scale)
            if out_dir is None:
                continue

            bw.submit_plot_generated_outputs(
                torch2np(facies_only_cpu, denormalize=True),
                torch2np(real_facies_only_cpu, denormalize=True),
                scale,
                epoch,
                out_dir,
                masks_np,
                batch_id=batch_id,
            )

            if use_rock_physics and rp_cpu is not None and real_rp_cpu is not None:

                # Ip  (channel 0)
                # rp_cpu is 5D (sample, gen, C, H, W) → channel dim is 2
                # real_rp_cpu is 4D (sample, C, H, W)  → channel dim is 1
                ip_path = out_dir.replace(OUTPUT_FACIES_PATH, OUTPUT_IP_PATH)
                os.makedirs(ip_path, exist_ok=True)
                bw.submit_plot_generated_outputs(
                    torch2np(rp_cpu[:, :, 0:1, ...], denormalize=True),
                    torch2np(real_rp_cpu[:, 0:1, ...], denormalize=True),
                    scale,
                    epoch,
                    ip_path,
                    None,
                    batch_id=batch_id,
                    plot_title="Acoustic Impedance (Ip)",
                    cmap="magma",
                )

                # Is  (channel 1)
                is_path = out_dir.replace(OUTPUT_FACIES_PATH, OUTPUT_IS_PATH)
                os.makedirs(is_path, exist_ok=True)
                bw.submit_plot_generated_outputs(
                    torch2np(rp_cpu[:, :, 1:2, ...], denormalize=True),
                    torch2np(real_rp_cpu[:, 1:2, ...], denormalize=True),
                    scale,
                    epoch,
                    is_path,
                    None,
                    batch_id=batch_id,
                    plot_title="Shear Impedance (Is)",
                    cmap="magma",
                )

                # Vp/Vs  (channel 2)
                vp_vs_path = out_dir.replace(OUTPUT_FACIES_PATH, OUTPUT_VP_VS_PATH)
                os.makedirs(vp_vs_path, exist_ok=True)
                bw.submit_plot_generated_outputs(
                    torch2np(rp_cpu[:, :, 2:3, ...], denormalize=True),
                    torch2np(real_rp_cpu[:, 2:3, ...], denormalize=True),
                    scale,
                    epoch,
                    vp_vs_path,
                    None,
                    batch_id=batch_id,
                    plot_title="Vp/Vs Ratio",
                    cmap="viridis",
                )

    def save_epoch_checkpoint(
        self,
        scales: tuple[int, ...],
        scale_paths: dict[int, str],
        epoch: int,
        batch_id: int,
    ):
        """Save a complete training checkpoint for the current epoch and batch."""
        if not self._is_main_process:
            return

        # Use the directory of the first scale in the group as the checkpoint anchor
        anchor_scale = min(scales)
        checkpoint_path = os.path.join(
            scale_paths[anchor_scale], "epoch_checkpoint.pth"
        )

        checkpoint: dict[str, object] = {
            "epoch": epoch,
            "batch_id": batch_id,
            "loss_scale_factors": self.model.loss_scale_factors,
            "generator_state_dict": unwrap_ddp(self.model.generator).state_dict(),
            "discriminator_states": {
                s: unwrap_ddp(self.model.discriminator.discs[s]).state_dict()
                for s in scales
            },
            "generator_optimizers": {
                s: self.generator_optimizers[s].state_dict() for s in scales
            },
            "discriminator_optimizers": {
                s: self.discriminator_optimizers[s].state_dict() for s in scales
            },
            "generator_schedulers": {
                s: self.generator_schedulers[s].state_dict() for s in scales
            },
            "discriminator_schedulers": {
                s: self.discriminator_schedulers[s].state_dict() for s in scales
            },
        }

        torch.save(checkpoint, checkpoint_path)
        from tqdm import tqdm

        tqdm.write(f"  --> Saved epoch checkpoint to {checkpoint_path}")

    def load_epoch_checkpoint(
        self, scales: tuple[int, ...], scale_paths: dict[int, str]
    ) -> tuple[int, int]:
        """Restore training state from a saved epoch checkpoint."""
        anchor_scale = min(scales)
        checkpoint_path = os.path.join(
            scale_paths[anchor_scale], "epoch_checkpoint.pth"
        )

        if not os.path.isfile(checkpoint_path):
            return 0, 0

        # Map to CPU first to avoid memory spikes on Rank 0
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # 1. Restore Model Weights
        unwrap_ddp(self.model.generator).load_state_dict(
            checkpoint["generator_state_dict"]
        )
        for s in scales:
            if s in checkpoint["discriminator_states"]:
                unwrap_ddp(self.model.discriminator.discs[s]).load_state_dict(
                    checkpoint["discriminator_states"][s]
                )

        # 2. Restore Optimizer and Scheduler States
        for s in scales:
            if s in checkpoint["generator_optimizers"]:
                self.generator_optimizers[s].load_state_dict(
                    checkpoint["generator_optimizers"][s]
                )
            if s in checkpoint["discriminator_optimizers"]:
                self.discriminator_optimizers[s].load_state_dict(
                    checkpoint["discriminator_optimizers"][s]
                )
            if s in checkpoint["generator_schedulers"]:
                self.generator_schedulers[s].load_state_dict(
                    checkpoint["generator_schedulers"][s]
                )
            if s in checkpoint["discriminator_schedulers"]:
                self.discriminator_schedulers[s].load_state_dict(
                    checkpoint["discriminator_schedulers"][s]
                )

        # 3. Restore Auxiliary Metadata
        self.model.loss_scale_factors = checkpoint.get("loss_scale_factors", {})

        return checkpoint["epoch"], checkpoint["batch_id"]
