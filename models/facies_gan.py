"""Unified FaciesGAN implementation.

This module provides the concrete PyTorch implementation of the FaciesGAN
architecture, supporting parallel training of multiple pyramid scales with
optimized PyTorch logic (AMP, DDP, torch.compile).
"""

import math
import os
import time
from typing import cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

# Removed legacy constants import
from config import CheckpointFilenames
from datasets.utils import generate_scales
from device import device_manager
from enums import DeviceType, MetricKey
from models import utils
from models.discriminator import Discriminator
from models.generator import Generator
from models.gradnorm import GradNorm
from models.utils import ChannelKey
from options import TrainingOptions
from physics.physics import PhysicsState
from training.metrics import (
    DiscriminatorMetrics,
    GeneratorMetrics,
    ScaleMetrics,
    compute_adversarial_loss,
    compute_diversity_loss,
    compute_gradient_penalty,
    compute_masked_loss,
    compute_reconstruction_loss,
    compute_rock_physics_loss,
)
from utils import get_padding_value


def unwrap_ddp(module: nn.Module) -> nn.Module:
    """Return the inner module if wrapped in ``DistributedDataParallel``.

    Parameters
    ----------
    module : nn.Module
        The module to unwrap.

    Returns
    -------
    nn.Module
        The unwrapped module.
    """
    return getattr(module, "module", module)  # type: ignore[no-any-return]


# noinspection PyDefaultArgument
class FaciesGAN(nn.Module):
    """Unified FaciesGAN implementation.

    This class manages the lifecycle of Generators and Discriminators,
    initializes them, and provides helpers for the training loop.
    It supports parallel training of multiple pyramid scales.

    Attributes
    ----------
    generator : Generator
        The multiscale PyTorch generator instance.
    discriminator : Discriminator
        The multiscale PyTorch discriminator instance.
    options : TrainingOptions
        Training configuration containing hyperparameters.
    """

    generator: Generator
    discriminator: Discriminator
    gradnorm: GradNorm | None

    noise_amp: torch.Tensor
    min_noise_amp: torch.Tensor
    scale0_noise_amp: torch.Tensor

    def __init__(
        self,
        options: TrainingOptions,
        channels: dict[ChannelKey, int],
    ) -> None:
        """Initialize the FaciesGAN model.

        Parameters
        ----------
        options : TrainingOptions
            Training options containing hyperparameters and configuration.
        channels : dict[ChannelKey, int]
            Dictionary mapping channel keys to the number of channels for each component.
        """
        super().__init__()

        # --- Architecture & Hyperparameters ---
        self.options = options
        self.num_facies_channels = channels[ChannelKey.FACIES]
        self.total_output_channels = channels[ChannelKey.GENERATOR_OUT]

        self.disc_input_channels: int = self.total_output_channels
        self.gen_input_channels: int = channels[ChannelKey.NOISE]
        self.gen_output_channels: int = self.total_output_channels
        self.base_channel = self.total_output_channels

        self.num_noise_channels = max(
            options.noise_channels, self.total_output_channels
        )

        # --- Calibration Buffers ---
        # Registered as buffers so they are part of the state_dict and
        # persist across checkpoints.
        self.register_buffer(
            "noise_amp",
            torch.tensor(
                options.noise_amp,
                dtype=torch.float32,
                device=device_manager.device,
            ),
        )
        self.register_buffer(
            "min_noise_amp",
            torch.tensor(
                options.min_noise_amp,
                dtype=torch.float32,
                device=device_manager.device,
            ),
        )
        self.register_buffer(
            "scale0_noise_amp",
            torch.tensor(
                options.scale0_noise_amp,
                dtype=torch.float32,
                device=device_manager.device,
            ),
        )

        # --- State Tracking ---
        self.shapes: tuple[tuple[int, ...], ...] = generate_scales(options)
        self.rec_noise: list[torch.Tensor] = []
        self.noise_amps: list[torch.Tensor] = []

        self.active_scales: set[int] = set()

        self.disc_step_counter: int = 0
        self.extra_disc_step_counter: int = 0

        self.padding_value: float = get_padding_value(options.normalization_range)
        self.zero_padding = int(options.num_layer * math.floor(options.kernel_size / 2))

        self._wavelets_z_cache: dict[int, torch.Tensor] = {}

        # --- Performance & Parallelism ---
        self.use_amp = device_manager.is_cuda
        self.amp_dtype = (
            torch.bfloat16
            if str(options.amp_dtype).lower() == "bf16"
            else torch.float16
        )
        self.use_grad_scaler = self.use_amp and self.amp_dtype == torch.float16
        self.grad_scaler_g = GradScaler(enabled=self.use_grad_scaler)

        self.use_compile = device_manager.is_cuda and options.compile_backend
        self.zero_scalar = torch.tensor(0.0, device=device_manager.device)
        self.use_gradient_checkpointing = options.gradient_checkpointing

        self.uncompiled_discs: dict[int, nn.Module] = {}
        self.pending_all_reduce_work: dist.Work | None = None

        self.current_epoch: int = 0
        self.rec_skip_epochs: int = 0
        self.div_skip_epochs: int = 0

        # Pre-allocated noise buffers for optimized training
        self.d_noise_buffers: dict[tuple[int, ...], torch.Tensor] = {}
        self.g_noise_buffers: dict[tuple[int, ...], torch.Tensor] = {}

        # self.zero_scalar already allocated

        # Profiling
        self.profile_all_reduce = False
        self.profile_all_reduce_calls = 0
        self.profile_all_reduce_total_s = 0.0
        self.profile_collective_total_s = 0.0
        self.profile_all_reduce_total_elems = 0

        # Initialization
        self.generator = Generator(
            num_layer=options.num_layer,
            kernel_size=options.kernel_size,
            padding_size=options.padding_size,
            padding_value=self.padding_value,
            stride=options.stride,
            input_channels=self.gen_input_channels,
            output_channels=self.gen_output_channels,
            num_facies=self.num_facies_channels,
            normalization_range=options.normalization_range,
        )
        self.discriminator = Discriminator(
            num_layer=options.num_layer,
            kernel_size=options.kernel_size,
            padding_size=options.padding_size,
            stride=options.stride,
            input_channels=self.disc_input_channels,
        )

        if self.use_gradient_checkpointing:
            self.generator.use_gradient_checkpointing = True

        self.physics_state = PhysicsState(self.options, self.shapes)

        if self.use_compile:
            # noinspection PyTypeChecker
            self.generator.color_quantizer = torch.compile(  # type: ignore
                self.generator.color_quantizer,
                fullgraph=True,
                dynamic=True,
                mode="default",
            )
            # noinspection PyTypeChecker
            self.generator._residual_clamp = torch.compile(  # type: ignore
                self.generator._residual_clamp,  # type: ignore
                fullgraph=True,
                dynamic=True,
            )

        # Rank-0 compile progress indicator (useful with max-autotune mode).
        is_rank0 = device_manager.is_main_process
        self._compile_progress_enabled = bool(self.use_compile and is_rank0)
        # Total tick count estimation for the progress bar.
        # planned = discriminator blocks + generator blocks + 2 (quantizer, clamp).
        planned_scales = int(getattr(options, "stop_scale", 0)) + 1
        planned_gen = (
            0 if getattr(options, "gradient_checkpointing", False) else planned_scales
        )
        planned_disc = planned_scales
        self._compile_progress_total = planned_disc + planned_gen + 2
        self._compile_progress_done = 0
        self._compile_progress_width = 34
        self._compile_progress_t0 = time.time()
        self._compile_progress_last_t = self._compile_progress_t0

        self._compiled_disc_seen: set[int] = set()

        # Route generator first-use compile events to this model progress bar.
        self.generator.compile_progress_callback = self._tick_compile_progress

        # --- GradNorm Initialization ---
        self.gradnorm = None
        if getattr(options, "use_gradnorm", False):
            all_scales = list(range(options.stop_scale + 1))
            self.gradnorm = GradNorm(
                scales=all_scales,
                alpha=getattr(options, "gradnorm_alpha", 0.15),
                lr=getattr(options, "gradnorm_lr", 0.0005),
            )

    # ---------------------------------------------------------------------------gradnorm.w
    # Training Orchestration
    # ---------------------------------------------------------------------------

    def _tick_compile_progress(self, label: str) -> None:
        """Render persistent compile progress lines on rank 0."""
        if not self._compile_progress_enabled:
            return

        now = time.time()
        self._compile_progress_done += 1
        total = max(1, int(self._compile_progress_total))
        done = min(int(self._compile_progress_done), total)
        ratio = done / total

        filled = int(self._compile_progress_width * ratio)
        bar = "#" * filled + "-" * (self._compile_progress_width - filled)

        elapsed = now - self._compile_progress_t0
        delta = now - self._compile_progress_last_t
        self._compile_progress_last_t = now

        msg = (
            f"\r  [compile] [{bar}] {done}/{total} "
            f"({ratio:4.0%}) | +{delta:4.1f}s | {elapsed:4.1f}s total | {label:<18}    "
        )

        # Use sys.stdout.write for stable \r carriage return across OSes.
        import sys

        sys.stdout.write(msg)
        sys.stdout.flush()

        if done >= total:
            sys.stdout.write("\n")
            sys.stdout.flush()
            # Disable bar so it does not accidentally re-render on later epochs.
            self._compile_progress_enabled = False

    def finish_compile_progress(self) -> None:
        """Force the compile progress bar to 100% and print a final newline.

        Should be called by the Trainer after all expected warmup traces have
        been triggered to ensure the terminal is cleanly yielded.
        """
        if not self._compile_progress_enabled:
            return
        self._compile_progress_done = self._compile_progress_total
        self._tick_compile_progress("finished")

    def _mark_disc_compile_progress(self, scale: int) -> None:
        """Tick compile progress once when a compiled discriminator is first used."""
        if scale in self._compiled_disc_seen:
            return
        self._compiled_disc_seen.add(scale)
        self._tick_compile_progress(f"disc_scale_{scale}")

    def forward(
        self,
        generator_optimizers: dict[int, torch.optim.Optimizer],
        discriminator_optimizers: dict[int, torch.optim.Optimizer],
        indexes: torch.Tensor,
        facies_pyramid: dict[int, torch.Tensor],
        rec_in_pyramid: dict[int, torch.Tensor],
        wells_pyramid: dict[int, torch.Tensor] = {},
        masks_pyramid: dict[int, torch.Tensor] = {},
        seismic_pyramid: dict[int, torch.Tensor] = {},
    ) -> ScaleMetrics:
        """Perform a forward pass and compute scale metrics for active scales.

        This method orchestrates both discriminator and generator optimization steps.

        Parameters
        ----------
        generator_optimizers : dict[int, torch.optim.Optimizer]
            Optimizers for the generator, keyed by scale.
        discriminator_optimizers : dict[int, torch.optim.Optimizer]
            Optimizers for the discriminator, keyed by scale.
        indexes : torch.Tensor
            Batch indices for the current step.
        facies_pyramid : dict[int, torch.Tensor]
            Real facies tensors for each scale.
        rec_in_pyramid : dict[int, torch.Tensor]
            Reconstruction inputs for each scale.
        wells_pyramid : dict[int, torch.Tensor], optional
            Well conditioning data.
        masks_pyramid : dict[int, torch.Tensor], optional
            Well masks for loss computation.
        seismic_pyramid : dict[int, torch.Tensor], optional
            Seismic conditioning data.

        Returns
        -------
        ScaleMetrics
            Detached metrics for logging.
        """
        disc_metrics_tuple = self.optimize_discriminator(
            indexes,
            discriminator_optimizers,
            facies_pyramid,
            wells_pyramid,
            seismic_pyramid,
        )
        gen_metrics_tuple = self.optimize_generator(
            indexes,
            generator_optimizers,
            facies_pyramid,
            rec_in_pyramid,
            wells_pyramid,
            masks_pyramid,
            seismic_pyramid,
        )

        discriminator_metrics = {
            scale: disc_metrics_tuple[i]
            for i, scale in enumerate(sorted(self.active_scales))
        }
        generator_metrics = {
            scale: gen_metrics_tuple[i]
            for i, scale in enumerate(sorted(self.active_scales))
        }

        return ScaleMetrics(
            discriminator=discriminator_metrics, generator=generator_metrics
        )

    def optimize_discriminator(
        self,
        indexes: torch.Tensor,
        optimizers: dict[int, torch.optim.Optimizer],
        facies_pyramid: dict[int, torch.Tensor],
        wells_pyramid: dict[int, torch.Tensor] = {},
        seismic_pyramid: dict[int, torch.Tensor] = {},
    ) -> tuple[DiscriminatorMetrics, ...]:
        """Perform discriminator optimization with gradient accumulation.

        Generates fakes upfront and iterates through discriminator steps.
        Supports parallel training optimizations and DDP coalesced all-reduce.

        Parameters
        ----------
        indexes : torch.Tensor
            Batch indices.
        optimizers : dict[int, torch.optim.Optimizer]
            Discriminator optimizers.
        facies_pyramid : dict[int, torch.Tensor]
            Real data.
        wells_pyramid : dict[int, torch.Tensor], optional
        seismic_pyramid : dict[int, torch.Tensor], optional

        Returns
        -------
        tuple[DiscriminatorMetrics, ...]
            Metrics from the last step for each active scale.
        """
        d = self.options.discriminator_steps
        if d <= 0:
            return ()

        sorted_scales = sorted(self.active_scales)
        b = len(indexes)

        scale0_multi = self.options.scale0_disc_steps_multiplier
        pre_faked: dict[int, list[torch.Tensor]] = {}

        # ── Pre-generate all d fakes per scale in one batched forward ──
        with torch.inference_mode(), autocast(
            DeviceType.CUDA, enabled=self.use_amp, dtype=self.amp_dtype
        ):
            for scale in sorted_scales:
                d_count = d * scale0_multi if scale == 0 else d
                batched_noises = self.get_batched_d_noise(
                    scale, d_count, b, indexes, wells_pyramid, seismic_pyramid
                )
                amps = self.get_noise_amplitude(scale)
                batched_fake = self.generator(batched_noises, amps, stop_scale=scale)
                pre_faked[scale] = list(torch.chunk(batched_fake, d_count, dim=0))

        last_gp: dict[int, torch.Tensor] = {}

        # Pre-fetch models to avoid dict lookups in the hot loop
        discs = {s: unwrap_ddp(self.discriminator.discs[s]) for s in sorted_scales}
        ddp_discs_all = [self.discriminator.discs[s] for s in sorted_scales]
        ddp_disc_0 = [self.discriminator.discs[0]] if 0 in sorted_scales else []

        def _disc_step(
            scale: int, step_idx: int, compute_gp: bool
        ) -> tuple[torch.Tensor, torch.Tensor]:
            self.optimizer_zero_grad(optimizers[scale])
            fake = pre_faked[scale][step_idx]
            real = facies_pyramid[scale]

            with autocast(DeviceType.CUDA, enabled=self.use_amp, dtype=self.amp_dtype):
                d_both = discs[scale](torch.cat([real, fake], dim=0))
                d_real, d_fake = d_both[:b], d_both[b:]
                rl = -d_real.mean()
                fl = d_fake.mean()

            if compute_gp:
                disc_mod = self.uncompiled_discs.get(scale)
                if disc_mod is None:
                    disc_mod = self.discriminator.discs[scale]
                disc = unwrap_ddp(disc_mod)
                lambda_gp = (
                    self.options.scale0_gp_alpha
                    if scale == 0 and self.options.scale0_gp_alpha > 0.0
                    else self.options.gradient_loss_penalty
                )
                gp = compute_gradient_penalty(disc, real, fake.detach(), lambda_gp)
                last_gp[scale] = gp.detach()
            else:
                gp = self.zero_scalar

            total = rl + fl + gp
            total.backward()
            return rl.detach(), fl.detach()

        max_steps = d * scale0_multi if (scale0_multi > 1 and 0 in sorted_scales) else d
        step_metrics: list[DiscriminatorMetrics] = [None] * len(sorted_scales)  # type: ignore

        # ── Unified d-step loop ──
        # gp_scale intentionally removed: multiplying by gp_interval creates
        # large gradient spikes that destabilise training at high resolutions.
        # The lazy interval only saves compute; the per-step penalty magnitude
        # is the same as gp_interval=1 (controlled by gradient_loss_penalty).
        for step_idx in range(max_steps):
            if step_idx < d:
                self.disc_step_counter += 1
                compute_gp = (self.disc_step_counter == 1) or (
                    self.disc_step_counter % self.options.gp_interval
                ) == 0
                active_scales = sorted_scales
                ddp_modules = ddp_discs_all
            else:
                self.extra_disc_step_counter += 1
                compute_gp = (self.extra_disc_step_counter == 1) or (
                    self.extra_disc_step_counter % self.options.gp_interval
                ) == 0
                active_scales = [0]
                ddp_modules = ddp_disc_0

            raw_losses: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
            for scale in active_scales:
                self._mark_disc_compile_progress(scale)
                raw_losses[scale] = _disc_step(scale, step_idx, compute_gp)

            # Phase 2: Coalesced all-reduce
            if device_manager.is_distributed:
                self.all_reduce_grads_coalesced(ddp_modules)

            # Phase 3: Step optimizers and build metrics
            for scale in active_scales:
                optimizers[scale].step()

                # Only record metrics on the final step for this scale
                is_final_step = (scale == 0 and step_idx == max_steps - 1) or (
                    scale != 0 and step_idx == d - 1
                )
                if is_final_step:
                    rl, fl = raw_losses[scale]
                    # last_gp is reset to {} at the start of this call, so it only
                    # contains GP values computed during the current forward() pass.
                    # Report whatever fired this call (may be zero if GP interval
                    # did not coincide with any step in this call).
                    gp_val = last_gp.get(scale, self.zero_scalar)
                    step_metrics[sorted_scales.index(scale)] = DiscriminatorMetrics(
                        total=rl + fl + gp_val, real=rl, fake=fl, gp=gp_val
                    )

        return tuple(step_metrics)

    def optimize_generator(
        self,
        indexes: torch.Tensor,
        optimizers: dict[int, torch.optim.Optimizer],
        facies_pyramid: dict[int, torch.Tensor],
        rec_in_pyramid: dict[int, torch.Tensor],
        wells_pyramid: dict[int, torch.Tensor] = {},
        masks_pyramid: dict[int, torch.Tensor] = {},
        seismic_pyramid: dict[int, torch.Tensor] = {},
    ) -> tuple[GeneratorMetrics, ...]:
        """Perform generator optimization with gradient accumulation and per-scale freezing.

        Parameters
        ----------
        indexes : torch.Tensor
            Batch indices.
        optimizers : dict[int, torch.optim.Optimizer]
            Generator optimizers, keyed by scale.
        facies_pyramid : dict[int, torch.Tensor]
            Real data.
        rec_in_pyramid : dict[int, torch.Tensor]
            Reconstruction inputs.
        wells_pyramid : dict[int, torch.Tensor], optional
        masks_pyramid : dict[int, torch.Tensor], optional
        seismic_pyramid : dict[int, torch.Tensor], optional

        Returns
        -------
        tuple[GeneratorMetrics, ...]
            Metrics from the last step for each active scale.
        """
        sorted_scales = sorted(self.active_scales)
        g = self.options.generator_steps
        if g <= 0:
            return ()

        step_metrics: list[GeneratorMetrics] = []

        # Freeze discriminator for the entire g-phase
        for s in sorted_scales:
            self.discriminator.discs[s].requires_grad_(False)

        for _ in range(g):
            step_metrics = []

            # Freeze all active gen blocks up front
            for s in sorted_scales:
                self.generator.gens[s].requires_grad_(False)

            losses_by_scale: dict[int, torch.Tensor] = {}
            for scale in sorted_scales:
                if scale >= len(facies_pyramid):
                    continue

                if len(self.noise_amps) < scale + 1:
                    raise RuntimeError(
                        f"noise_amp not initialized for scale {scale}. "
                        "Call the project's noise initialization before training."
                    )

                # Unfreeze only the target scale's gen block
                self.generator.gens[scale].requires_grad_(True)

                if self.gradnorm is not None:
                    # 1. Obter as losses individuais em float32
                    curr_losses = self.compute_generator_metrics(
                        indexes,
                        scale,
                        facies_pyramid[scale],
                        rec_in_pyramid,
                        wells_pyramid,
                        masks_pyramid,
                        seismic_pyramid,
                        return_vector=True,
                    )

                    # 2. Obter a loss total ponderada para treinar o Generator
                    total_loss = (
                        self.gradnorm.w[scale].detach() * curr_losses.clone()
                    ).sum()

                    # Obter GeneratorMetrics para logging diretamente (evita overhead massivo)
                    metrics = GeneratorMetrics(
                        total=total_loss,
                        fake=curr_losses[0],
                        rec_facies=curr_losses[1],
                        well=curr_losses[2],
                        div=curr_losses[3],
                        rec_rock_physics=curr_losses[4],
                        tv=curr_losses[5],
                        elastic=curr_losses[6],
                        seismic=curr_losses[7],
                    )
                else:
                    metrics = self.compute_generator_metrics(
                        indexes,
                        scale,
                        facies_pyramid[scale],
                        rec_in_pyramid,
                        wells_pyramid,
                        masks_pyramid,
                        seismic_pyramid,
                    )
                    total_loss = metrics.total

                # zero_grad
                self.optimizer_zero_grad(optimizers[scale])

                # GradNorm weights update: isolated single-block forward pass.
                # Builds z_in with no_grad (all previous scales + noise construction),
                # then activates grad only for the current block — giving a tiny autograd
                # graph instead of traversing the full discriminator + pyramid.
                if (
                    self.gradnorm is not None
                    and self.disc_step_counter % self.options.gradnorm_interval == 0
                ):
                    gen = self.generator
                    # Use the block as-is (compiled or not): torch.autograd.grad
                    # walks the C++ autograd graph and does NOT need the unwrapped
                    # module. Unwrapping was only required for torch.func.vjp.
                    scale_block = gen.gens[scale]

                    # Build z_in for the current scale with no_grad (no graph through pyramid)
                    with torch.no_grad():
                        noise_pyramid = self.get_pyramid_noise(scale, indexes, wells_pyramid, seismic_pyramid)
                        amp = self.get_noise_amplitude(scale)

                        # Forward through all PREVIOUS scales to get out_facie (no graph)
                        if scale == 0:
                            channels = gen.output_channels
                            b_ = noise_pyramid[0].shape[0]
                            h = noise_pyramid[0].shape[2] - gen.full_zero_padding
                            w_ = noise_pyramid[0].shape[3] - gen.full_zero_padding
                            out_facie = torch.zeros(
                                (b_, channels, h, w_),
                                device=noise_pyramid[0].device,
                                dtype=noise_pyramid[0].dtype,
                            )
                        else:
                            # Run scales 0..scale-1 with no_grad to get out_facie
                            out_facie = gen(
                                noise_pyramid,
                                amp,
                                in_noise=None,
                                start_scale=0,
                                stop_scale=scale - 1,
                            )

                        # Build z_in for the current scale
                        z_s = noise_pyramid[scale]
                        out_up = utils.interpolate(
                            out_facie,
                            (z_s.shape[2] - gen.full_zero_padding, z_s.shape[3] - gen.full_zero_padding),
                        )
                        n_in = gen.output_channels
                        p = gen.zero_padding
                        base_out = out_up[:, :n_in, ...]
                        if p > 0:
                            padded = torch.empty(
                                (base_out.shape[0], base_out.shape[1],
                                 base_out.shape[2] + 2 * p, base_out.shape[3] + 2 * p),
                                dtype=base_out.dtype, device=base_out.device,
                            ).fill_(gen.padding_value)
                            padded[..., p:-p, p:-p] = base_out
                        else:
                            padded = base_out
                        amp_s = amp[scale]
                        z_in_detached = (amp_s * z_s[:, :n_in, ...] + padded)
                        if gen.cond_channels > 0:
                            z_in_detached = torch.cat([z_in_detached, z_s[:, n_in:, ...]], dim=1)
                        # out_facie_detached for residual clamp
                        out_facie_detached = out_up.detach()

                    # Enable grad only on the current block input
                    z_in_gn = z_in_detached.detach().requires_grad_(False)

                    # Minimal task_losses_fn: only generator-local losses (no discriminator)
                    real_s = facies_pyramid[scale]
                    wells_s = wells_pyramid.get(scale)
                    masks_s = masks_pyramid.get(scale)
                    seismic_s_dict = seismic_pyramid
                    opts = self.options
                    phys = self.physics_state
                    zero = self.zero_scalar
                    _scale = scale
                    _out_detached = out_facie_detached

                    def _task_losses_fn(fake_local: torch.Tensor) -> torch.Tensor:
                        # Residual clamp (same as Generator._residual_clamp_method)
                        from models.generator import Generator as _Gen
                        fake_full = _Gen.residual_clamp_fn(
                            fake_local, _out_detached, gen.num_facies, gen.normalization_range
                        )
                        # 1. well loss
                        wl = compute_masked_loss(fake_full, real_s, wells_s, masks_s, opts)
                        # 2. TV + elastic + seismic
                        tv_l, el_l, seis_l = compute_rock_physics_loss(
                            fake_full, seismic_s_dict, _scale, opts, phys
                        )
                        # 3. diversity (single sample → zero)
                        div_l = zero
                        # 4. adv + rec approximated as zero (cannot compute efficiently here)
                        adv_l = zero
                        rec_fa_l = zero
                        rec_rp_l = zero
                        target_dtype = torch.float32
                        return torch.stack([
                            adv_l.to(dtype=target_dtype),
                            rec_fa_l.to(dtype=target_dtype),
                            wl.to(dtype=target_dtype),
                            div_l.to(dtype=target_dtype),
                            rec_rp_l.to(dtype=target_dtype),
                            tv_l.to(dtype=target_dtype),
                            el_l.to(dtype=target_dtype),
                            seis_l.to(dtype=target_dtype),
                        ])

                    self.gradnorm.update_weights_from_isolated_forward(
                        scale_block=scale_block,
                        z_in=z_in_gn,
                        task_losses_fn=_task_losses_fn,
                        scale_idx=scale,
                    )

                if self.use_grad_scaler:
                    self.grad_scaler_g.scale(total_loss).backward()  # type: ignore
                else:
                    total_loss.backward()  # type: ignore

                losses_by_scale[scale] = total_loss
                # Re-freeze
                self.generator.gens[scale].requires_grad_(False)

                detached_metrics = GeneratorMetrics(
                    total=metrics.total.detach(),
                    fake=metrics.fake.detach(),
                    rec_facies=metrics.rec_facies.detach(),
                    well=metrics.well.detach(),
                    div=metrics.div.detach(),
                    rec_rock_physics=metrics.rec_rock_physics.detach(),
                    tv=metrics.tv.detach(),
                    elastic=metrics.elastic.detach(),
                    seismic=metrics.seismic.detach(),
                )
                step_metrics.append(detached_metrics)

            # Restore requires_grad BEFORE all-reduce
            for s in sorted_scales:
                if s in losses_by_scale:
                    self.generator.gens[s].requires_grad_(True)

            # Phase 2: Coalesced all_reduce for all gen modules
            if device_manager.is_distributed:
                self.all_reduce_grads_coalesced(
                    [
                        self.generator.gens[s]
                        for s in sorted_scales
                        if s in losses_by_scale
                    ]
                )

            # Phase 3: unscale + clip + step optimizers
            _clip_norm = self.options.grad_clip_norm
            for scale in sorted_scales:
                if scale not in losses_by_scale:
                    continue
                if self.use_grad_scaler:
                    self.grad_scaler_g.unscale_(optimizers[scale])
                    if _clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.generator.gens[scale].parameters(), max_norm=_clip_norm, foreach=True
                        )
                    self.grad_scaler_g.step(optimizers[scale])
                else:
                    if _clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.generator.gens[scale].parameters(), max_norm=_clip_norm, foreach=True
                        )
                    optimizers[scale].step()
                # noinspection PyProtectedMember
                setattr(optimizers[scale], "_opt_called", True)

            if self.use_grad_scaler:
                self.grad_scaler_g.update()

            # Restore requires_grad for next g-step
            for s in sorted_scales:
                self.generator.gens[s].requires_grad_(True)

        # Unfreeze discriminator
        for s in sorted_scales:
            self.discriminator.discs[s].requires_grad_(True)

        return tuple(step_metrics)

    # ---------------------------------------------------------------------------
    # Loss & Metrics Computation
    # ---------------------------------------------------------------------------

    def compute_generator_metrics(
        self,
        indexes: torch.Tensor,
        scale: int,
        real: torch.Tensor,
        rec_in_pyramid: dict[int, torch.Tensor] = {},
        wells_pyramid: dict[int, torch.Tensor] = {},
        masks_pyramid: dict[int, torch.Tensor] = {},
        seismic_pyramid: dict[int, torch.Tensor] = {},
        precomputed_fakes: dict[int, torch.Tensor] | None = None,
        precomputed_rec: dict[int, torch.Tensor] | None = None,
        force_f32: bool = False,
        return_vector: bool = False,
    ) -> GeneratorMetrics | torch.Tensor:
        """Compute generator losses and return comprehensive metrics or raw loss vector.

        Parameters
        ----------
        indexes : torch.Tensor
            Batch indices.
        scale : int
            Current scale.
        real : torch.Tensor
            Real sample.
        rec_in_pyramid : dict[int, torch.Tensor], optional
        wells_pyramid : dict[int, torch.Tensor], optional
        masks_pyramid : dict[int, torch.Tensor], optional
        seismic_pyramid : dict[int, torch.Tensor], optional
        precomputed_fakes : dict[int, torch.Tensor], optional
        precomputed_rec : dict[int, torch.Tensor], optional
        force_f32 : bool, optional
        return_vector : bool, optional

        Returns
        -------
        GeneratorMetrics | torch.Tensor
            Computed metrics or raw stacked loss vector.
        """
        use_amp = self.use_amp and not force_f32
        amp_dtype = torch.float32 if force_f32 else self.amp_dtype

        with autocast(DeviceType.CUDA, enabled=use_amp, dtype=amp_dtype):
            # Generate diversity candidates
            if precomputed_fakes is not None and scale in precomputed_fakes:
                n_div = (
                    1
                    if self.current_epoch < self.div_skip_epochs
                    else self.options.num_diversity_samples
                )
                fake_samples = list(torch.chunk(precomputed_fakes[scale], n_div, dim=0))
            else:
                fake_samples = self.generate_diverse_samples(
                    indexes, scale, wells_pyramid, seismic_pyramid
                )
            fake = fake_samples[0]

            # WGAN generator adversarial loss
            disc_mod = self.uncompiled_discs.get(scale)
            if disc_mod is None:
                disc_mod = self.discriminator.discs[scale]
            adv_disc = unwrap_ddp(disc_mod)
            adv_loss = compute_adversarial_loss(
                adv_disc, fake, self.options.adversarial_loss_penalty
            )

            well_loss = compute_masked_loss(
                fake,
                real,
                wells_pyramid.get(scale),
                masks_pyramid.get(scale),
                self.options,
            )

            tv_loss, elastic_loss, seismic_loss = compute_rock_physics_loss(
                fake, seismic_pyramid, scale, self.options, self.physics_state
            )

            div_loss = compute_diversity_loss(
                fake_samples, self.options.diversity_loss_penalty, self.zero_scalar
            )

            if (
                getattr(self.options, "rec_facies_loss_penalty", 0) == 0
                or self.current_epoch < self.rec_skip_epochs
            ):
                rec_facies_loss, rec_rp_loss = self.zero_scalar, self.zero_scalar
            else:
                if precomputed_rec is not None and scale in precomputed_rec:
                    rec = precomputed_rec[scale]
                else:
                    rec_noise = self.get_pyramid_noise(
                        scale, indexes, wells_pyramid, seismic_pyramid, rec=True
                    )
                    rec = self.generator(
                        rec_noise,
                        self.noise_amps[: scale + 1],
                        in_noise=rec_in_pyramid[scale],
                        start_scale=scale,
                        stop_scale=scale,
                    )

                if self.options.use_rock_physics:
                    fc = self.options.num_facies_channels
                    rec_loss_facies = self.options.rec_facies_loss_penalty * F.mse_loss(
                        rec[:, :fc, ...], real[:, :fc, ...]
                    )
                    rec_loss_rp = (
                        self.options.rec_rock_physics_loss_penalty
                        * F.huber_loss(
                            rec[:, fc : fc + 3, ...], real[:, fc : fc + 3, ...]
                        )
                    )
                    rec_facies_loss, rec_rp_loss = rec_loss_facies, rec_loss_rp
                else:
                    rec_loss_facies = self.options.rec_facies_loss_penalty * F.mse_loss(
                        rec, real
                    )
                    rec_facies_loss, rec_rp_loss = rec_loss_facies, self.zero_scalar

            # Apply extra loss weight at scale 0 to anchor the pyramid
            if scale == 0 and self.options.scale0_loss_multiplier != 1.0:
                rec_facies_loss *= self.options.scale0_loss_multiplier
                rec_rp_loss *= self.options.scale0_loss_multiplier
                well_loss *= self.options.scale0_loss_multiplier
                adv_loss *= self.options.scale0_loss_multiplier
                div_loss *= self.options.scale0_loss_multiplier
                tv_loss *= self.options.scale0_loss_multiplier
                elastic_loss *= self.options.scale0_loss_multiplier
                seismic_loss *= self.options.scale0_loss_multiplier

            if return_vector:
                # Ensure correct order mapping to MetricKey.generator_keys()
                target_dtype = (
                    torch.float32
                    if force_f32
                    else (amp_dtype if use_amp else torch.float32)
                )
                return torch.stack(
                    [
                        adv_loss.to(device=indexes.device, dtype=target_dtype),
                        rec_facies_loss.to(device=indexes.device, dtype=target_dtype),
                        well_loss.to(device=indexes.device, dtype=target_dtype),
                        div_loss.to(device=indexes.device, dtype=target_dtype),
                        rec_rp_loss.to(device=indexes.device, dtype=target_dtype),
                        tv_loss.to(device=indexes.device, dtype=target_dtype),
                        elastic_loss.to(device=indexes.device, dtype=target_dtype),
                        seismic_loss.to(device=indexes.device, dtype=target_dtype),
                    ]
                )

            total = (
                adv_loss
                + well_loss
                + rec_facies_loss
                + rec_rp_loss
                + div_loss
                + tv_loss
                + elastic_loss
                + seismic_loss
            )

        metrics = GeneratorMetrics(
            total=total,
            fake=adv_loss.detach(),
            rec_facies=rec_facies_loss.detach(),
            well=well_loss.detach(),
            div=div_loss.detach(),
            rec_rock_physics=rec_rp_loss.detach(),
            tv=tv_loss.detach(),
            elastic=elastic_loss.detach(),
            seismic=seismic_loss.detach(),
        )

        return metrics

    # ---------------------------------------------------------------------------
    # Noise & Sample Generation
    # ---------------------------------------------------------------------------

    def generate_diverse_samples(
        self,
        indexes: torch.Tensor,
        scale: int,
        wells_pyramid: dict[int, torch.Tensor] = {},
        seismic_pyramid: dict[int, torch.Tensor] = {},
    ) -> list[torch.Tensor]:
        """Generate multiple candidate outputs for `scale` using current generator."""
        n = (
            1
            if self.current_epoch < self.div_skip_epochs
            else self.options.num_diversity_samples
        )
        b = len(indexes)

        batched_noises = self.build_batched_noise(
            n, b, scale, indexes, wells_pyramid, seismic_pyramid
        )
        amps = self.get_noise_amplitude(scale)
        batched_out = self.generator(batched_noises, amps, stop_scale=scale)

        if n <= 1:
            return [batched_out]

        return list(torch.chunk(batched_out, n, dim=0))

    def get_batched_noise(
        self,
        n: int,
        b: int,
        scale: int,
        indexes: torch.Tensor,
        wells_pyramid: dict[int, torch.Tensor] = {},
        seismic_pyramid: dict[int, torch.Tensor] = {},
    ) -> list[torch.Tensor]:
        """Generate optimized pre-allocated noise buffers for the G-phase."""
        return self.build_batched_noise(
            n, b, scale, indexes, wells_pyramid, seismic_pyramid
        )

    def get_batched_d_noise(
        self,
        scale: int,
        d: int,
        b: int,
        indexes: torch.Tensor,
        wells_pyramid: dict[int, torch.Tensor],
        seismic_pyramid: dict[int, torch.Tensor],
    ) -> list[torch.Tensor]:
        """Generate optimized noise buffers for the D-phase."""
        return self.build_batched_noise(
            d, b, scale, indexes, wells_pyramid, seismic_pyramid, channels_last=True
        )

    def build_batched_noise(
        self,
        steps: int,
        b: int,
        scale: int,
        indexes: torch.Tensor,
        wells_pyramid: dict[int, torch.Tensor],
        seismic_pyramid: dict[int, torch.Tensor],
        *,
        channels_last: bool = False,
    ) -> list[torch.Tensor]:
        """Build pre-allocated batched noise buffers for G- or D-phase.

        Parameters
        ----------
        steps : int
            Number of steps (n for G-phase, d for D-phase).
        b : int
            Batch size.
        scale : int
            Maximum pyramid scale to build noise for.
        indexes : torch.Tensor | list[int]
            Sample indices used for conditioning alignment.
        wells_pyramid : dict[int, torch.Tensor]
        seismic_pyramid : dict[int, torch.Tensor]
        channels_last : bool, optional
            Use ``torch.channels_last`` memory format (default False).
        """
        total_samples = steps * b
        p = self.zero_padding
        result: list[torch.Tensor] = []

        for lvl in range(scale + 1):
            spatial = self.get_noise_shape(lvl, use_base_channel=False)
            height, width = spatial[0], spatial[1]
            total_channels = self.gen_input_channels
            noise_channels = total_channels

            w_on_device: torch.Tensor | None = None
            s_on_device: torch.Tensor | None = None

            if wells_pyramid:
                w_local = wells_pyramid[lvl]
                w_local = w_local.to(device_manager.device, non_blocking=True)
                if isinstance(indexes, torch.Tensor) and w_local.shape[0] != b:
                    w_local = w_local[indexes]
                elif isinstance(indexes, list) and w_local.shape[0] != b:
                    w_local = w_local[indexes]
                w_on_device = w_local
                noise_channels -= w_local.shape[1]

            if seismic_pyramid:
                s_local = (
                    seismic_pyramid[lvl]
                    if not isinstance(indexes, torch.Tensor)
                    else seismic_pyramid.get(lvl)
                )
                if s_local is not None:
                    s_local = s_local.to(device_manager.device, non_blocking=True)
                    if s_local.shape[0] != b:
                        s_local = s_local[indexes]
                    elif isinstance(indexes, list) and s_local.shape[0] != b:
                        s_local = s_local[indexes]
                    s_on_device = s_local
                    noise_channels -= s_local.shape[1]

            pad_height, pad_width = height + 2 * p, width + 2 * p
            key = (lvl, total_samples, total_channels, pad_height, pad_width)
            buf_cache = self.d_noise_buffers if channels_last else self.g_noise_buffers

            buf = buf_cache.get(key)
            if buf is None:
                if channels_last:
                    buf = torch.empty(
                        total_samples,
                        total_channels,
                        pad_height,
                        pad_width,
                        device=device_manager.device,
                        memory_format=torch.channels_last,  # type: ignore
                    ).zero_()
                else:
                    buf = torch.zeros(
                        total_samples,
                        total_channels,
                        pad_height,
                        pad_width,
                        device=device_manager.device,
                    )
                buf_cache[key] = buf

            buf[:, :noise_channels, p : p + height, p : p + width].normal_()

            if w_on_device is not None:
                w_c = w_on_device.shape[1]
                dst = buf[
                    :,
                    noise_channels : noise_channels + w_c,
                    p : p + height,
                    p : p + width,
                ]
                dst.copy_(
                    w_on_device.repeat(steps, 1, 1, 1) if steps > 1 else w_on_device
                )

            if s_on_device is not None:
                s_c = s_on_device.shape[1]
                off = noise_channels + (
                    w_on_device.shape[1] if w_on_device is not None else 0
                )
                dst_s = buf[:, off : off + s_c, p : p + height, p : p + width]
                dst_s.copy_(
                    s_on_device.repeat(steps, 1, 1, 1) if steps > 1 else s_on_device
                )

            result.append(buf)

        return result

    def get_pyramid_noise(
        self,
        scale: int,
        indexes: torch.Tensor | list[int],
        wells_pyramid: dict[int, torch.Tensor] = {},
        seismic_pyramid: dict[int, torch.Tensor] = {},
        rec: bool = False,
    ) -> list[torch.Tensor]:
        """Generate noise tensors up to a specific pyramid scale.

        Parameters
        ----------
        scale : int
        indexes : torch.Tensor | list[int]
        wells_pyramid : dict[int, torch.Tensor], optional
        seismic_pyramid : dict[int, torch.Tensor], optional
        rec : bool, optional
            If True, return stored reconstruction noise.

        Returns
        -------
        list[torch.Tensor]
            Pyramid of noise tensors.
        """
        if rec:
            # Use a relative slice instead of absolute indexes because rec_noise
            # is a rank-local buffer initialized for the current batch samples.
            return [
                n[: len(indexes)].to(device_manager.device)
                for n in self.rec_noise[: scale + 1]
            ]

        if isinstance(indexes, list):
            indexes = torch.tensor(indexes, device=device_manager.device)

        return [
            self.generate_noise(
                i,
                indexes,
                wells_pyramid[i] if wells_pyramid else None,
                seismic_pyramid[i] if seismic_pyramid else None,
            )
            for i in range(scale + 1)
        ]

    def generate_noise(
        self,
        scale: int,
        indexes: torch.Tensor,
        well: torch.Tensor | None = None,
        seismic: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Create a noise tensor for a single pyramid level.

        Applies conditioning (well/seismic) and padding.

        Parameters
        ----------
        scale : int
        indexes : torch.Tensor
        well : torch.Tensor, optional
        seismic : torch.Tensor, optional

        Returns
        -------
        torch.Tensor
            Constructed noise tensor.
        """
        batch = len(indexes)
        spatial_shape = self.get_noise_shape(scale, use_base_channel=False)
        noise_channels = self.gen_input_channels

        parts: list[torch.Tensor] = []

        w_on_device: torch.Tensor | None = None
        if well is not None:
            w_local = well.to(device_manager.device, non_blocking=True)
            if w_local.shape[0] != batch:
                w_local = w_local[indexes]
            w_on_device = w_local
            noise_channels -= w_local.shape[1]

        s_on_device: torch.Tensor | None = None
        if seismic is not None:
            s_local = seismic.to(device_manager.device, non_blocking=True)
            if s_local.shape[0] != batch:
                s_local = s_local[indexes]
            s_on_device = s_local
            noise_channels -= s_local.shape[1]

        parts.append(
            utils.generate_noise((noise_channels, *spatial_shape), num_samp=batch)
        )
        if w_on_device is not None:
            parts.append(w_on_device)
        if s_on_device is not None:
            parts.append(s_on_device)

        z = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        return self.generate_padding(z, value=self.padding_value)

    def generate_fake(self, noises: list[torch.Tensor], scale: int) -> torch.Tensor:
        """Generate a fake sample at the requested `scale` using `noises`.

        Parameters
        ----------
        noises : list[torch.Tensor]
        scale : int

        Returns
        -------
        torch.Tensor
        """
        with torch.no_grad():
            amps = self.get_noise_amplitude(scale)
            fake = self.generator(noises, amps, stop_scale=scale)
        return fake

    def get_synthetic_seismic(
        self, g_out: torch.Tensor, scale: int | None = None
    ) -> torch.Tensor:
        """Compute synthetic seismic for the given generator output.

        Parameters
        ----------
        g_out : torch.Tensor
            Full generator output (facies + rock physics).
        scale : int | None, optional
            Current scale. If None, uses the finest scale.

        Returns
        -------
        torch.Tensor
            Synthetic seismic tensor.
        """
        if scale is None:
            scale = len(self.noise_amps) - 1

        # Extract IP channel
        # ip_norm is at index self.num_facies_channels in g_out
        ip_norm = g_out[:, self.num_facies_channels : self.num_facies_channels + 1, ...]

        from physics.seismic import calculate_synthetic_seismic

        return calculate_synthetic_seismic(
            ip_norm,
            self.physics_state.vp_ref,
            self.physics_state.dz_pyramid[scale],
            self.physics_state,
        )

    def generate_padding(self, z: torch.Tensor, value: float) -> torch.Tensor:
        """Pad tensor `z` using the model's zero-padding size."""
        p = self.zero_padding
        if p > 0:
            padded = torch.full(
                (z.shape[0], z.shape[1], z.shape[2] + 2 * p, z.shape[3] + 2 * p),
                value,
                dtype=z.dtype,
                device=z.device,
            )
            padded[..., p:-p, p:-p] = z
            return padded
        return z

    # ---------------------------------------------------------------------------
    # Scale Lifecycle Management
    # ---------------------------------------------------------------------------

    def init_scales(self, start_scale: int, num_scales: int) -> None:
        """Initialize a consecutive range of scales for training.

        Parameters
        ----------
        start_scale : int
        num_scales : int
        """
        new_scales: set[int] = set()
        for scale in range(start_scale, start_scale + num_scales):
            self.init_generator_for_scale(scale)
            self.init_discriminator_for_scale(scale)
            new_scales.add(scale)
        self.active_scales = new_scales

    def init_generator_for_scale(self, scale: int) -> None:
        """Initialize generator for a new pyramid scale."""
        num_feature, min_num_feature = self.get_num_features(scale)
        self.generator.create_scale(scale, num_feature, min_num_feature)
        prev_is_spade = self.is_spade_scale(scale - 1) if scale > 0 else False
        curr_is_spade = self.is_spade_scale(scale)
        reinit = prev_is_spade or curr_is_spade
        self.finalize_generator_scale(scale, reinit)

    def init_discriminator_for_scale(self, scale: int) -> None:
        """Initialize discriminator for a new pyramid scale."""
        num_feature, min_num_feature = self.get_num_features(scale)
        self.discriminator.create_scale(num_feature, min_num_feature)
        self.finalize_discriminator_scale(scale)

    def finalize_generator_scale(self, scale: int, reinit: bool) -> None:
        """Finalize generator block after creation, applying weights and DDP logic."""
        if reinit:
            self.generator.gens[scale].apply(utils.weights_init)
        else:
            prev = self.generator.gens[scale - 1]
            src_state = prev.state_dict()
            tgt_state = self.generator.gens[scale].state_dict()

            filtered: dict[str, torch.Tensor] = {}
            for k, v in src_state.items():
                if k in tgt_state and v.shape == tgt_state[k].shape:
                    filtered[k] = v

            if filtered:
                self.generator.gens[scale].load_state_dict(filtered, strict=False)
            else:
                self.generator.gens[scale].apply(utils.weights_init)

        self.generator.gens[scale] = self.generator.gens[scale].to(  # type: ignore[call-overload]
            device_manager.device, memory_format=torch.channels_last
        )
        if device_manager.is_distributed:
            for p in self.generator.gens[scale].parameters():
                dist.broadcast(p.data, src=0)
        if self.use_compile and not self.generator.use_gradient_checkpointing:
            # noinspection PyTypeChecker
            self.generator.gens[scale] = torch.compile(  # type: ignore
                self.generator.gens[scale], fullgraph=True, dynamic=True
            )

    def finalize_discriminator_scale(self, scale: int) -> None:
        """Finalize discriminator block after creation."""
        self.discriminator.discs[scale].apply(utils.weights_init)
        self.discriminator.discs[scale] = self.discriminator.discs[scale].to(  # type: ignore[call-overload]
            device_manager.device, memory_format=torch.channels_last
        )
        if device_manager.is_distributed:
            # Broadcast initial weights from rank 0
            for p in self.discriminator.discs[scale].parameters():
                dist.broadcast(p.data, src=0)

        # Keep uncompiled reference for GP computation
        self.uncompiled_discs[scale] = self.discriminator.discs[scale]
        if self.use_compile:
            # noinspection PyTypeChecker
            self.discriminator.discs[scale] = torch.compile(  # type: ignore
                self.discriminator.discs[scale], fullgraph=True, dynamic=False
            )

    def freeze_generator_scales(self, active_scales: tuple[int, ...]) -> None:
        """Freeze generator blocks outside the active training set."""
        active_set = set(active_scales)
        for i, gen in enumerate(self.generator.gens):
            if hasattr(gen, "requires_grad_"):
                gen.requires_grad_(i in active_set)

    def is_spade_scale(self, scale: int) -> bool:
        """Return True if `scale` uses SPADE."""
        return scale in self.generator.spade_scales

    def trim_rec_noise(self, keep_up_to: int) -> None:
        """Move reconstruction noise tensors outside the active range to CPU."""
        for i in range(min(keep_up_to, len(self.rec_noise))):
            t = self.rec_noise[i]
            if t.is_cuda:
                self.rec_noise[i] = t.cpu()

    def clear_stale_generator_grads(self) -> None:
        """Free ``.grad`` tensors on generator parameters outside active scales."""
        for _idx, gen in enumerate(self.generator.gens):
            if not hasattr(gen, "parameters"):
                continue
            for p in gen.parameters():
                if p.grad is not None:
                    p.grad = None

    # ---------------------------------------------------------------------------
    # Distributed Training & Utilities
    # ---------------------------------------------------------------------------

    def setup_framework(self) -> None:
        """Create the Generator and Discriminator instances.

        .. deprecated::
            This method is kept for backwards compatibility only.
            The framework objects are now initialized directly in ``__init__``.
        """

    def get_noise_amplitude(self, scale: int) -> list[torch.Tensor]:
        """Return noise amplitude list up to a given scale."""
        target_len = scale + 1
        if len(self.noise_amps) > 0:
            amps = list(self.noise_amps[:target_len])
            if len(amps) < target_len:
                ref = amps[-1]
                one = torch.ones_like(ref)
                amps.extend(one.clone() for _ in range(target_len - len(amps)))
            return amps
        return [
            torch.tensor(1.0, device=device_manager.device) for _ in range(target_len)
        ]

    def get_num_features(self, scale: int) -> tuple[int, int]:
        """Calculate feature counts for networks at a given scale."""
        num_feature = min(self.options.num_feature * pow(2, math.floor(scale / 4)), 128)
        min_num_feature = min(
            self.options.min_num_feature * pow(2, math.floor(scale / 4)), 128
        )
        return num_feature, min_num_feature

    def get_noise_shape(
        self, scale: int, use_base_channel: bool = True
    ) -> tuple[int, ...]:
        """Return the noise shape tuple for a given scale.

        Parameters
        ----------
        scale : int
            Scale index to query.
        use_base_channel : bool, optional
            When True (default) include ``self.base_channel`` as the first
            element of the returned tuple (channel dimension). When False only
            the spatial dimensions from ``self.shapes[scale]`` are returned.

        Returns
        -------
        tuple[int, ...]
            A tuple describing the noise tensor shape for the requested scale.
            When ``use_base_channel`` is True the shape is
            ``(channels, height, width)``; otherwise only ``(height, width)``
            (or the corresponding spatial dims) are returned.
        """
        return (
            (self.base_channel, *self.shapes[scale][2:])
            if use_base_channel
            else self.shapes[scale][2:]
        )

    @staticmethod
    def optimizer_zero_grad(optimizer: torch.optim.Optimizer) -> None:
        """Clear gradients using set_to_none when optimizer API supports it."""
        try:
            optimizer.zero_grad(set_to_none=True)
        except TypeError:
            optimizer.zero_grad()

    @staticmethod
    def all_reduce_grads_coalesced(
        modules: list[nn.Module], async_op: bool = False
    ) -> dist.Work | None:
        """Average gradients across all DDP ranks with a **single** all-reduce.

        Parameters
        ----------
        modules : list[nn.Module]
        async_op : bool, optional

        Returns
        -------
        dist.Work | None
        """
        params: list[nn.Parameter] = []
        for m in modules:
            params.extend(p for p in m.parameters() if p.requires_grad)
        if not params:
            return None
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p.data)
        grads = [p.grad for p in params]

        # noinspection PyProtectedMember
        flat = cast(
            torch.Tensor,
            torch._utils._flatten_dense_tensors(grads),  # type: ignore
        )

        work: dist.Work | None = dist.all_reduce(flat, op=dist.ReduceOp.AVG, async_op=async_op)  # type: ignore

        if async_op:
            return work  # type: ignore

        # Scatter back
        # noinspection PyProtectedMember
        for g, synced in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):  # type: ignore
            if g is not None:
                g.copy_(synced)  # type: ignore

        return None

    def all_reduce_grads(self, module: nn.Module) -> dist.Work | None:
        """Average gradients of a single module across DDP ranks."""
        return self.all_reduce_grads_coalesced([module])

    def wait_pending_all_reduce(self) -> None:
        """Wait for any pending async all-reduce operation to complete."""
        if self.pending_all_reduce_work is not None:
            self.pending_all_reduce_work.wait()

            self.pending_all_reduce_work = None

    def report_all_reduce_profile(self) -> None:
        """Print aggregate timing for coalesced gradient sync profiling."""
        if not self.profile_all_reduce or self.profile_all_reduce_calls == 0:
            return
        avg_s = self.profile_all_reduce_total_s / self.profile_all_reduce_calls
        avg_collective_s = (
            self.profile_collective_total_s / self.profile_all_reduce_calls
        )
        avg_elems = self.profile_all_reduce_total_elems / self.profile_all_reduce_calls
        print(f"\n[ALLREDUCE_PROFILE] coalesced gradient sync summary")
        print(
            f"[ALLREDUCE_PROFILE] mode=ddp_all_reduce profiling=detailed calls={self.profile_all_reduce_calls} "
            f"avg_total_time={avg_s:.6f}s avg_collective_time={avg_collective_s:.6f}s avg_elems={avg_elems:.0f} "
            f"total_time={self.profile_all_reduce_total_s:.6f}s collective_time={self.profile_collective_total_s:.6f}s"
        )

    # ---------------------------------------------------------------------------
    # Serialization (I/O)
    # ---------------------------------------------------------------------------

    def load(
        self,
        path: str,
        load_shapes: bool = True,
        until_scale: int | None = None,
        load_discriminator: bool = False,
        load_wells: bool = False,
    ) -> int:
        """Load saved models and metadata from a checkpoint directory.

        Parameters
        ----------
        path : str
        load_shapes : bool, optional
        until_scale : int | None, optional
        load_discriminator : bool, optional
        load_wells : bool, optional

        Returns
        -------
        int
            The next scale index to train.
        """
        scale = 0

        # If loading for inference/generation (no discriminator), adjust the
        # compile progress total so the progress bar correctly reflects 100%
        # completion after the generator and helpers are triggered.
        if not load_discriminator and self._compile_progress_enabled:
            planned_scales = int(getattr(self.options, "stop_scale", 0)) + 1
            self._compile_progress_total -= planned_scales
        while os.path.exists(os.path.join(path, str(scale))):
            if until_scale is not None and scale > until_scale:
                break

            scale_path = os.path.join(path, str(scale))
            ckpt_path = os.path.join(scale_path, CheckpointFilenames.EPOCH_CKPT)

            if os.path.isfile(ckpt_path):
                # Load from structured epoch checkpoint
                from training.checkpoint import Checkpoint

                ckpt = Checkpoint.load(ckpt_path)

                # Restore global training state from the monolithic checkpoint
                if ckpt.noise_amps:
                    self.noise_amps = ckpt.noise_amps
                if ckpt.rec_noise:
                    self.rec_noise = ckpt.rec_noise
                self.disc_step_counter = ckpt.disc_step_counter
                self.extra_disc_step_counter = ckpt.extra_disc_step_counter
                self.current_epoch = ckpt.epoch

                for s, sd in ckpt.scales.items():
                    if until_scale is not None and s > until_scale:
                        continue

                    if s >= len(self.generator.gens):
                        self.init_generator_for_scale(s)
                    self.load_state_dict_compat(
                        unwrap_ddp(self.generator.gens[s]), sd.generator
                    )

                    if load_discriminator:
                        if s >= len(self.discriminator.discs):
                            self.init_discriminator_for_scale(s)
                        self.load_state_dict_compat(
                            unwrap_ddp(self.discriminator.discs[s]), sd.discriminator
                        )

                # Skip ahead if this checkpoint already covered multiple scales
                if ckpt.scales:
                    scale = max(ckpt.scales.keys()) + 1
                    continue
            else:
                # Fallback to legacy individual file loading
                if self.has_generator_checkpoint(scale_path):
                    self.init_generator_for_scale(scale)
                    self.load_generator_state(scale_path, scale)

                if load_discriminator and self.has_discriminator_checkpoint(scale_path):
                    self.init_discriminator_for_scale(scale)
                    self.load_discriminator_state(scale_path, scale)

            if self.has_amp_file(scale_path):
                self.load_amp(scale_path)

            if load_shapes and self.has_shape_file(scale_path):
                self.load_shape(scale_path)

            if load_wells and self.has_wells_file(scale_path):
                self.load_wells(scale_path)

            scale += 1

        return scale

    def save_scale(self, scale: int, path: str) -> None:
        """Save all model states and metadata for a specific scale."""
        os.makedirs(path, exist_ok=True)
        self.save_generator_state(path, scale)
        self.save_discriminator_state(path, scale)
        self.save_amp(path, scale)
        self.save_shape(path, scale)

    def load_generator_state(self, scale_path: str, scale: int) -> None:
        """Load generator state dict for a scale."""
        gen_path = os.path.join(scale_path, CheckpointFilenames.GENERATOR)
        if os.path.exists(gen_path):
            state = torch.load(
                gen_path, map_location=device_manager.device, weights_only=False
            )
            self.load_state_dict_compat(unwrap_ddp(self.generator.gens[scale]), state)  # type: ignore

    def load_discriminator_state(self, scale_path: str, scale: int) -> None:
        """Load discriminator state dict for a scale."""
        disc_path = os.path.join(scale_path, CheckpointFilenames.DISCRIMINATOR)
        if os.path.exists(disc_path):
            state = torch.load(
                disc_path, map_location=device_manager.device, weights_only=False
            )
            self.load_state_dict_compat(  # type: ignore
                unwrap_ddp(self.discriminator.discs[scale]), state
            )

    def load_state_dict_compat(
        self, target: nn.Module, state: dict[str, torch.Tensor]
    ) -> None:
        """Load state dict handling torch.compile ``_orig_mod.`` prefix mismatches."""
        prefix = "_orig_mod."
        model_keys = set(target.state_dict().keys())
        ck_keys = set(state.keys())
        ck_has_prefix = any(k.startswith(prefix) for k in ck_keys)
        mod_has_prefix = any(k.startswith(prefix) for k in model_keys)
        if ck_has_prefix and not mod_has_prefix:
            state = {
                (k[len(prefix) :] if k.startswith(prefix) else k): v
                for k, v in state.items()
            }
        elif mod_has_prefix and not ck_has_prefix:
            state = {f"{prefix}{k}": v for k, v in state.items()}
        target.load_state_dict(state)

    def save_generator_state(self, scale_path: str, scale: int) -> None:
        """Save generator state dict for a scale."""
        if scale < len(self.generator.gens):
            torch.save(
                unwrap_ddp(self.generator.gens[scale]).state_dict(),
                os.path.join(scale_path, CheckpointFilenames.GENERATOR),
            )

    def save_discriminator_state(self, scale_path: str, scale: int) -> None:
        """Save discriminator state dict for a scale."""
        if scale < len(self.discriminator.discs):
            torch.save(
                unwrap_ddp(self.discriminator.discs[scale]).state_dict(),
                os.path.join(scale_path, CheckpointFilenames.DISCRIMINATOR),
            )

    def load_amp(self, scale_path: str) -> None:
        """Load noise amplitude from file."""
        amp_path = os.path.join(scale_path, CheckpointFilenames.NOISE_AMP)
        if os.path.exists(amp_path):
            with open(amp_path, "r") as f:
                self.noise_amps.append(
                    torch.tensor(float(f.read().strip()), device=device_manager.device)
                )

    def save_amp(self, scale_path: str, scale: int) -> None:
        """Save noise amplitude to file."""
        if scale < len(self.noise_amps):
            amp_path = os.path.join(scale_path, CheckpointFilenames.NOISE_AMP)
            with open(amp_path, "w") as f:
                f.write(str(float(self.noise_amps[scale])))

    def load_shape(self, scale_path: str) -> None:
        """Load shape metadata for a scale."""
        shape_path = os.path.join(scale_path, CheckpointFilenames.SHAPE)
        if os.path.exists(shape_path):
            self.shapes += tuple(
                torch.load(
                    shape_path, map_location=device_manager.device, weights_only=False
                )
            )

    def save_shape(self, scale_path: str, scale: int) -> None:
        """Save shape metadata for a scale."""
        if scale < len(self.shapes):
            torch.save(
                self.shapes[scale], os.path.join(scale_path, CheckpointFilenames.SHAPE)
            )

    def load_wells(self, scale_path: str) -> None:
        """Load well conditioning data for a scale."""
        loaded = utils.load(os.path.join(scale_path, CheckpointFilenames.MASKS))
        wells = [
            (
                loaded
                if isinstance(loaded, torch.Tensor)
                else torch.as_tensor(loaded, device=device_manager.device)
            )
        ]
        # noinspection PyAttributeOutsideInit
        self.wells = tuple(wells)

    @staticmethod
    def has_generator_checkpoint(scale_path: str) -> bool:
        """Return True if generator checkpoint exists."""
        return os.path.exists(os.path.join(scale_path, CheckpointFilenames.GENERATOR))

    @staticmethod
    def has_discriminator_checkpoint(scale_path: str) -> bool:
        """Return True if discriminator checkpoint exists."""
        return os.path.exists(
            os.path.join(scale_path, CheckpointFilenames.DISCRIMINATOR)
        )

    @staticmethod
    def has_amp_file(scale_path: str) -> bool:
        """Return True if amplitude file exists."""
        return os.path.exists(os.path.join(scale_path, CheckpointFilenames.NOISE_AMP))

    @staticmethod
    def has_shape_file(scale_path: str) -> bool:
        """Return True if shape file exists."""
        return os.path.exists(os.path.join(scale_path, CheckpointFilenames.SHAPE))

    @staticmethod
    def has_wells_file(scale_path: str) -> bool:
        """Return True if wells file exists."""
        return os.path.exists(os.path.join(scale_path, CheckpointFilenames.MASKS))
