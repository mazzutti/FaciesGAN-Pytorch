"""Parallel FACIESGAN implementation for training multiple scales simultaneously.

This module extends the standard FaciesGAN to support parallel training of
multiple pyramid scales. Instead of training scales sequentially, this
implementation can train multiple scales at once using separate optimizers
and discriminators for each scale.
"""

import math
import os
import time
from typing import Any, cast

import torch
import torch._dynamo
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

from datasets.utils import get_global_stats
from physics.seismic import torch_ricker_wavelet

from .utils import (
    calculate_physics_loss,
    dice_loss,
    masked_cross_entropy,
    total_variation_loss,
)

# Raise the per-code-object recompile cache.
#
# Gen blocks are compiled with dynamic=True so spatial dims are symbolic
# (no per-shape specialization).  The remaining specialization axes are:
#   - is_inference_mode (bool)     — 2 variants
#   - is_grad_enabled (bool)       — 2 variants (subset of above, 3 total modes)
#   - input requires_grad (bool)   — 2 variants per mode
#
# With 7 gen blocks × up to ~4 specializations each, plus facies_quantizer
# and _residual_clamp, the code-object cache may accumulate ~30-40 entries
# across all instances sharing the same forward __code__.  64 provides
# comfortable headroom; raising it prevents FailOnRecompileLimitHit
# (fullgraph=True treats the limit as a hard error instead of fallback).
torch._dynamo.config.cache_size_limit = 64  # type: ignore[attr-defined]

from config import AMP_FILE, D_FILE, G_FILE, M_FILE, SHAPE_FILE
from metrics import DiscriminatorMetrics, GeneratorMetrics, ScaleMetrics
from models.base import FaciesGAN, IterableMetrics
from options import TrainingOptions

from . import utils
from .discriminator import Discriminator
from .generator import Generator


def unwrap_ddp(module: nn.Module) -> nn.Module:
    """Return the inner module if wrapped in ``DistributedDataParallel``."""
    return getattr(module, "module", module)  # type: ignore[no-any-return]


class TorchFaciesGAN(
    FaciesGAN,
    nn.Module,
):
    """PyTorch implementation of the FaciesGAN architecture.

    This class manages the lifecycle of Generators and Discriminators,
    initializes them, and provides helpers for the training loop.
    Unlike PyTorch, we don't inherit from a base class with a strict
    call graph here, but rather provide the necessary functional hooks.
    """

    def __init__(
        self,
        options: TrainingOptions,
        device: torch.device = torch.device("cpu"),
        noise_channels: int = 4,
        use_ddp: bool = False,
        *args: tuple[Any, ...],
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize the parallel FaciesGAN model.

        Parameters
        ----------
        device : torch.device
            Primary device for computation.
        options : TrainingOptions
            Training configuration containing hyperparameters.
        noise_channels : int, optional
            Number of input noise channels, by default 3.
        use_ddp : bool, optional
            When ``True``, each per-scale sub-module is wrapped with
            ``DistributedDataParallel`` after creation.  Requires that
            ``torch.distributed`` has been initialised before training
            starts.  Defaults to ``False``.
        """
        nn.Module.__init__(self)  # type: ignore
        # Initialize base class attributes
        super().__init__(options, noise_channels, *args, **kwargs)

        self.zero_padding = int(options.num_layer * math.floor(options.kernel_size / 2))

        # Framework-specific attributes
        self.device = device
        self.stop_scale = options.stop_scale

        # Multi-GPU setup via manual gradient all-reduce.
        self.use_ddp = use_ddp

        # AMP (Automatic Mixed Precision) for faster CUDA training.
        # The generator always uses AMP.  The discriminator uses AMP
        # only on non-GP steps (7 of 8 with gp_interval=8); GP steps
        # stay fp32 because autograd.grad(create_graph=True) is
        # incompatible with AMP autocast.
        self._use_amp = device.type == "cuda"
        amp_dtype_opt = str(getattr(options, "amp_dtype", "fp16")).lower()
        self._amp_dtype = torch.bfloat16 if amp_dtype_opt == "bf16" else torch.float16
        # Grad scaling is only needed for fp16; bf16 is numerically robust.
        self._use_grad_scaler = self._use_amp and self._amp_dtype == torch.float16
        self._grad_scaler_g = GradScaler(enabled=self._use_grad_scaler)

        # torch.compile gives a meaningful speedup on CUDA when gradient
        # checkpointing is OFF (the two features are incompatible because
        # compiled graphs reorder saved tensors, breaking checkpoint
        # recomputation metadata checks).  Enabled by default on CUDA;
        # pass ``--no-compile`` to disable.
        self._use_compile = device.type == "cuda" and getattr(
            options, "compile_backend", True
        )

        # Pre-allocate constant zero scalars on device so the hot path
        # avoids repeated small CUDA allocations.
        self._zero_scalar = torch.tensor(0.0, device=device)

        # Gradient (activation) checkpointing trades compute for memory.
        self._use_gradient_checkpointing = getattr(
            options, "gradient_checkpointing", False
        )

        # Mapping from scale → original (uncompiled) disc module used
        # exclusively for gradient-penalty computation that requires
        # ``create_graph=True`` (incompatible with compiled graphs).
        self._uncompiled_discs: dict[int, nn.Module] = {}

        # NVLink optimization: track pending async all-reduce work for overlap.
        # Set to a dist.Work object when an async all-reduce is initiated;
        # cleared by _wait_pending_allreduce() before the next step.
        self._pending_allreduce_work: dist.Work | None = None

        # Deferred disc gradient sync for D/G overlap.
        # On the last D-step, the all-reduce is launched asynchronously so
        # that NCCL communication overlaps with the generator forward pass
        # at the start of the G-phase.  Completed inside
        # compute_generator_metrics before the disc is evaluated.
        self._pending_disc_ar_work: dist.Work | None = None
        self._pending_disc_ar_flat: torch.Tensor | None = None
        self._pending_disc_ar_grads: list[torch.Tensor] | None = None

        # Load stats for Elastic Consistency Loss denormalization
        self.stats = get_global_stats(options.input_path)

        # Pre-compute tensors for fast denormalization on GPU
        self.phys_min: dict[str, torch.Tensor] = {}
        self.phys_diff: dict[str, torch.Tensor] = {}
        from datasets.data_files import DataFiles

        for comp in DataFiles.all_rock_physics() + [DataFiles.SEISMIC]:
            key = comp.name
            s = self.stats[key]
            # Note: VP/VS in stats.json are in Km/s, convert to m/s
            scale = 1000.0 if comp in [DataFiles.VP, DataFiles.VS] else 1.0
            self.phys_min[key] = torch.tensor(
                s["min"] * scale, device=device, dtype=torch.float32
            )
            self.phys_diff[key] = torch.tensor(
                (s["max"] - s["min"]) * scale, device=device, dtype=torch.float32
            )

        # Populate pyramid shapes once so they can be used for pre-calculating constants
        from datasets.utils import generate_scales

        self.shapes = list(generate_scales(options))

        # Register physical stats as buffers for fast access in physics loss
        # These will stay on the GPU and move with the model
        ip_min_t = self.phys_min[DataFiles.Ip.name]
        self.register_buffer("ip_min", ip_min_t)
        self.register_buffer("ip_max", ip_min_t + self.phys_diff[DataFiles.Ip.name])

        self.register_buffer("vp_min", self.phys_min[DataFiles.VP.name])
        # Add vp_max from stats (scaled from Km/s to m/s)
        self.register_buffer(
            "vp_max",
            torch.tensor(
                self.stats["VP"]["max"] * 1000.0, device=device, dtype=torch.float32
            ),
        )

        seis_min_t = self.phys_min[DataFiles.SEISMIC.name]
        self.register_buffer("seis_min", seis_min_t)
        self.register_buffer(
            "seis_max", seis_min_t + self.phys_diff[DataFiles.SEISMIC.name]
        )

        # Register rho_mean as a buffer for fast access in physics loss
        rho_mean_val = float(self.stats["RHO"]["mean"])
        self.register_buffer(
            "rho_mean", torch.tensor(rho_mean_val, device=device, dtype=torch.float32)
        )

        # Pre-calculate dz for every scale to avoid redundant float math in G-loop
        dz_values: list[float] = []
        h_target = float(self.shapes[options.stop_scale][2])
        for s in range(len(self.shapes)):
            h_scale = float(self.shapes[s][2])
            ratio = h_target / h_scale
            dz_values.append(options.dz_pixel * ratio)
        self.register_buffer(
            "dz_pyramid", torch.tensor(dz_values, device=device, dtype=torch.float32)
        )

        # Reference velocity for Wavelet Resampling (avoids per-batch sync)
        ip_mean = float(self.stats["Ip"]["mean"])
        self.vp_ref = ip_mean / rho_mean_val

        self._pending_disc_ar_opts: dict[int, torch.optim.Optimizer] | None = None
        self._pending_disc_ar_scales: list[int] | None = None

        # 2. Pre-generate time-domain wavelet for Physics Loss
        self.register_buffer(
            "wavelet_dt",
            torch.tensor(options.wavelet_dt, device=device, dtype=torch.float32),
        )

        self.wavelet_t = torch_ricker_wavelet(
            options.wavelet_f_peak,
            options.wavelet_dt,
            options.wavelet_length,
            device=device,
        )

        self._profile_allreduce = os.environ.get("FG_PROFILE_ALLREDUCE", "0") == "1"
        self._profile_allreduce_total_s = 0.0
        self._profile_collective_total_s = 0.0
        self._profile_allreduce_calls = 0
        self._profile_allreduce_total_elems = 0

        self._current_epoch: int = 0
        # Recovery loss is critical from epoch 0: it anchors the generator
        # to the facies_rec mode and provides the main training signal
        # once the scale's discriminator has stabilized.
        self._facies_rec_skip_epochs: int = 0
        # Diversity loss is active from epoch 0 to match the original
        # training behavior.  Setting to 0 disables the warmup window.
        self._div_skip_epochs: int = 0

        self._noise_manager = utils.NoiseBufferManager(self.device)

        # Create framework objects via the base class helper (calls build_* hooks)
        self.setup_framework()

        # Propagate the checkpointing flag to the generator after it is
        # constructed (setup_framework calls build_generator).
        if self._use_gradient_checkpointing:
            self.generator.use_gradient_checkpointing = True

        # Compile the one-hot quantizer — it runs on every gen forward
        # (~49 times per iteration) and has a simple compute graph
        # (einsum + softmax) that benefits from operator fusion.
        # dynamic=True: pyramid levels have different H/W; symbolic shapes
        # prevent a new Triton kernel per spatial size.
        if self._use_compile:
            gen = self.generator
            gen.facies_quantizer = torch.compile(  # type: ignore[assignment]
                gen.facies_quantizer,
                fullgraph=True,
                dynamic=True,
                mode="default",
            )

        # Compile the residual-add + clamp helper so Inductor fuses
        # them into a single pointwise kernel, saving one kernel launch
        # per scale per generator forward pass.
        # dynamic=True: called for every pyramid level (7 different H/W).
        if self._use_compile:
            gen = self.generator
            gen._residual_clamp = torch.compile(  # type: ignore[assignment]
                Generator.residual_clamp_fn,
                fullgraph=True,
                dynamic=True,
            )

    # ── GPU-resident loss scale factors ─────────────────────────
    # Override the base-class EMA helpers so that scale factors stay
    # as CUDA scalar tensors.  This eliminates 21 GPU→CPU .item()
    # sync stalls per iteration (D-steps × scales) and lets the
    # DDP sync build its all-reduce tensor without a round-trip.

    def update_loss_scale_factor(self, scale: int, d_mag: float | torch.Tensor) -> None:  # type: ignore[override]
        """EMA update keeping values as device-resident scalar tensors.

        The 1e-4 lower bound is enforced here at write time so that
        ``get_loss_scale_factor`` can return the stored tensor directly
        without an extra ``torch.clamp`` call on every read (which would
        allocate a new CUDA scalar tensor each time).
        """
        if not isinstance(d_mag, torch.Tensor):
            d_mag = torch.tensor(d_mag, device=self.device)
        if scale not in self.loss_scale_factors:
            self.loss_scale_factors[scale] = (  # type: ignore[assignment]
                d_mag.detach().clone().clamp_(min=1e-4)
                if d_mag > 0
                else torch.tensor(1e-4, device=self.device)
            )
        else:
            prev = self.loss_scale_factors[scale]
            if not isinstance(prev, torch.Tensor):
                prev = torch.tensor(prev, device=self.device)
            decay = self.loss_scale_ema_decay
            self.loss_scale_factors[scale] = (  # type: ignore[assignment]
                (decay * prev + (1 - decay) * d_mag).clamp_(min=1e-4)
            ).detach()

    def get_loss_scale_factor(self, scale: int) -> float | torch.Tensor:  # type: ignore[override]
        """Return the current GPU-resident scale factor (no sync).

        The 1e-4 minimum is already enforced by ``update_loss_scale_factor``
        so no additional ``torch.clamp`` allocation is needed here.
        """
        sf = self.loss_scale_factors.get(scale, 1.0)
        if isinstance(sf, torch.Tensor):
            return sf  # already clamped to >= 1e-4 on every write
        return max(sf, 1e-4)  # fallback before first update (float 1.0)

    def __call__(self, *args: Any, **kwds: Any) -> ScaleMetrics:
        return nn.Module.__call__(self, *args, **kwds)

    def device_for_scale(self, scale: int) -> torch.device:
        """Return the primary device (all modules live there).

        With DataParallel the modules are replicated at forward time;
        their parameters always reside on ``self.device``.

        Parameters
        ----------
        scale : int
            Pyramid scale index (unused — kept for API compatibility).

        Returns
        -------
        torch.device
            The primary CUDA device.
        """
        return self.device

    def build_discriminator(self) -> Discriminator:
        """Build and return the PyTorch `Discriminator` instance (not moved).

        Returns:
            Discriminator: Newly constructed discriminator instance.
        """
        return Discriminator(
            self.num_layer,
            self.kernel_size,
            self.padding_size,
            self.disc_input_channels,
        ).to(self.device)

    def build_generator(self) -> Generator:
        """Build and return the PyTorch `Generator` instance (not moved).

        Returns:
            Generator: Newly constructed generator instance.
        """
        gen = Generator(
            self.num_layer,
            self.kernel_size,
            self.padding_size,
            self.gen_input_channels,
            self.gen_output_channels,
            num_facies_classes=getattr(
                self, "num_facies_classes", self.gen_output_channels
            ),
            noise_channels=self.num_noise_channels,
        )
        # apply color quantization only to the facies channels when
        # rock_physics channels are appended to the output.
        return gen.to(self.device)

    def compute_discriminator_metrics(
        self,
        indexes: torch.Tensor,
        scale: int,
        real: torch.Tensor,
        wells_pyramid: dict[int, torch.Tensor] = {},
        seismic_pyramid: dict[int, torch.Tensor] = {},
    ) -> tuple[DiscriminatorMetrics, dict[str, Any] | None]:
        """Compute discriminator losses and gradient penalty for a scale.

        Parameters
        ----------
        indexes (tuple[int, ...]):
            Batch/sample indices used to generate fake inputs.
        scale (int):
            Pyramid scale index for which to compute the metrics.
        real_facies (torch.Tensor):
            Ground-truth tensor for the current scale.
        wells_pyramid (dict[int, torch.Tensor], optional):
            Wells tensors dict for conditioning, keyed by scale.
        seismic_pyramid (dict[int, torch.Tensor], optional):
            Seismic  tensors dict for conditioning, keyed by scale.

        Returns
        -------
        tuple[DiscriminatorMetrics, dict[Any, Any] | None]:
            Container with total, real, fake and gp losses, and optional gradients dict.
        """

        d_real = self.discriminator(scale, real.to(self.device))
        noises = self.get_pyramid_noise(scale, indexes, wells_pyramid, seismic_pyramid)
        fake = self.generate_fake(noises, scale)
        d_fake = self.discriminator(scale, fake.detach())  # type: ignore

        # WGAN-GP losses.
        real_loss = -d_real.mean()
        fake_loss = d_fake.mean()
        gp = self.compute_gradient_penalty(scale, real, fake.detach())

        total = real_loss + fake_loss + gp
        return (
            DiscriminatorMetrics(
                total=total,
                real=real_loss.detach(),
                fake=fake_loss.detach(),
                gp=gp.detach(),
            ),
            None,
        )

    def _get_batched_d_noise(
        self,
        scale: int,
        D: int,
        B: int,
        indexes: torch.Tensor,
        wells_pyramid: dict[int, torch.Tensor],
        seismic_pyramid: dict[int, torch.Tensor],
    ) -> list[torch.Tensor]:
        """Generate D×B noise per pyramid level using pre-allocated buffers.

        Instead of D separate ``get_pyramid_noise`` calls followed by
        ``torch.cat`` per level, this method reuses cached zero-padded
        buffers and fills them in-place with ``.normal_()`` / ``.copy_()``.
        This eliminates D×(scale+1) ``torch.randn`` + ``F.pad`` +
        ``torch.cat`` allocations per scale on every iteration.

        The returned buffers are owned by ``self._d_noise_bufs`` and will
        be overwritten on the next call, which is safe because the
        generator forward pass only reads from them (all arithmetic is
        out-of-place).
        """
        DB = D * B
        p = self.zero_padding
        result: list[torch.Tensor] = []

        for lvl in range(scale + 1):
            spatial = self.get_noise_shape(lvl, use_base_channel=False)
            H, W = spatial[0], spatial[1]
            total_C = self.gen_input_channels
            noise_C = total_C

            w: torch.Tensor | None = None
            s: torch.Tensor | None = None

            if wells_pyramid:
                w = wells_pyramid[lvl].to(self.device, non_blocking=True)
                if w.shape[0] != B:
                    w = w[indexes]
                noise_C -= w.shape[1]

            if seismic_pyramid:
                s = seismic_pyramid[lvl].to(self.device, non_blocking=True)
                if s.shape[0] != B:
                    s = s[indexes]
                noise_C -= s.shape[1]

            padH, padW = H + 2 * p, W + 2 * p
            key = (lvl, DB, total_C, padH, padW)

            buf = self._noise_manager.get_buffer(key, (DB, total_C, padH, padW))

            # Fill noise channels in the inner (unpadded) region.
            buf[:, :noise_C, p : p + H, p : p + W].normal_()

            # Copy conditioning into each D-chunk with a single broadcast copy
            # instead of a Python for-loop, reducing D serial CUDA launches to 1.
            if w is not None:
                wC = w.shape[1]
                dst = buf[:, noise_C : noise_C + wC, p : p + H, p : p + W]
                dst.copy_(w.repeat(D, 1, 1, 1))

            if s is not None:
                sC = s.shape[1]
                off = noise_C + (w.shape[1] if w is not None else 0)
                dst_s = buf[:, off : off + sC, p : p + H, p : p + W]
                dst_s.copy_(s.repeat(D, 1, 1, 1))

            result.append(buf)

        return result

    def _get_batched_g_noise(
        self,
        N: int,
        B: int,
        scale: int,
        wells_pyramid: dict[int, torch.Tensor],
        seismic_pyramid: dict[int, torch.Tensor],
    ) -> list[torch.Tensor]:
        """Pre-allocated noise buffers for the G-phase — analogous to
        ``_get_batched_d_noise`` but for ``N`` diversity samples (N=1 for the
        standard path, N=num_diversity_samples for the batched diversity path).

        Reusing the same buffer across G-steps is safe because the noise
        tensors are leaf tensors with ``requires_grad=False``. They are used
        read-only during the generator forward; ``backward()`` does not
        traverse back into them, so the buffer can be overwritten as soon
        as the previous step's ``backward()`` has returned.
        """
        NB = N * B
        p = self.zero_padding
        result: list[torch.Tensor] = []

        for lvl in range(scale + 1):
            spatial = self.get_noise_shape(lvl, use_base_channel=False)
            H, W = spatial[0], spatial[1]
            total_C = self.gen_input_channels
            noise_C = total_C

            w: torch.Tensor | None = None
            s: torch.Tensor | None = None

            if wells_pyramid:
                w = wells_pyramid[lvl]
                noise_C -= w.shape[1]

            if seismic_pyramid:
                s = seismic_pyramid[lvl]
                noise_C -= s.shape[1]

            padH, padW = H + 2 * p, W + 2 * p
            key = (lvl, NB, total_C, padH, padW)

            buf = self._noise_manager.get_buffer(key, (NB, total_C, padH, padW))

            # Fill noise channels in-place (eliminates randn + pad allocations).
            buf[:, :noise_C, p : p + H, p : p + W].normal_()

            # Broadcast conditioning to all N samples with a single copy.
            if w is not None:
                wC = w.shape[1]
                dst = buf[:, noise_C : noise_C + wC, p : p + H, p : p + W]
                dst.copy_(w.repeat(N, 1, 1, 1) if N > 1 else w)

            if s is not None:
                sC = s.shape[1]
                off = noise_C + (w.shape[1] if w is not None else 0)
                dst_s = buf[:, off : off + sC, p : p + H, p : p + W]
                dst_s.copy_(s.repeat(N, 1, 1, 1) if N > 1 else s)

            result.append(buf)

        return result

    def generate_diverse_samples(
        self,
        indexes: torch.Tensor,
        scale: int,
        wells_pyramid: dict[int, torch.Tensor] = {},
        seismic_pyramid: dict[int, torch.Tensor] = {},
    ) -> list[torch.Tensor]:
        """Override base implementation to use pre-allocated G-noise buffers.

        Eliminates ``G × (scale+1)`` ``torch.randn + F.pad`` allocations per
        training iteration by filling pre-allocated ``_g_noise_bufs`` in-place.
        For ``N > 1`` (diversity), a single batch of size ``N*B`` is run
        through the generator instead of N separate forwards, matching the
        existing base-class batching strategy but without the intermediate
        per-level allocation overhead.
        """
        div_skip = getattr(self, "_div_skip_epochs", 0)
        cur_epoch = getattr(self, "_current_epoch", 0)
        N = 1 if cur_epoch < div_skip else self.num_diversity_samples
        B = len(indexes)

        batched_noises = self._get_batched_g_noise(
            N, B, scale, wells_pyramid, seismic_pyramid
        )
        amps = self.get_noise_amplitude(scale)

        if N <= 1:
            return [self.generator(batched_noises, amps, stop_scale=scale)]

        batched_out = self.generator(batched_noises, amps, stop_scale=scale)
        return list(batched_out.split(B, dim=0))  # type: ignore[arg-type]

    def optimize_discriminator(
        self,
        indexes: torch.Tensor,
        optimizers: dict[int, torch.optim.Optimizer],
        facies_pyramid: dict[int, torch.Tensor],
        wells_pyramid: dict[int, torch.Tensor] = {},
        seismic_pyramid: dict[int, torch.Tensor] = {},
    ) -> tuple[DiscriminatorMetrics, ...]:
        """Discriminator optimization with gradient accumulation.

        All D fakes per scale are generated in a single batched generator
        forward (batch = D × B) before the D-step loop begins.  This is
        safe because generator weights are frozen during D optimisation.

        Gradients are accumulated across all D forward-backward passes
        (each scaled by 1/D) with a single all-reduce + optimizer step
        at the end, reducing NCCL collectives from D to 1 per iteration.

        Lazy gradient penalty (``gp_interval``) amortises the expensive
        ``create_graph=True`` double backward.

        Returns
        -------
        tuple[DiscriminatorMetrics, ...]
            Metrics from the last discriminator step for each active scale.
        """

        D = self.discriminator_steps
        if D <= 0:
            return ()

        sorted_scales = sorted(self.active_scales)
        B = len(indexes)

        # ── Pre-generate all D fakes per scale in one batched forward ──
        # Generator weights are frozen during D optimisation, so all
        # fakes can be produced upfront.  One forward with batch = D*B
        # is cheaper than D separate forwards with batch = B (fewer
        # kernel launches).
        # Noise buffers are pre-allocated and reused across iterations
        # to eliminate D×(scale+1) randn + F.pad + cat allocations.
        scale0_multi = self.scale0_disc_steps_multiplier
        prefaked: dict[int, list[torch.Tensor]] = {}
        with torch.no_grad():
            for scale in sorted_scales:
                d_count = D * scale0_multi if scale == 0 else D
                batched_noises = self._get_batched_d_noise(
                    scale, d_count, B, indexes, wells_pyramid, seismic_pyramid
                )
                amps = self.get_noise_amplitude(scale)
                batched_fake = self.generator(batched_noises, amps, stop_scale=scale)
                # Split back into d_count chunks of size B.
                prefaked[scale] = list(batched_fake.split(B, dim=0))  # type: ignore[arg-type]

        # ── D-step loop: each step gets its own zero_grad → backward →
        # all-reduce → optimizer.step() cycle.  This is correct for
        # Adam: D separate parameter updates produce different dynamics
        # than 1 update with accumulated gradients (Adam's moment
        # estimates are updated D times, not once).
        step_metrics: list[DiscriminatorMetrics] = []
        # Track the last GP value computed per scale across all D-steps
        # so it can be reported even when the final step skips GP.
        last_gp: dict[int, torch.Tensor] = {}
        last_gp_raw: dict[int, torch.Tensor] = {}
        last_gp_scale: dict[int, int] = {}

        for step_idx in range(D):
            step_metrics = []
            self._disc_step_counter += 1
            # Always compute GP on the very first disc step to ensure the
            # Lipschitz constraint is active from the start — short runs
            # (e.g. smoke tests with num_iter < gp_interval) would otherwise
            # never trigger GP, causing WGAN to diverge immediately.
            compute_gp = (self._disc_step_counter == 1) or (
                self._disc_step_counter % self.gp_interval
            ) == 0

            # Phase 1: zero_grad for this D-step.
            for scale in sorted_scales:
                self._optimizer_zero_grad(optimizers[scale])

            # Store per-scale raw losses so metrics can be built after
            # the cross-rank scale-factor sync that follows.
            raw_losses: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
            # Discriminator always runs fp32: GP steps require it
            # (create_graph=True incompatible with AMP), and fp32 avoids
            # NaN from fp16 overflow in WGAN critic scores.
            use_disc_amp = False
            for scale in sorted_scales:
                fake = prefaked[scale][step_idx]
                real = facies_pyramid[scale]

                # Batch real+fake into a single disc forward to halve
                # kernel launches and improve GPU utilization (especially
                # for small batch sizes where each launch is under-utilized).
                _disc = unwrap_ddp(self.discriminator.discs[scale])
                with autocast("cuda", enabled=use_disc_amp, dtype=self._amp_dtype):
                    d_both = _disc(torch.cat([real, fake], dim=0))
                    d_real, d_fake = d_both[:B], d_both[B:]

                    real_loss = -d_real.mean()
                    fake_loss = d_fake.mean()

                if compute_gp:
                    # The lazy GP scheme multiplies raw penalty by gp_interval
                    # to compensate for computing it only 1-in-gp_interval steps.
                    # On the forced first step we use scale=1 (no compensation)
                    # so the initial update is not 8× too aggressive.
                    gp_scale = 1 if self._disc_step_counter == 1 else self.gp_interval
                    gp_raw = self.compute_gradient_penalty(scale, real, fake.detach())
                    gp = gp_raw * gp_scale
                    last_gp[scale] = gp.detach()
                    last_gp_raw[scale] = gp_raw.detach()
                    last_gp_scale[scale] = gp_scale
                else:
                    gp = self._zero_scalar

                # Per-scale loss normalization: divide by EMA of discriminator
                # output magnitude so coarse scales don't dominate training.
                sf = self.get_loss_scale_factor(scale)
                total = (real_loss + fake_loss + gp) / sf
                total.backward()  # type: ignore[no-untyped-call]

                # Update the EMA scale factor AFTER backward (no side effects
                # during autograd, and computed from raw un-normalized losses).
                # Keep d_mag as a GPU tensor to avoid a sync stall.
                d_mag = (real_loss.abs() + fake_loss.abs()).detach()
                self.update_loss_scale_factor(scale, d_mag)
                raw_losses[scale] = (
                    real_loss.detach(),
                    fake_loss.detach(),
                )

            # Synchronize loss_scale_factors across DDP ranks periodically.
            # The EMA (decay=0.99) converges quickly, so syncing every
            # 50 D-steps is sufficient to prevent drift while eliminating
            # ~98% of the per-step collectives.
            if (
                self.use_ddp
                and dist.is_initialized()
                and self._disc_step_counter % 50 == 0
            ):
                scales_list = sorted_scales
                # Scale factors are already GPU tensors — stack into a
                # contiguous buffer for the all-reduce without .item().
                sf_tensors: list[torch.Tensor] = [
                    (
                        self.loss_scale_factors[s]  # type: ignore[misc]
                        if isinstance(self.loss_scale_factors.get(s), torch.Tensor)
                        else torch.tensor(
                            self.loss_scale_factors.get(s, 1.0), device=self.device
                        )
                    )
                    for s in scales_list
                ]
                sf_buf = torch.stack(sf_tensors)
                dist.all_reduce(sf_buf, op=dist.ReduceOp.AVG)  # type: ignore[arg-type]
                for i, s in enumerate(scales_list):
                    self.loss_scale_factors[s] = sf_buf[i].detach()  # type: ignore[assignment]

            # Build per-scale metrics using the now-synced scale factors.
            for scale in sorted_scales:
                sf = self.get_loss_scale_factor(scale)
                rl, fl = raw_losses[scale]
                gp_val = last_gp.get(scale, self._zero_scalar)
                step_metrics.append(
                    DiscriminatorMetrics(
                        total=(rl + fl + gp_val) / sf,
                        real=rl / sf,
                        fake=fl / sf,
                        gp=gp_val / sf,
                    )
                )

            # Phase 2: coalesced all-reduce across all disc modules.
            if self.use_ddp:
                self._allreduce_grads_coalesced(
                    [self.discriminator.discs[s] for s in sorted_scales]
                )

            # Phase 3: step all disc optimizers (plain fp32).
            # NOTE: do NOT clip discriminator parameter gradients here.
            # WGAN-GP enforces the Lipschitz constraint via the gradient
            # penalty (on D's output gradient w.r.t. inputs), which is
            # fundamentally different from parameter-space clipping.
            # Adding clip_grad_norm_ on top of GP double-regularises D,
            # collapses the adversarial signal G needs to learn from, and
            # stalls training. The reduced gp_interval alone is sufficient
            # to prevent D_Total spikes without harming the critic.
            for scale in sorted_scales:
                optimizers[scale].step()

        # ── Extra discriminator steps for scale 0 ──────────────────────────
        # scale0_disc_steps_multiplier > 1 means scale 0 receives additional
        # D-step updates per iteration to keep the coarsest discriminator
        # sharper and prevent the generator from drifting at that scale.
        if scale0_multi > 1 and 0 in sorted_scales:
            s0_idx = sorted_scales.index(0)
            for step_idx in range(D, D * scale0_multi):
                extra_idx = step_idx - D  # 0-based index within the extra steps
                # Use a local index so the extra scale-0 steps do not pollute
                # _disc_step_counter, which controls GP timing for ALL scales
                # in the main D-step loop.  Incrementing it here would shift
                # the gp_interval modulo for scales 1-6 on subsequent calls.
                compute_gp = (extra_idx % self.gp_interval) == 0

                self._optimizer_zero_grad(optimizers[0])

                fake = prefaked[0][step_idx]
                real = facies_pyramid[0]
                _disc = unwrap_ddp(self.discriminator.discs[0])
                with autocast("cuda", enabled=False, dtype=self._amp_dtype):
                    d_both = _disc(torch.cat([real, fake], dim=0))
                    d_real, d_fake = d_both[:B], d_both[B:]
                    real_loss = -d_real.mean()
                    fake_loss = d_fake.mean()

                if compute_gp:
                    gp_raw = self.compute_gradient_penalty(0, real, fake.detach())
                    gp = gp_raw * self.gp_interval
                    last_gp[0] = gp.detach()
                    last_gp_raw[0] = gp_raw.detach()
                    last_gp_scale[0] = self.gp_interval
                else:
                    gp = self._zero_scalar

                sf = self.get_loss_scale_factor(0)
                total = (real_loss + fake_loss + gp) / sf
                total.backward()  # type: ignore[no-untyped-call]

                d_mag = (real_loss.abs() + fake_loss.abs()).detach()
                self.update_loss_scale_factor(0, d_mag)

                if self.use_ddp:
                    self._allreduce_grads_coalesced([self.discriminator.discs[0]])

                optimizers[0].step()

                # Keep scale 0 metrics up to date with latest extra step.
                gp_val = last_gp.get(0, self._zero_scalar)
                sf = self.get_loss_scale_factor(0)
                step_metrics[s0_idx] = DiscriminatorMetrics(
                    total=(real_loss.detach() + fake_loss.detach() + gp_val) / sf,
                    real=real_loss.detach() / sf,
                    fake=fake_loss.detach() / sf,
                    gp=gp_val / sf,
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
        """Generator optimization with gradient accumulation and per-scale freezing.

        Overrides the base ``optimize_generator`` to temporarily freeze
        all active-group gen blocks **except** the one being trained.
        This prevents ``backward()`` from computing (and then discarding)
        gradients for gen blocks that participate in the progressive
        forward pass but whose optimizer is not being stepped.

        Gradients are accumulated across all G forward-backward passes
        (each scaled by 1/G) with a single all-reduce + optimizer step
        at the end, reducing NCCL collectives from G to 1 per iteration.

        For *S* parallel scales training simultaneously, the base
        implementation computes gradients for 1+2+…+S = S(S+1)/2 block
        instances per G step.  This override reduces that to just *S*
        block instances — a ~(S+1)/2× speedup in backward computation
        (e.g. ~4× for 7 parallel scales).

        Returns
        -------
        tuple[GeneratorMetrics, ...]
            Metrics from the last generator step for each active scale.
        """
        sorted_scales = sorted(self.active_scales)
        G = self.generator_steps
        if G <= 0:
            return ()

        step_metrics: list[GeneratorMetrics] = []

        # Freeze discriminator for the entire G-phase — its weights are
        # never updated here and keeping requires_grad=False avoids
        # saving per-parameter activation buffers in the autograd graph
        # during the adversarial-loss forward through the disc.
        for s in sorted_scales:
            self.discriminator.discs[s].requires_grad_(False)

        # ── G-step loop: each step gets its own zero_grad → backward →
        # all-reduce → optimizer.step() cycle.  This matches the original
        # behavior and is correct for Adam (G separate parameter updates
        # vs 1 accumulated update produce different dynamics).

        for _ in range(G):
            step_metrics = []

            # Freeze all active gen blocks up front.  Blocks from
            # previous groups are already frozen by freeze_generator_scales.
            for s in sorted_scales:
                self.generator.gens[s].requires_grad_(False)

            # Phase 1: forward + backward for each scale (sequential
            # because each scale's forward depends on earlier frozen
            # blocks in the progressive chain).
            losses_by_scale: dict[int, torch.Tensor] = {}
            for scale in sorted_scales:
                if scale >= len(facies_pyramid):
                    continue

                if len(self.noise_amps) < scale + 1:
                    raise RuntimeError(
                        f"noise_amp not initialized for scale {scale}. "
                        "Call the project's noise initialization before training."
                    )

                # Unfreeze only the target scale's gen block.
                self.generator.gens[scale].requires_grad_(True)

                result, _ = self.compute_generator_metrics(
                    indexes,
                    scale,
                    facies_pyramid[scale],
                    rec_in_pyramid,
                    wells_pyramid,
                    masks_pyramid,
                    seismic_pyramid,
                )
                metrics = cast(GeneratorMetrics, result)

                # zero_grad + backward per scale per G-step.
                self._optimizer_zero_grad(optimizers[scale])
                if self._use_grad_scaler:
                    self._grad_scaler_g.scale(metrics.total).backward()  # type: ignore[no-untyped-call]
                else:
                    metrics.total.backward()  # type: ignore[no-untyped-call]

                losses_by_scale[scale] = metrics.total

                # Re-freeze so the next scale's backward skips this block.
                self.generator.gens[scale].requires_grad_(False)

                step_metrics.append(
                    GeneratorMetrics(
                        total=metrics.total.detach(),
                        fake=metrics.fake.detach(),
                        facies_rec=metrics.facies_rec.detach(),
                        well=metrics.well.detach(),
                        div=metrics.div.detach(),
                        rec_rock_physics=metrics.rec_rock_physics.detach(),
                        tv=metrics.tv.detach(),
                        elastic=metrics.elastic.detach(),
                        physics=metrics.physics.detach(),
                    )
                )

            # Restore requires_grad BEFORE the all-reduce so that
            # _allreduce_grads_coalesced's ``if p.requires_grad`` filter
            # includes the parameters whose .grad was filled by backward.
            for s in sorted_scales:
                if s in losses_by_scale:
                    self.generator.gens[s].requires_grad_(True)

            # Phase 2: coalesced all_reduce for all gen modules.
            if self.use_ddp:
                self._allreduce_grads_coalesced(
                    [
                        self.generator.gens[s]
                        for s in sorted_scales
                        if s in losses_by_scale
                    ]
                )

            # Phase 3: unscale + clip + step all gen optimizers.
            _clip_norm = getattr(self.options, "grad_clip_norm", 1.0)
            for scale in sorted_scales:
                if scale not in losses_by_scale:
                    continue
                if self._use_grad_scaler:
                    self._grad_scaler_g.unscale_(optimizers[scale])
                    if _clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.generator.gens[scale].parameters(), max_norm=_clip_norm
                        )
                    self._grad_scaler_g.step(optimizers[scale])
                else:
                    if _clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.generator.gens[scale].parameters(), max_norm=_clip_norm
                        )
                    optimizers[scale].step()
                optimizers[scale]._opt_called = True  # type: ignore[attr-defined]

            if self._use_grad_scaler:
                self._grad_scaler_g.update()

            # Restore requires_grad on remaining blocks for next G-step.
            for s in sorted_scales:
                self.generator.gens[s].requires_grad_(True)

        # Unfreeze discriminator so the next D-phase can compute grad.
        for s in sorted_scales:
            self.discriminator.discs[s].requires_grad_(True)

        return tuple(step_metrics)

    def _denormalize_rock_physics(
        self, tensor: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Denormalize a rock physics tensor (B, 6, H, W) to physical units.

        Returns a dictionary of physical property tensors.
        """
        from datasets.data_files import DataFiles

        out: dict[str, torch.Tensor] = {}
        # Iterate through the rock physics components produced by the generator
        for i, comp in enumerate(DataFiles.generator_output_rock_physics()):
            name = comp.name
            # Denormalize from [-1, 1] to physical units
            # Using direct buffer indexing for speed
            out[name] = ((tensor[:, i : i + 1, ...] + 1) / 2) * self.phys_diff[
                name
            ] + self.phys_min[name]

        return out

    def _get_wavelet_for_scale(self, scale: int) -> torch.Tensor:
        """Get or create the depth-resampled wavelet for a specific scale.

        Adjusts dz based on the height ratio between the target scale and
        the current scale.
        """
        if not hasattr(self, "_wavelets_z_cache"):
            self._wavelets_z_cache: dict[int, torch.Tensor] = {}

        if scale not in self._wavelets_z_cache:
            from physics.seismic import resample_wavelet_to_depth

            self._wavelets_z_cache[scale] = resample_wavelet_to_depth(
                self.wavelet_t,
                self.vp_ref,
                cast(torch.Tensor, self.wavelet_dt),
                cast(torch.Tensor, self.dz_pyramid)[scale],
            )

        return self._wavelets_z_cache[scale]

    def compute_generator_metrics(
        self,
        indexes: torch.Tensor,
        scale: int,
        real: torch.Tensor,
        facies_in_pyramid: dict[int, torch.Tensor],
        wells_pyramid: dict[int, torch.Tensor] = {},
        masks_pyramid: dict[int, torch.Tensor] = {},
        seismic_pyramid: dict[int, torch.Tensor] = {},
    ) -> tuple[
        GeneratorMetrics | IterableMetrics,
        dict[str, Any] | None,
    ]:
        """Common generator-metrics flow shared by frameworks.

        Parameters
        ----------
        indexes (list[int]):
            Batch/sample indices used to generate noise.
        scale (int):
            Pyramid scale index for which to compute the metrics.
        real (torch.Tensor):
            Ground-truth tensor for the current scale.
        facies_in (torch.Tensor):
            Reconstruction input tensor for the current scale.
        wells_pyramid (dict[int, torch.Tensor], optional):
            Wells tensors dict for conditioning, keyed by scale.
        masks_pyramid (dict[int, torch.Tensor], optional):
            Well mask tensors dict for conditioning, keyed by scale.
        seismic_pyramid (dict[int, torch.Tensor], optional):
            Seismic tensors dict for conditioning, keyed by scale.

        Returns
        -------
        tuple[
            GeneratorMetrics | dict[Any, Any] | None
        ]:
            Container with total, fake, facies_rec, well and div losses, and optional gradients dict.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """

        with autocast("cuda", enabled=self._use_amp, dtype=self._amp_dtype):
            # Generate diversity candidates
            fake_samples = self.generate_diverse_samples(
                indexes,
                scale,
                wells_pyramid,
                seismic_pyramid,
            )
            fake = fake_samples[0]

            # WGAN generator adversarial loss: -E[D(fake)].
            # Discriminator params are frozen for the entire G-phase
            # (see optimize_generator) so no per-scale toggle is needed.
            adv = self.compute_adversarial_loss(scale, fake)

            # Per-scale loss normalization (consistent with discriminator).
            # D runs first and populates scale factors; G reads them.
            adv = adv / self.get_loss_scale_factor(scale)

            mask = masks_pyramid.get(scale, None)
            well = wells_pyramid.get(scale, None)
            rec_rock_physics_loss = self._zero_scalar

            if getattr(self.options, "use_rock_physics", False):
                facies_C = self.num_facies_classes
                # Split real/fake into facies and rock_physics channels
                real_facies = real[:, :facies_C, ...]
                real_rock_physics = real[:, facies_C:, ...]
                fake_facies = fake[:, :facies_C, ...]
                fake_rock_physics = fake[:, facies_C:, ...]

                well = self.compute_masked_loss(
                    fake_facies,
                    real_facies,
                    well,
                    mask,
                )

                # --- Rock Physics Metrics ---
                # Pre-calculate physical units ONLY if needed
                needs_phys = (
                    getattr(self.options, "elastic_loss_penalty", 0) > 0
                    or getattr(self.options, "physics_loss_penalty", 0) > 0
                )
                phys = (
                    self._denormalize_rock_physics(fake_rock_physics)
                    if needs_phys
                    else {}
                )

                if getattr(self.options, "rec_rock_physics_loss_penalty", 0) > 0:
                    rec_rock_physics_loss = (
                        self.options.rec_rock_physics_loss_penalty
                        * F.huber_loss(fake_rock_physics, real_rock_physics)
                    )

                tv_loss = self._zero_scalar
                if getattr(self.options, "tv_loss_penalty", 0) > 0:
                    tv_loss = self.options.tv_loss_penalty * total_variation_loss(
                        fake_rock_physics
                    )

                elastic_loss = self._zero_scalar
                if getattr(self.options, "elastic_loss_penalty", 0) > 0:
                    # 2. Compute Elastic Consistency Loss: MSE(Ip/Is, VpVs)
                    eps = 1e-6
                    calc_vpvs = phys["Ip"] / (phys["Is"] + eps)
                    elastic_loss = self.options.elastic_loss_penalty * F.mse_loss(
                        calc_vpvs, phys["VP_VS"]
                    )

                physics_loss = self._zero_scalar
                if (
                    getattr(self.options, "physics_loss_penalty", 0) > 0
                    and seismic_pyramid.get(scale) is not None
                ):

                    # Estimate Vp for dynamic wavelet resampling
                    # Vp is estimated from Ip and mean density. We multiply by 1000
                    # to convert from Km/s to m/s. We clamp to the physical range.
                    rho_mean = cast(torch.Tensor, self.rho_mean)
                    vp_min = cast(torch.Tensor, self.vp_min)
                    vp_max = cast(torch.Tensor, self.vp_max)
                    vp_phys = phys["Ip"] / rho_mean
                    vp_mean = torch.mean(vp_phys).clamp(vp_min, vp_max)

                    physics_loss = (
                        self.options.physics_loss_penalty
                        * calculate_physics_loss(
                            fake_rock_physics[:, 0:1, ...],  # gen_ip_norm
                            seismic_pyramid[scale],  # real_seismic
                            self.wavelet_t,
                            cast(torch.Tensor, self.wavelet_dt),
                            cast(torch.Tensor, self.dz_pyramid)[scale],
                            vp_mean,
                            vp_min,
                            vp_max,
                            ip_min=cast(torch.Tensor, self.ip_min),
                            ip_max=cast(torch.Tensor, self.ip_max),
                            seis_min=cast(torch.Tensor, self.seis_min),
                            seis_max=cast(torch.Tensor, self.seis_max),
                            loss_fn="huber",
                            fixed_kernel_size=255,
                        )
                    )
            else:
                well = self.compute_masked_loss(
                    fake,
                    real,
                    well,
                    mask,
                )
                elastic_loss = self._zero_scalar
                physics_loss = self._zero_scalar
                tv_loss = self._zero_scalar

            div = self.compute_diversity_loss(fake_samples)
            facies_in = facies_in_pyramid[scale]
            facies_rec_loss = self.compute_facies_recovery_loss(
                indexes,
                scale,
                real,
                facies_in,
                wells_pyramid,
                seismic_pyramid,
            )

            # Apply extra loss weight at scale 0 to anchor the pyramid.
            if scale == 0 and self.scale0_loss_multiplier != 1.0:
                facies_rec_loss = facies_rec_loss * self.scale0_loss_multiplier
                rec_rock_physics_loss = (
                    rec_rock_physics_loss * self.scale0_loss_multiplier
                )
                well = well * self.scale0_loss_multiplier
                tv_loss = tv_loss * self.scale0_loss_multiplier
                elastic_loss = elastic_loss * self.scale0_loss_multiplier
                physics_loss = physics_loss * self.scale0_loss_multiplier

            total = (
                adv
                + well
                + facies_rec_loss
                + div
                + rec_rock_physics_loss
                + tv_loss
                + elastic_loss
                + physics_loss
            )

        del fake_samples  # free diversity candidates early

        # Detach component losses — backward() will be called on total;
        # the individual components are only needed as scalar logs.
        metrics = GeneratorMetrics(
            total=total,
            fake=adv.detach(),
            facies_rec=facies_rec_loss.detach(),
            well=well.detach(),
            div=div.detach(),
            rec_rock_physics=rec_rock_physics_loss.detach(),
            tv=tv_loss.detach(),
            elastic=elastic_loss.detach(),
            physics=physics_loss.detach(),
        )

        return metrics, None

    def concatenate_tensors(
        self, tensors: list[torch.Tensor], dim: int = 1
    ) -> torch.Tensor:
        """Concatenate a list of tensors along dimension `dim`.

        Uses PyTorch `torch.cat` and preserves device placement.

        Parameters
        ----------
        tensors (list):
            List of tensors to concatenate.
        dim (int, optional):

            Dimension along which to concatenate, by default 1.
        """
        return torch.cat(tensors, dim=dim)

    def split_tensor(self, tensor: torch.Tensor, chunks: int) -> list[torch.Tensor]:
        """Split a tensor into ``chunks`` equal parts along the batch dimension.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to split (batch dimension is dim 0).
        chunks : int
            Number of equal-sized chunks.

        Returns
        -------
        list
            List of ``chunks`` tensors.
        """
        return list(torch.chunk(tensor, chunks, dim=0))

    def cat_batch(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        """Concatenate tensors along the batch (first) dimension."""
        return torch.cat(tensors, dim=0)

    def compute_adversarial_loss(self, scale: int, fake: torch.Tensor) -> torch.Tensor:
        """Compute adversarial loss.

        Parameters
        ----------
        scale : int
            Pyramid scale index.
        fake : torch.Tensor
            Generated tensor.

        Returns
        -------
        torch.Tensor
            Negative mean discriminator score.
        """
        # Use the uncompiled discriminator so the backward pass runs
        # through PyTorch-native ops rather than Inductor-compiled fusion.
        # torch.compile's fused backward can produce NaN for multi-channel
        # inputs (e.g. 6-ch facies+rock_physics) where InstanceNorm variance
        # underflows in float16, while the uncompiled version stays stable.
        disc = unwrap_ddp(
            self._uncompiled_discs.get(scale, self.discriminator.discs[scale])
        )
        return self.adversarial_loss_penalty * (-disc(fake).mean())

    def compute_diversity_loss(self, fake_samples: list[torch.Tensor]) -> torch.Tensor:
        """Compute diversity loss across multiple generated `fake_samples`.

        Encourages different noise inputs to produce diverse outputs by
        penalizing small pairwise distances between flattened samples.
        Uses ``exp(-mean_sq_diff * 10)`` per pair, which saturates toward 0
        as outputs become more distinct.

        Uses a vectorized approach: stacks all samples into an ``(N, -1)``
        matrix, computes the full pairwise squared-distance matrix with a
        single matmul, and extracts the upper-triangular pairs.

        Parameters:
            fake_samples (list): List of generated samples to
                compare for diversity.

        Returns:
            torch.Tensor: Scalar diversity loss; zero when disabled or when
                fewer than two samples are provided.
        """
        if self.diversity_loss_penalty <= 0 or len(fake_samples) < 2:
            return self._zero_scalar
        n = len(fake_samples)
        if n == 2:
            # Fast path for the common N=2 case: single pairwise distance,
            # avoids triu_indices / sq_norms / indexing overhead.
            diff = fake_samples[0] - fake_samples[1]
            pair_dist = (diff * diff).mean()
            return self.diversity_loss_penalty * torch.exp(-pair_dist * 10)
        # Stack into (N, D) where D = B*C*H*W — single flatten + stack.
        flat = torch.stack([s.flatten() for s in fake_samples])  # (N, D)
        # Pairwise squared distances via ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        sq_norms = (flat * flat).sum(dim=1)  # (N,)
        # Only compute upper-triangle pairs (i < j)
        idx_i, idx_j = torch.triu_indices(n, n, offset=1, device=flat.device)
        pair_dists = (
            sq_norms[idx_i] + sq_norms[idx_j] - 2 * (flat[idx_i] * flat[idx_j]).sum(1)
        ) / flat.shape[
            1
        ]  # mean over D
        div_loss = torch.exp(-pair_dists * 10).mean()
        return self.diversity_loss_penalty * div_loss

    def compute_gradient_penalty(
        self, scale: int, real: torch.Tensor, fake: torch.Tensor
    ) -> torch.Tensor:
        """Compute the gradient penalty for WGAN-GP style regularization.

        The gradient penalty uses ``autograd.grad(create_graph=True)``
        which requires float32 tensors, so AMP autocast is explicitly
        disabled here.

        Args:
            scale (int): Discriminator scale index used for the penalty.
            real (torch.Tensor): Real samples tensor.
            fake (torch.Tensor): Fake samples tensor.

        Returns:
            torch.Tensor: Scalar gradient penalty term.
        """
        disc = unwrap_ddp(
            self._uncompiled_discs.get(scale, self.discriminator.discs[scale])
        )
        with autocast("cuda", enabled=False):
            return utils.calc_gradient_penalty(
                disc,
                real.float(),
                fake.float(),
                self.gradient_loss_penalty,
                self.device,
            )

    def compute_masked_loss(
        self,
        fake: torch.Tensor,
        real: torch.Tensor,
        well: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute mask-weighted MSE between `fake` and `real` at `scale`.

        parameters
        ----------
        fake (torch.Tensor):
            Generated tensor samples for the current scale.
        real (torch.Tensor):
            Ground-truth tensor samples for the current scale.
        wells (torch.Tensor):
            Well-conditioning tensor for the current scale.
        masks (torch.Tensor):
            Well mask tensor for the current scale.

        Returns
        -------
        torch.Tensor: Scalar masked MSE loss scaled by
            `self.well_loss_penalty`, or zero if no wells are used.
        """
        if well is None or mask is None:
            return self._zero_scalar
        return self.well_loss_penalty * masked_cross_entropy(fake, real, mask)

    def compute_facies_recovery_loss(
        self,
        indexes: torch.Tensor,
        scale: int,
        real: torch.Tensor,
        rec_in: torch.Tensor,
        wells_pyramid: dict[int, torch.Tensor] = {},
        seismic_pyramid: dict[int, torch.Tensor] = {},
    ) -> torch.Tensor:
        """Compute facies reconstruction (facies_rec) loss for given inputs.

        Parameters
        ----------
        indexes (list[int]):
            Batch/sample indices used to generate reconstruction noise.
        scale (int):
            Current pyramid scale.
        real (torch.Tensor):
            Ground-truth tensor (B, C, H, W).
        rec_in (torch.Tensor):
            Input tensor for reconstruction (from lower scale).
        wells_pyramid (dict[int, torch.Tensor], optional):
            Wells tensors dict for conditioning.
        seismic_pyramid (dict[int, torch.Tensor], optional):
            Seismic tensors dict for conditioning.

        Returns
        -------
            torch.Tensor: Scalar facies reconstruction loss weighted by `penalty`,
                or zero when facies_rec is disabled.
        """
        if (
            self.facies_rec_loss_penalty == 0
            or self._current_epoch < self._facies_rec_skip_epochs
        ):
            return self._zero_scalar

        rec_noise = self.get_pyramid_noise(
            scale,
            indexes,
            wells_pyramid,
            seismic_pyramid,
            rec=True,
        )

        fake_rec = self.generator(
            rec_noise,
            self.get_noise_amplitude(scale),
            in_noise=rec_in,
            stop_scale=scale,
        )

        # Rock Physics handling in facies_rec loop
        if getattr(self.options, "use_rock_physics", False):
            facies_C = self.num_facies_classes
            real_facies = real[:, :facies_C, ...]
            fake_rec_facies = fake_rec[:, :facies_C, ...]
            # dice_loss requires [0, 1] range; convert from [-1, 1]
            loss = self.facies_rec_loss_penalty * dice_loss(
                (fake_rec_facies + 1.0) / 2.0, (real_facies + 1.0) / 2.0
            )
        else:
            # dice_loss requires [0, 1] range; convert from [-1, 1]
            loss = self.facies_rec_loss_penalty * dice_loss(
                (fake_rec + 1.0) / 2.0, (real + 1.0) / 2.0
            )

        return loss

    def finalize_discriminator_scale(self, scale: int) -> None:
        """Finalize discriminator block after creation.

        Applies weight initialization, moves the block to the primary
        device, and broadcasts parameters from rank 0 when DDP is
        enabled.

        Note: Discriminators are **not** wrapped with DDP because the
        WGAN-GP gradient penalty uses ``autograd.grad(create_graph=True)``
        which is incompatible with DDP's in-place backward hooks.
        Gradients are instead all-reduced manually in
        :meth:`update_discriminator_weights`.

        Args:
            scale (int): Index of the discriminator scale to finalize.
        """
        self.discriminator.discs[scale].apply(utils.weights_init)
        self.discriminator.discs[scale] = self.discriminator.discs[scale].to(
            self.device
        )
        self.discriminator.discs[scale] = self.discriminator.discs[scale].to(  # type: ignore[call-overload]
            memory_format=torch.channels_last
        )
        if self.use_ddp:
            # Broadcast initial weights from rank 0 (DDP constructor does
            # this automatically for wrapped modules; we replicate it here).
            for p in self.discriminator.discs[scale].parameters():
                dist.broadcast(p.data, src=0)

        # Keep an uncompiled reference for gradient-penalty computation
        # (``create_graph=True`` is incompatible with compiled graphs).
        # Then compile the disc block for all regular forward passes.
        # The two share the same underlying parameters so gradient
        # updates through either are visible to both.
        self._uncompiled_discs[scale] = self.discriminator.discs[scale]
        if self._use_compile:
            self.discriminator.discs[scale] = torch.compile(  # type: ignore[assignment]
                self.discriminator.discs[scale],
                fullgraph=True,
                dynamic=False,
            )

    def finalize_generator_scale(self, scale: int, reinit: bool) -> None:
        """Finalize generator block after creation.

        Either initialize weights for a freshly reinitialized block or copy
        weights from the previous scale, then move to primary device and
        broadcast parameters from rank 0 when DDP is enabled.

        Note: Generators are **not** wrapped with DDP because the
        multi-scale forward pass shares intermediate tensors across
        scales, and DDP's in-place backward hooks corrupt the
        computation graph.  Gradients are instead all-reduced manually
        in :meth:`update_generator_weights`.

        Args:
            scale (int): Index of the generator scale to finalize.
            reinit (bool): Whether to initialize weights instead of copying.
        """
        if reinit:
            self.generator.gens[scale].apply(utils.weights_init)
        else:
            # Attempt to copy parameters from previous scale but only for
            # matching parameter shapes. This handles cases where feature
            # counts change between scales (e.g., parallel initialization)
            prev = self.generator.gens[scale - 1]
            src_state = prev.state_dict()
            tgt_state = self.generator.gens[scale].state_dict()

            # Build filtered state with only keys present in both and with
            # identical tensor shapes.
            filtered: dict[str, torch.Tensor] = {}
            for k, v in src_state.items():
                if k in tgt_state and v.shape == tgt_state[k].shape:
                    filtered[k] = v

            if filtered:
                # Load only the matching parameters; allow missing keys.
                self.generator.gens[scale].load_state_dict(filtered, strict=False)
            else:
                # No compatible parameters to copy; fall back to weight init.
                self.generator.gens[scale].apply(utils.weights_init)

        self.generator.gens[scale] = self.generator.gens[scale].to(self.device)
        self.generator.gens[scale] = self.generator.gens[scale].to(  # type: ignore[call-overload]
            memory_format=torch.channels_last
        )
        if self.use_ddp:
            # Broadcast initial weights from rank 0.
            for p in self.generator.gens[scale].parameters():
                dist.broadcast(p.data, src=0)
        # torch.compile is incompatible with torch.utils.checkpoint:
        # the compiled graph reorders saved tensors, causing metadata
        # mismatches during checkpoint recomputation.  Skip compile on
        # gen blocks when gradient checkpointing is active.
        if self._use_compile and not getattr(
            self.generator, "use_gradient_checkpointing", False
        ):
            # dynamic=True: all 7 pyramid levels share the same compiled
            # graph via symbolic H/W dims.  With dynamic=False each level's
            # unique spatial shape counts as a new specialization for the
            # shared nn.Sequential.forward code object, quickly hitting
            # torch._dynamo.config.cache_size_limit and raising
            # FailOnRecompileLimitHit (fullgraph=True treats it as hard error).
            self.generator.gens[scale] = torch.compile(  # type: ignore[assignment]
                self.generator.gens[scale],
                fullgraph=True,
                dynamic=True,
            )

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
        """Perform a forward pass and compute scale metrics.

        Parameters
        ----------
        indexes (list[int]):
            List of batch/sample indices used to generate noise.
        facies_pyramid (dict[int, torch.Tensor]):
            Dictionary mapping scale indices to real tensor samples.
        rec_in_pyramid (dict[int, torch.Tensor]):
            Dictionary mapping scale indices to reconstruction input tensors.
        wells_pyramid (dict[int, torch.Tensor], optional):
            Wells tensors dictionary for conditioning, keyed by scale.
        masks_pyramid (dict[int, torch.Tensor], optional):
            Well masks dictionary for conditioning, keyed by scale.
        seismic_pyramid (dict[int, torch.Tensor], optional):
            Seismic tensors dictionary for conditioning, keyed by scale.
        Returns
        -------
        ScaleMetrics:
            Container with discriminator and generator metrics for the scale.
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

        # Metrics from both optimizers are already detached; pass through
        # directly without redundant .detach() calls.

        # Convert tuples to dicts mapping scale index to metrics
        discriminator_metrics = {
            scale: disc_metrics_tuple[i]
            for i, scale in enumerate(sorted(self.active_scales))
        }
        generator_metrics = {
            scale: gen_metrics_tuple[i]
            for i, scale in enumerate(sorted(self.active_scales))
        }

        return ScaleMetrics(
            discriminator=discriminator_metrics,
            generator=generator_metrics,
        )

    def generate_fake(self, noises: list[torch.Tensor], scale: int) -> torch.Tensor:
        """Generate a fake sample at the requested `scale` using `noises`.

        Uses ``no_grad`` to avoid tracking generator computation during
        discriminator optimization.  ``inference_mode`` cannot be used here
        because WGAN-GP's gradient penalty passes the resulting tensor
        through the discriminator with ``create_graph=True``, and inference
        tensors cannot be saved for backward.

        Args:
            noises (list): Noise inputs for the generator per scale.
            scale (int): Target scale index to generate.

        Returns:
            torch.Tensor: Generated fake tensor for the requested scale.
        """
        with torch.no_grad():
            amps = self.get_noise_amplitude(scale)
            fake = self.generator(noises, amps, stop_scale=scale)
        return fake

    def generate_noise(
        self,
        scale: int,
        indexes: torch.Tensor,
        well: torch.Tensor | None = None,
        seismic: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Create a noise tensor for a single pyramid level, optionally
        concatenating conditioning channels and applying padding.

        Parameters
        ----------
        scale : int
            Pyramid level index used to select shapes and conditioning tensors.
        indexes : list[int]
            Batch/sample indices to select conditioning slices from stored per-scale tensors
        wells : torch.Tensor, optional
            Well-conditioning tensor for the current scale, by default torch.Tensor().
        seismic : torch.Tensor, optional
            Seismic-conditioning tensor for the current scale, by default torch.Tensor().

        Returns
        -------
        torch.Tensor
            Padded noise tensor for the requested level, possibly concatenated with well
            and/or seismic conditioning.
        """

        batch = len(indexes)
        spatial_shape = self.get_noise_shape(scale, use_base_channel=False)
        noise_channels = self.gen_input_channels
        tensors_to_concat: list[torch.Tensor] = []

        w: torch.Tensor | None = None
        s: torch.Tensor | None = None

        if well is not None:
            w = well.to(self.device, non_blocking=True)
            if w.shape[0] != batch:
                w = w[indexes]
            noise_channels -= w.shape[1]

        if seismic is not None:
            s = seismic.to(self.device, non_blocking=True)
            if s.shape[0] != batch:
                s = s[indexes]
            noise_channels -= s.shape[1]

        z = utils.generate_noise(
            (noise_channels, *spatial_shape),
            num_samp=batch,
            device=self.device,
        )
        tensors_to_concat.append(z)

        if w is not None:
            tensors_to_concat.append(w)

        if s is not None:
            tensors_to_concat.append(s)

        if len(tensors_to_concat) > 1:
            z = self.concatenate_tensors(tensors_to_concat)

        return self.generate_padding(z, value=0)

    def get_rec_noise(self, scale: int) -> list[float]:
        return self.rec_noise[: scale + 1]

    def generate_padding(self, z: torch.Tensor, value: int = 0) -> torch.Tensor:
        """Pad tensor `z` using the model's zero-padding size.

        Args:
            z (torch.Tensor): Input tensor to pad.
            value (int): Padding fill value (default: 0).

        Returns:
            torch.Tensor: Padded tensor.
        """
        return F.pad(z, [self.zero_padding] * 4, value=value)

    def load_amp(self, scale_path: str) -> None:
        """Default loader for amplitude files created by `save_amp`.

        Reads the text file named by `AMP_FILE` and appends the parsed float to
        `self.noise_amps` if present.
        while providing a sensible default implementation.
        """
        amp_path = os.path.join(scale_path, AMP_FILE)
        if os.path.exists(amp_path):
            with open(amp_path, "r") as f:
                self.noise_amps.append(torch.tensor(float(f.read().strip())))

    def get_noise_shape(
        self, scale: int, use_base_channel: bool = True
    ) -> tuple[int, ...]:
        """Return the noise shape tuple for a given `scale`.

        Args:
            scale (int): Scale index for which to get the noise shape.

        Returns:
            tuple[int, ...]: Noise shape tuple as (channels, height, width).
        """
        return (
            (self.base_channel, *self.shapes[scale][2:])
            if use_base_channel
            else self.shapes[scale][2:]
        )

    def load_discriminator_state(self, scale_path: str, scale: int) -> None:
        """Load discriminator state dict for `scale` from `scale_path` if present.

        Handles DDP-wrapped modules by loading into the inner
        ``.module`` when applicable.  Also normalises ``_orig_mod.``
        prefix from ``torch.compile``.

        Args:
            scale_path (str): Directory path for the given scale.
            scale (int): Index of the discriminator scale to load.
        """
        disc_path = os.path.join(scale_path, D_FILE)
        if os.path.exists(disc_path):
            state = torch.load(disc_path, map_location=self.device)
            utils.load_framework_state_dict(self.discriminator.discs[scale], state)

    def load_generator_state(self, scale_path: str, scale: int) -> None:
        """Load generator state dict for the latest generator in the scale.

        Handles DDP-wrapped modules by loading into the inner
        ``.module`` when applicable.  Also strips the ``_orig_mod.``
        prefix added by ``torch.compile`` when the current model is
        uncompiled (and vice-versa).

        Args:
            scale_path (str): Directory path for the given scale.
            scale (int): Index of the generator scale to load (unused here).
        """
        gen_path = os.path.join(scale_path, G_FILE)
        if os.path.exists(gen_path):
            state = torch.load(gen_path, map_location=self.device)
            utils.load_framework_state_dict(self.generator.gens[scale], state)

    def load_shape(self, scale_path: str) -> None:
        """Load saved shape tensor for a scale and append to `self.shapes`.

        Args:
            scale_path (str): Directory path for the given scale.
        """
        shape_path = os.path.join(scale_path, SHAPE_FILE)
        if os.path.exists(shape_path):
            self.shapes.append(torch.load(shape_path, map_location=self.device))

    def load_wells(self, scale_path: str) -> None:
        """Load well-conditioning mask for a scale and append to `self.wells`.

        Args:
            scale_path (str): Directory path for the given scale.
        """
        wells: list[torch.Tensor] = []
        wells.append(
            utils.load(
                os.path.join(scale_path, M_FILE),
                self.device,
                as_type=torch.Tensor,
            )
        )
        self.wells = tuple(wells)

    def move_to_device(self, obj: Any, device: torch.device | None = None) -> Any:
        """Move PyTorch modules or tensors to a target device.

        Args:
            obj (Any): Module or tensor to move.
            device (torch.device | None): Destination device. If None, uses
                `self.device`.

        Returns:
            Any: The object moved to the target device.
        """
        return obj.to(device or self.device)

    def _wait_pending_allreduce(self) -> None:
        """Wait for any pending async all-reduce operation to complete.

        When _pending_allreduce_work is set (from an async all-reduce),
        this method blocks until the operation finishes and clears the reference.
        Safe to call even if no work is pending.

        This enables NVLink overlap: compute (backward on next layer/scale)
        can proceed while the all-reduce of a previous scale's gradients
        is in flight.
        """
        if self._pending_allreduce_work is not None:
            self._pending_allreduce_work.wait()
            self._pending_allreduce_work = None

    def _complete_pending_disc_allreduce(self) -> None:
        """Complete deferred disc gradient sync and optimizer step.

        On the last D-step, the disc gradient all-reduce is launched
        asynchronously so that NCCL runs on its own stream while the
        generator forward pass queues kernels on the compute stream.
        This method waits for the NCCL collective, scatters the averaged
        gradients back into ``param.grad``, and steps the disc optimizers.

        Safe to call when no async work is pending (no-op).
        """
        if self._pending_disc_ar_work is None:
            return
        self._pending_disc_ar_work.wait()
        assert self._pending_disc_ar_flat is not None
        assert self._pending_disc_ar_grads is not None
        assert self._pending_disc_ar_opts is not None
        assert self._pending_disc_ar_scales is not None
        synced_grads = cast(
            list[torch.Tensor],
            torch._utils._unflatten_dense_tensors(  # type: ignore[attr-defined]
                self._pending_disc_ar_flat, self._pending_disc_ar_grads
            ),
        )
        for g, synced in zip(self._pending_disc_ar_grads, synced_grads):
            g.copy_(synced)
        for scale in self._pending_disc_ar_scales:
            self._pending_disc_ar_opts[scale].step()
        self._pending_disc_ar_work = None
        self._pending_disc_ar_flat = None
        self._pending_disc_ar_grads = None
        self._pending_disc_ar_opts = None
        self._pending_disc_ar_scales = None

    @staticmethod
    def _optimizer_zero_grad(optimizer: torch.optim.Optimizer) -> None:
        """Clear gradients using set_to_none when optimizer API supports it."""
        try:
            optimizer.zero_grad(set_to_none=True)
        except TypeError:
            optimizer.zero_grad()

    def _allreduce_grads(self, module: nn.Module) -> dist.Work | None:
        """Average gradients of a single module across DDP ranks (wrapped helper)."""
        return self._allreduce_grads_coalesced([module])

    def report_allreduce_profile(self) -> None:
        """Print aggregate timing for coalesced gradient sync profiling."""
        if not self._profile_allreduce or self._profile_allreduce_calls == 0:
            return
        avg_s = self._profile_allreduce_total_s / self._profile_allreduce_calls
        avg_collective_s = (
            self._profile_collective_total_s / self._profile_allreduce_calls
        )
        avg_elems = self._profile_allreduce_total_elems / self._profile_allreduce_calls
        print("\n[ALLREDUCE_PROFILE] coalesced gradient sync summary")
        print(
            "[ALLREDUCE_PROFILE] "
            "mode=ddp_all_reduce "
            "profiling=detailed "
            f"calls={self._profile_allreduce_calls} "
            f"avg_total_time={avg_s:.6f}s "
            f"avg_collective_time={avg_collective_s:.6f}s "
            f"avg_elems={avg_elems:.0f} "
            f"total_time={self._profile_allreduce_total_s:.6f}s "
            f"collective_time={self._profile_collective_total_s:.6f}s"
        )

    def _sync_profile_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _allreduce_grads_coalesced(
        self, modules: list[nn.Module], async_op: bool = False
    ) -> dist.Work | None:
        """Average gradients across all DDP ranks with a **single** all-reduce.

        Flattens trainable-parameter gradients from *all* provided modules
        into one contiguous buffer, performs a single NCCL ``all_reduce``,
        and scatters the averaged gradients back.

        Every trainable parameter is included even if ``backward()`` left
        its ``.grad`` as ``None`` (a zero tensor is substituted so the
        flat buffer size is identical across ranks — avoiding NCCL
        deadlocks from mismatched collective calls).

        Parameters
        ----------
        modules : list[nn.Module]
            Modules whose gradients to synchronize.
        async_op : bool, optional
            If True, return the Work object and do not block.
            Caller must call .wait() on the returned object before using
            the synchronized gradients. Enables compute/comm overlap on
            high-bandwidth links (NVLink, etc.). Default is False (blocking).

        Returns
        -------
        dist.Work | None
            If async_op=True: Work object to be waited on later.
            If async_op=False: None (all-reduce is complete).
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

        flatten_t0 = 0.0
        if self._profile_allreduce:
            self._sync_profile_device()
            flatten_t0 = time.perf_counter()
        flat = cast(
            torch.Tensor,
            torch._utils._flatten_dense_tensors(grads),  # type: ignore[attr-defined]
        )
        if self._profile_allreduce:
            self._sync_profile_device()
            _ = time.perf_counter() - flatten_t0

        prof_t0 = time.perf_counter() if self._profile_allreduce else 0.0
        collective_s = 0.0
        collective_t0 = 0.0

        if self._profile_allreduce and not async_op:
            self._sync_profile_device()
            collective_t0 = time.perf_counter()
        work = dist.all_reduce(  # type: ignore[arg-type]
            flat, op=dist.ReduceOp.AVG, async_op=async_op
        )
        if self._profile_allreduce and not async_op:
            self._sync_profile_device()
            collective_s = time.perf_counter() - collective_t0

        # For async_op=False, work is None and gradients are already synced.
        # For async_op=True, caller must .wait() on the returned Work before
        # using the gradients.
        if async_op:
            return work  # type: ignore[return-value]

        # Synchronous path: scatter synced gradients immediately
        scatter_t0 = 0.0
        if self._profile_allreduce:
            self._sync_profile_device()
            scatter_t0 = time.perf_counter()
        for g, synced in zip(  # type: ignore[assignment]
            grads, torch._utils._unflatten_dense_tensors(flat, grads)  # type: ignore[attr-defined]
        ):
            g.copy_(synced)  # type: ignore[arg-type]
        if self._profile_allreduce:
            self._sync_profile_device()
            _ = time.perf_counter() - scatter_t0
        if self._profile_allreduce:
            self._sync_profile_device()
            self._profile_allreduce_total_s += time.perf_counter() - prof_t0
            self._profile_collective_total_s += collective_s
            self._profile_allreduce_calls += 1
            self._profile_allreduce_total_elems += flat.numel()
        return None

    def update_discriminator_weights(
        self,
        scale: int,
        optimizer: torch.optim.Optimizer,
        loss: torch.Tensor,
        gradients: Any | None,
    ) -> None:
        """Perform standard PyTorch discriminator optimization step (fp32).

        The discriminator always uses fp32 (no AMP / GradScaler) for two
        reasons: (1) GP steps require fp32 because
        ``autograd.grad(create_graph=True)`` is incompatible with loss
        scaling; (2) non-GP and GP steps are accumulated into the same
        ``param.grad`` buffer, so mixing scaled fp16 and unscaled fp32
        gradients in one accumulation cycle would corrupt the gradient.

        When running under DDP, discriminator gradients are manually
        all-reduced across ranks because the discriminator is not
        wrapped with DDP (see :meth:`finalize_discriminator_scale`).

        NVLink optimization: waits for pending async all-reduce before step.
        """
        self._optimizer_zero_grad(optimizer)
        loss.backward()  # type: ignore[no-untyped-call]
        if self.use_ddp:
            self._allreduce_grads(self.discriminator.discs[scale])
        self._wait_pending_allreduce()
        optimizer.step()

    def update_generator_weights(
        self,
        scale: int,
        optimizer: torch.optim.Optimizer,
        loss: torch.Tensor,
        gradients: Any | None,
    ) -> None:
        """Perform standard PyTorch generator optimization step with AMP.

        When running under DDP, generator gradients are manually
        all-reduced across ranks because the generator is not wrapped
        with DDP (see :meth:`finalize_generator_scale`).

        NVLink optimization: waits for pending async all-reduce before step.
        """
        self._optimizer_zero_grad(optimizer)
        if self._use_grad_scaler:
            self._grad_scaler_g.scale(loss).backward()  # type: ignore[no-untyped-call]
        else:
            loss.backward()  # type: ignore[no-untyped-call]
        if self.use_ddp:
            self._allreduce_grads(self.generator.gens[scale])
        self._wait_pending_allreduce()
        if self._use_grad_scaler:
            self._grad_scaler_g.unscale_(optimizer)
            self._grad_scaler_g.step(optimizer)
        else:
            optimizer.step()
        # GradScaler may skip optimizer.step() on inf/nan, leaving
        # _opt_called unset → spurious LRScheduler warning.
        optimizer._opt_called = True  # type: ignore[attr-defined]
        # NOTE: scaler.update() is called once per G iteration in
        # optimize_generator, not here — calling it per-scale would
        # adjust the scale factor 7× too often.

    def save_discriminator_state(self, scale_path: str, scale: int) -> None:
        """Save discriminator state dict for `scale` to `scale_path`.

        Unwraps DDP to save clean state dicts without ``module.``
        prefix.

        Args:
            scale_path (str): Directory path for the given scale.
            scale (int): Index of the discriminator scale to save.
        """
        if scale < len(self.discriminator.discs):
            discriminator_path = os.path.join(scale_path, f"{D_FILE}")
            torch.save(
                unwrap_ddp(self.discriminator.discs[scale]).state_dict(),
                discriminator_path,
            )

    def save_generator_state(self, scale_path: str, scale: int) -> None:
        """Save generator state dict for `scale` to `scale_path`.

        Unwraps DDP to save clean state dicts without ``module.``
        prefix.

        Args:
            scale_path (str): Directory path for the given scale.
            scale (int): Index of the generator scale to save.
        """
        if scale < len(self.generator.gens):
            generator_path = os.path.join(scale_path, f"{G_FILE}")
            torch.save(
                unwrap_ddp(self.generator.gens[scale]).state_dict(), generator_path
            )

    def save_shape(self, scale_path: str, scale: int) -> None:
        """Save shape tensor for `scale` to disk at `scale_path`.

        Args:
            scale_path (str): Directory path for the given scale.
            scale (int): Index of the shape to save.
        """
        if scale < len(self.shapes):
            shape_path = os.path.join(scale_path, SHAPE_FILE)
            torch.save(self.shapes[scale], shape_path)
