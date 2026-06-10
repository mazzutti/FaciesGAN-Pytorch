"""Typed dataclasses for transporting per-scale training metrics.

The dataclasses in this module are used to return tensor-valued losses and
metrics from model optimization routines. Fields intentionally remain as
``torch.Tensor`` scalars during the forward/backward pass so autograd is
preserved; callers should convert to Python floats (``.item()``) when logging
or writing to external sinks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from config import DomainConfig, PhysicsConfig
from device import device_manager
from enums import DeviceType, MetricKey
from options import TrainingOptions

if TYPE_CHECKING:
    from physics.physics import PhysicsState


@dataclass
class DiscriminatorMetrics:
    """Per-scale discriminator metric container.

    Fields are scalar ``torch.Tensor`` values representing the total loss,
    real/fake component losses and gradient penalty (``gp``).
    """

    total: torch.Tensor
    real: torch.Tensor
    fake: torch.Tensor
    gp: torch.Tensor

    def as_dict(self) -> dict[str, float]:
        """Return the metric values as a dictionary.

        Uses a single ``torch.stack().tolist()`` call to convert all fields
        in one GPU→CPU sync instead of one sync per field.

        Returns
        -------
        dict[str, float]
            Dictionary mapping metric names to their tensor values.
        """
        vals: list[float] = torch.stack(  # type: ignore[call-overload]
            device_manager.to_cpu(
                [self.total, self.real, self.fake, self.gp],
                non_blocking=True,
            )
        ).tolist()  # type: ignore[return-value]
        return {
            MetricKey.D_TOTAL: vals[0],
            MetricKey.D_REAL: vals[1],
            MetricKey.D_FAKE: vals[2],
            MetricKey.D_GP: vals[3],
        }

    def as_tuple(self) -> tuple[torch.Tensor, ...]:
        """Return the metric values as a tuple in a fixed order.

        Returns
        -------
        tuple[torch.Tensor, ...]
            Tuple of metric values in the order:
            (total, real, fake, gp).
        """
        return self.total, self.real, self.fake, self.gp


@dataclass
class GeneratorMetrics:
    """Per-scale generator metric container.

    Fields are scalar ``torch.Tensor`` values for adversarial (``fake``),
    reconstruction (``rec``), well-constraint (``well``), diversity (``div``)
    and aggregated total loss (``total``).
    """

    total: torch.Tensor
    fake: torch.Tensor
    rec_facies: torch.Tensor
    well: torch.Tensor
    div: torch.Tensor
    rec_rock_physics: torch.Tensor
    tv: torch.Tensor
    elastic: torch.Tensor
    seismic: torch.Tensor

    def as_dict(self) -> dict[str, float]:
        """Return the metric values as a dictionary.

        Uses a single ``torch.stack().tolist()`` call to convert all fields
        in one GPU→CPU sync instead of one sync per field.

        Returns
        -------
        dict[str, float]
            Dictionary mapping metric names to their tensor values.
        """
        vals: list[float] = torch.stack(  # type: ignore[call-overload]
            device_manager.to_cpu(
                [
                    self.total,
                    self.fake,
                    self.rec_facies,
                    self.well,
                    self.div,
                    self.rec_rock_physics,
                    self.tv,
                    self.elastic,
                    self.seismic,
                ],
                non_blocking=True,
            )
        ).tolist()  # type: ignore[return-value]

        return {
            MetricKey.G_TOTAL: vals[0],
            MetricKey.G_FAKE: vals[1],
            MetricKey.G_REC_FACIES: vals[2],
            MetricKey.G_WELL: vals[3],
            MetricKey.G_DIV: vals[4],
            MetricKey.G_REC_ROCK_PHYSICS: vals[5],
            MetricKey.G_TV: vals[6],
            MetricKey.G_ELASTIC: vals[7],
            MetricKey.G_SEISMIC: vals[8],
        }

    def as_tuple(self) -> tuple[torch.Tensor, ...]:
        """Return the metric values as a list in a fixed order.

        Returns
        -------
        tuple[torch.Tensor, ...]
            Tuple of metric values in the order:
            (total, fake, rec_facies, well, div, rp, elastic, seismic).
        """
        return (
            self.total,
            self.fake,
            self.rec_facies,
            self.well,
            self.div,
            self.rec_rock_physics,
            self.tv,
            self.elastic,
            self.seismic,
        )


@dataclass
class ScaleMetric:
    """Container for generator and discriminator metrics at a specific scale."""

    generator: GeneratorMetrics
    discriminator: DiscriminatorMetrics


@dataclass
class ScaleMetrics:
    """Mapping of scale index to per-scale metric dataclasses.

    ``generator`` and ``discriminator`` map integer scale indices to the
    corresponding metric dataclasses defined above.
    """

    generator: dict[int, GeneratorMetrics]
    discriminator: dict[int, DiscriminatorMetrics]

    def get_metrics(self, scale: int) -> ScaleMetric | None:
        """Return a ScaleMetric container for the given scale if it exists."""
        if scale in self.generator and scale in self.discriminator:
            return ScaleMetric(self.generator[scale], self.discriminator[scale])
        return None

    @staticmethod
    def from_dicts(
        gen_dict: dict[int, list[torch.Tensor]],
        disc_dict: dict[int, list[torch.Tensor]],
    ) -> ScaleMetrics:
        """Construct a ``ScaleMetrics`` instance from per-scale metric dicts.

        Parameters
        ----------
        gen_dict : dict[int, list[torch.Tensor]]
            Mapping of scale index to generator metrics.
        disc_dict : dict[int, list[torch.Tensor]]
            Mapping of scale index to discriminator metrics.

        Returns
        -------
        ScaleMetrics
            Constructed ``ScaleMetrics`` instance.
        """
        return ScaleMetrics(
            generator={
                int(scale): GeneratorMetrics(
                    total=metrics[0],
                    fake=metrics[1],
                    rec_facies=metrics[2],
                    well=metrics[3],
                    div=metrics[4],
                    rec_rock_physics=metrics[5],
                    tv=metrics[6],
                    elastic=metrics[7],
                    seismic=metrics[8],
                )
                for scale, metrics in gen_dict.items()
            },
            discriminator={
                int(scale): DiscriminatorMetrics(*metrics)
                for scale, metrics in disc_dict.items()
            },
        )

    def as_flat_dict(self) -> dict[int, dict[str, float]]:
        """Return a mapping of scale index to flattened metric dictionary.

        Merges generator and discriminator metrics for each scale into a
        single flat dictionary suitable for logging.
        """
        return {
            s: {
                **self.generator[s].as_dict(),
                **self.discriminator[s].as_dict(),
            }
            for s in sorted(self.generator.keys())
        }

    def as_tuple_of_dicts(self) -> tuple[dict[str, float], ...]:
        """Return all metric values as a tuple of dictionaries in a fixed order.

        The order is by scale index (ascending), with generator metrics
        preceding discriminator metrics at each scale.

        Returns
        -------
        tuple[dict[str, float], ...]
            Tuple of dictionaries mapping metric names to their values in the order:
            (gen_scale0..., disc_scale0..., gen_scale1..., disc_scale1..., ...).
        """
        return tuple(
            x
            for scale in sorted(self.generator.keys())
            for x in (
                self.generator[scale].as_dict(),
                self.discriminator[scale].as_dict(),
            )
        )

    def as_tuple(self) -> tuple[torch.Tensor, ...]:
        """Return all metric values as a flat list in a fixed order.

        The order is by scale index (ascending), with generator metrics
        preceding discriminator metrics at each scale.

        Returns
        -------
        tuple[torch.Tensor, ...]
            Flat tuple of all metric values in the order:
            (gen_scale0..., disc_scale0..., gen_scale1..., disc_scale1..., ...).
        """
        return tuple(
            x
            for scale in sorted(self.generator.keys())
            for x in (
                *self.generator[scale].as_tuple(),
                *self.discriminator[scale].as_tuple(),
            )
        )


IterableMetrics = tuple[
    dict[int, list[tuple[torch.Tensor, ...]]],
    dict[int, list[dict[str, Any]]],
]


class MetricSmoother:
    """Exponential Moving Average smoother (same formula as TensorBoard).

    Smoothed value:  ``s_t = alpha * s_{t-1} + (1 - alpha) * x_t``

    Parameters
    ----------
    alpha : float
        Smoothing factor in [0, 1).  Higher = smoother / slower to react.
    """

    __slots__ = ("alpha", "value")

    def __init__(self, alpha: float = 0.9) -> None:
        self.alpha = alpha
        self.value: float | None = None

    def update(self, raw: float) -> float:
        """Feed a new raw value and return the smoothed result."""
        if self.value is None:
            self.value = raw
        else:
            self.value = self.alpha * self.value + (1.0 - self.alpha) * raw
        return self.value

    def reset(self) -> None:
        self.value = None


class MetricArraySmoother:
    """Vectorized EMA smoother for a fixed-length array of metrics.

    Equivalent to ``n`` independent :class:`MetricSmoother` instances but
    uses a single :mod:`numpy` multiply-add instead of ``n`` Python calls.

    Smoothed value:  ``s_t = alpha * s_{t-1} + (1 - alpha) * x_t``

    Parameters
    ----------
    n : int
        Number of metrics to track simultaneously.
    alpha : float
        Smoothing factor in [0, 1).  Higher = smoother / slower to react.
    """

    __slots__ = ("alpha", "_one_minus_alpha", "values", "_n")

    def __init__(self, n: int, alpha: float = 0.9) -> None:
        self.alpha = alpha
        self._one_minus_alpha = 1.0 - alpha
        self.values: np.ndarray | None = None
        self._n = n

    def update(self, raw: list[float]) -> list[float]:
        """Feed a new raw vector and return the smoothed result as a list."""
        raw_arr = np.asarray(raw, dtype=np.float64)
        if self.values is None:
            self.values = raw_arr.copy()
        else:
            self.values = self.alpha * self.values + self._one_minus_alpha * raw_arr
        return self.values.tolist()  # type: ignore[return-value]

    def reset(self) -> None:
        self.values = None


def rms_normalize(x: torch.Tensor, eps: float = DomainConfig.EPSILON) -> torch.Tensor:
    """Apply Root-Mean-Square (RMS) normalization to a tensor.

    Computes the mean square in float32 and limits amplification to prevent
    explosive values (NaNs) when the signal variance is near zero.
    """
    ms = torch.mean(x.to(torch.float32) ** 2)
    # ms is mean square. Add epsilon inside sqrt to prevent NaN gradients at zero.
    denom = torch.sqrt(ms + eps)
    return (x.to(torch.float32) / denom).to(x.dtype)


def compute_adversarial_loss(
    disc: torch.nn.Module, fake: torch.Tensor, penalty: float
) -> torch.Tensor:
    """Compute adversarial loss for a generated tensor at a scale."""
    return penalty * (-disc(fake).mean())


import torch.nn.functional as F
from torch.amp.autocast_mode import autocast

from enums import LossFn


def compute_diversity_loss(
    fake_samples: list[torch.Tensor], penalty_weight: float, zero_scalar: torch.Tensor
) -> torch.Tensor:
    """Compute diversity loss across multiple generated `fake_samples`."""
    if penalty_weight <= 0 or len(fake_samples) < 2:
        return zero_scalar
    n = len(fake_samples)
    if n == 2:
        diff = fake_samples[0] - fake_samples[1]
        pair_dist = torch.mean(torch.abs(diff))
        return penalty_weight * torch.exp(-pair_dist * 10)

    flat = torch.stack([s.flatten() for s in fake_samples])
    sq_norms = (flat * flat).sum(dim=1)
    idx_i, idx_j = torch.triu_indices(n, n, offset=1, device=flat.device)
    pair_dists = (
        sq_norms[idx_i] + sq_norms[idx_j] - 2 * (flat[idx_i] * flat[idx_j]).sum(1)
    ) / flat.shape[1]
    div_loss = torch.exp(-pair_dists * 10).mean()
    return penalty_weight * div_loss


def compute_gradient_penalty(
    disc: torch.nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    lambda_gp: float,
) -> torch.Tensor:
    """Compute the gradient penalty for WGAN-GP style regularization."""
    import models.utils as model_utils

    with autocast(DeviceType.CUDA, enabled=False):
        return model_utils.calc_gradient_penalty(
            disc,
            real.float(),
            fake.float(),
            lambda_gp,
        )


def compute_masked_loss(
    fake: torch.Tensor,
    real: torch.Tensor,
    well: torch.Tensor | None,
    mask: torch.Tensor | None,
    options: TrainingOptions,
    eps: float = DomainConfig.EPSILON,
) -> torch.Tensor:
    """Compute mask-weighted MSE between `fake` and `real`."""
    if well is None or mask is None:
        return DomainConfig.ZERO_SCALAR

    if options.use_rock_physics:
        fc = options.num_facies_channels
        fake = fake[:, :fc, ...]
        real = real[:, :fc, ...]

    mse = F.mse_loss(fake * mask, real * mask)
    rmse = torch.sqrt(mse + eps)

    return options.well_loss_penalty * rmse


def compute_seismic_loss(
    gen_ip_norm: torch.Tensor,
    real_seismic: torch.Tensor,
    vp_mean: torch.Tensor,
    physics_state: PhysicsState,
    dz_pixel: torch.Tensor | None = None,
    loss_fn: LossFn = LossFn.HUBER,
) -> torch.Tensor:
    """Calculate Geophysical Consistency Loss (Seismic Loss).

    This loss is computed on soft-RMS normalized signals to ensure scale
    invariance while maintaining robustness against low-variance patches.
    It combines a spatial point-wise loss (Huber) with a phase-sensitive
    correlation term.
    """
    from physics.seismic import calculate_synthetic_seismic

    if dz_pixel is None:
        dz_pixel = PhysicsConfig.DZ_PIXEL

    synth = calculate_synthetic_seismic(
        gen_ip_norm,
        vp_mean,
        dz_pixel,
        physics_state,
    )

    if synth.shape != real_seismic.shape:
        synth = F.interpolate(
            synth, size=(real_seismic.shape[2], real_seismic.shape[3])
        )

    # 1. Zero-mean the signals to remove DC bias
    synth_zero = synth - synth.mean(dim=(2, 3), keepdim=True)
    real_zero = real_seismic - real_seismic.mean(dim=(2, 3), keepdim=True)

    # 2. Soft-RMS Normalization (Safe floor to prevent division by tiny noise)
    # Using a larger epsilon (1e-4) to prevent gradient explosions on flat patches.
    # Specify dim=(2, 3) to compute RMS per-sample, not across the whole batch!
    eps_safe = 1e-4
    synth_rms = torch.sqrt(
        torch.mean(synth_zero**2, dim=(2, 3), keepdim=True) + eps_safe
    )
    real_rms = torch.sqrt(torch.mean(real_zero**2, dim=(2, 3), keepdim=True) + eps_safe)

    synth_norm = synth_zero / synth_rms
    real_norm = real_zero / real_rms

    # 3. Spatial Point-wise Loss
    if loss_fn == LossFn.HUBER:
        spatial_loss = F.huber_loss(synth_norm, real_norm)
    else:
        spatial_loss = F.mse_loss(synth_norm, real_norm)

    # 4. Phase-sensitive Correlation Loss (1 - Cosine Similarity)
    # This helps anchor the waveform phase regardless of amplitude mismatches.
    cos_sim = F.cosine_similarity(
        synth_zero.flatten(1), real_zero.flatten(1), dim=1
    ).mean()
    corr_loss = 1.0 - cos_sim

    # Weighted combination: spatial matching + phase anchoring
    return spatial_loss + 0.1 * corr_loss


def total_variation_loss(
    rock_physics: torch.Tensor,
    facies: torch.Tensor | None = None,
    eps: float = DomainConfig.EPSILON,
) -> torch.Tensor:
    """Compute intra-facies Total Variation (TV) loss for a 4D tensor."""
    diff_z = torch.abs(rock_physics[:, :, 1:, :] - rock_physics[:, :, :-1, :])
    diff_x = torch.abs(rock_physics[:, :, :, 1:] - rock_physics[:, :, :, :-1])

    if facies is None:
        return diff_z.mean() + diff_x.mean()

    # Usamos softmax para obter uma distribuição de probabilidade contínua e diferenciável
    probs = torch.softmax(facies, dim=1)

    # Máscara contínua: produto interno das probabilidades adjacentes.
    # Se os pixels tiverem a mesma distribuição, sim ~ 1.0. Se forem diferentes, sim ~ 0.0.
    sim_z = (probs[:, :, 1:, :] * probs[:, :, :-1, :]).sum(dim=1, keepdim=True)
    sim_x = (probs[:, :, :, 1:] * probs[:, :, :, :-1]).sum(dim=1, keepdim=True)

    # Multiplicamos a variação pela máscara suave e tiramos a média diretamente.
    # Isso evita a divisão por somas dinâmicas que desestabilizam os gradientes no DDP.
    tv_z = (diff_z * sim_z).mean()
    tv_x = (diff_x * sim_x).mean()

    return tv_z + tv_x


def compute_rock_physics_loss(
    fake: torch.Tensor,
    seismic_pyramid: dict[int, torch.Tensor],
    scale: int,
    options: TrainingOptions,
    physics_state: PhysicsState,
    eps: float = DomainConfig.EPSILON,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute rock physics loss."""

    def needs_phys() -> bool:
        return options.elastic_loss_penalty > 0 or options.seismic_loss_penalty > 0

    fake_rock_physics = fake[:, options.num_facies_channels :, ...]
    fake_facies = fake[:, : options.num_facies_channels, ...]

    phys = (
        physics_state.denormalize_rock_physics(fake_rock_physics)
        if needs_phys()
        else {}
    )

    tv_loss = DomainConfig.ZERO_SCALAR
    if options.tv_loss_penalty > 0:
        tv_unweighted = total_variation_loss(fake_rock_physics, fake_facies)
        tv_loss = options.tv_loss_penalty * tv_unweighted

    elastic_loss = DomainConfig.ZERO_SCALAR
    if (
        options.elastic_loss_penalty > 0
        and getattr(options, "use_ip", True)
        and getattr(options, "use_is", True)
        and getattr(options, "use_vpvs", True)
    ):
        log_ip = torch.log(phys["Ip"] + eps)
        log_is = torch.log(phys["Is"] + eps)
        log_vpvs_target = torch.log(phys["VP_VS"] + eps)

        elastic_loss = options.elastic_loss_penalty * F.mse_loss(
            log_ip - log_is, log_vpvs_target
        )

    seismic_loss = DomainConfig.ZERO_SCALAR
    if (
        options.seismic_loss_penalty > 0
        and seismic_pyramid.get(scale) is not None
        and getattr(options, "use_ip", True)
    ):
        vp_phys = phys["Ip"] / physics_state.rho_mean
        vp_mean = torch.mean(vp_phys).clamp(physics_state.vp_min, physics_state.vp_max)

        seismic_unweighted = compute_seismic_loss(
            fake_rock_physics[:, 0:1, ...],
            seismic_pyramid[scale],
            vp_mean,
            physics_state,
            dz_pixel=(
                physics_state.dz_pyramid[scale]
                if scale in range(len(physics_state.dz_pyramid))
                else PhysicsConfig.DZ_PIXEL
            ),
            loss_fn=LossFn.HUBER,
        )
        seismic_loss = options.seismic_loss_penalty * seismic_unweighted

    return tv_loss, elastic_loss, seismic_loss


def compute_reconstruction_loss(
    generator: torch.nn.Module,
    noise_amps: list[torch.Tensor],
    rec_noise: list[torch.Tensor],
    scale: int,
    real: torch.Tensor,
    rec_in: torch.Tensor,
    options: TrainingOptions,
    zero_scalar: torch.Tensor,
    current_epoch: int,
    rec_skip_epochs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute reconstruction (recovery) loss for given inputs."""
    if options.rec_facies_loss_penalty == 0 or current_epoch < rec_skip_epochs:
        return zero_scalar, zero_scalar

    rec = generator(
        rec_noise,
        noise_amps[: scale + 1],
        in_noise=rec_in,
        start_scale=scale,
        stop_scale=scale,
    )

    if options.use_rock_physics:
        fc = options.num_facies_channels
        num_rp = sum(
            [
                getattr(options, "use_ip", True),
                getattr(options, "use_is", True),
                getattr(options, "use_vpvs", True),
            ]
        )
        rec_loss_facies = options.rec_facies_loss_penalty * F.mse_loss(
            rec[:, :fc, ...], real[:, :fc, ...]
        )
        rec_loss_rp = options.rec_rock_physics_loss_penalty * F.huber_loss(
            rec[:, fc : fc + num_rp, ...], real[:, fc : fc + num_rp, ...]
        )
        return rec_loss_facies, rec_loss_rp
    else:
        rec_loss_facies = options.rec_facies_loss_penalty * F.mse_loss(rec, real)
        return rec_loss_facies, zero_scalar
