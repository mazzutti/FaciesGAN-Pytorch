"""Typed dataclasses for transporting per-scale training metrics.

The dataclasses in this module are used to return tensor-valued losses and
metrics from model optimization routines. Fields intentionally remain as
``torch.Tensor`` scalars during the forward/backward pass so autograd is
preserved; callers should convert to Python floats (``.item()``) when logging
or writing to external sinks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from log import Any


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

        Returns
        -------
        dict[str, float]
            Dictionary mapping metric names to their tensor values.
        """
        return {
            "d_total": self.total.item(),  # type: ignore
            "d_real": self.real.item(),  # type: ignore
            "d_fake": self.fake.item(),  # type: ignore
            "d_gp": self.gp.item(),  # type: ignore
        }

    def as_tuple(self) -> tuple[torch.Tensor, ...]:
        """Return the metric values as a tuple in a fixed order.

        Returns
        -------
        tuple[torch.Tensor, ...]
            Tuple of metric values in the order:
            (total, real, fake, gp).
        """
        return (self.total, self.real, self.fake, self.gp)


@dataclass
class GeneratorMetrics:
    """Per-scale generator metric container.

    Fields are scalar ``torch.Tensor`` values for adversarial (``fake``),
    reconstruction (``rec``), well-constraint (``well``), diversity (``div``)
    and aggregated total loss (``total``).
    """

    total: torch.Tensor
    fake: torch.Tensor
    facies_rec: torch.Tensor
    well: torch.Tensor
    div: torch.Tensor
    rec_rock_physics: torch.Tensor
    tv: torch.Tensor
    elastic: torch.Tensor
    physics: torch.Tensor

    def as_dict(self) -> dict[str, float]:
        """Return the metric values as a dictionary.

        Returns
        -------
        dict[str, float]
            Dictionary mapping metric names to their tensor values.
        """
        return {
            "g_total": self.total.item(),  # type: ignore
            "g_fake": self.fake.item(),  # type: ignore
            "g_rec_facies": self.facies_rec.item(),  # type: ignore
            "g_well": self.well.item(),  # type: ignore
            "g_div": self.div.item(),  # type: ignore
            "g_rec_rock_physics": self.rec_rock_physics.item(),  # type: ignore
            "g_tv": self.tv.item(),  # type: ignore
            "g_elastic": self.elastic.item(),  # type: ignore
            "g_physics": self.physics.item(),  # type: ignore
        }

    def as_tuple(self) -> tuple[torch.Tensor, ...]:
        """Return the metric values as a list in a fixed order.

        Returns
        -------
        tuple[torch.Tensor, ...]
            Tuple of metric values in the order:
            (total, fake, facies_rec, well, div, rp, elastic, physics).
        """
        return (
            self.total,
            self.fake,
            self.facies_rec,
            self.well,
            self.div,
            self.rec_rock_physics,
            self.tv,
            self.elastic,
            self.physics,
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
                int(scale): GeneratorMetrics(*metrics)
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
