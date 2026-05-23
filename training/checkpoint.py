"""Checkpoint representations for FaciesGAN training.

This module provides structured containers for saving and restoring training
state across different pyramid scales.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch

from device import device_manager


@dataclass
class ScaleCheckpoint:
    """State dictionaries for a single pyramid scale.

    Contains the weights for the scale-specific generator and discriminator
    blocks, along with their optimizer and scheduler states.
    """

    generator: dict[str, Any]
    discriminator: dict[str, Any]
    opt_g: dict[str, Any]
    opt_d: dict[str, Any]
    sch_g: dict[str, Any]
    sch_d: dict[str, Any]


def _default_scales() -> Dict[int, ScaleCheckpoint]:
    return {}


def _default_rec_noise() -> List[torch.Tensor]:
    return []


def _default_rng_state() -> Dict[str, Any]:
    return {}


def _default_seen_indices() -> List[Any]:
    return []


@dataclass
class Checkpoint:
    """Global training state checkpoint.

    Encapsulates the full state required to resume training, including
    global counters (epoch, batch), adaptive parameters (noise amplitudes),
    and per-scale model/optimizer states.
    """

    epoch: int
    batch_id: int
    noise_amps: List[torch.Tensor]
    disc_step_counter: int
    extra_disc_step_counter: int
    scales: Dict[int, ScaleCheckpoint] = field(default_factory=_default_scales)
    grad_scaler_g: Dict[str, Any] | None = None
    rec_noise: List[torch.Tensor] = field(default_factory=_default_rec_noise)
    rng_state: Dict[str, Any] = field(default_factory=_default_rng_state)
    seen_indices: List[Any] = field(default_factory=_default_seen_indices)
    last_gp_value: Dict[int, torch.Tensor] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the checkpoint to a plain dictionary for torch.save."""
        return {
            "epoch": self.epoch,
            "batch_id": self.batch_id,
            "noise_amps": self.noise_amps,
            "disc_step_counter": self.disc_step_counter,
            "extra_disc_step_counter": self.extra_disc_step_counter,
            "grad_scaler_g": self.grad_scaler_g,
            "rec_noise": self.rec_noise,
            "rng_state": self.rng_state,
            "seen_indices": self.seen_indices,
            "last_gp_value": {s: v.cpu() for s, v in self.last_gp_value.items()},
            "scales": {
                s: {
                    "generator": sc.generator,
                    "discriminator": sc.discriminator,
                    "opt_g": sc.opt_g,
                    "opt_d": sc.opt_d,
                    "sch_g": sc.sch_g,
                    "sch_d": sc.sch_d,
                }
                for s, sc in self.scales.items()
            },
        }

    @classmethod
    def load(cls, path: str) -> Checkpoint:
        """Load a checkpoint from a file, handling both new and legacy formats."""
        # Load directly to the managed device for faster restoration.
        data = torch.load(path, map_location=device_manager.device, weights_only=False)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Checkpoint:
        """Create a Checkpoint instance from a dictionary, handling legacy keys."""
        if "scales" in data:
            # New format
            scales_data = data["scales"]
            scales: dict[int, ScaleCheckpoint] = {
                int(s): ScaleCheckpoint(
                    generator=sd["generator"],
                    discriminator=sd["discriminator"],
                    opt_g=sd["opt_g"],
                    opt_d=sd["opt_d"],
                    sch_g=sd["sch_g"],
                    sch_d=sd["sch_d"],
                )
                for s, sd in scales_data.items()
            }
            return cls(
                epoch=data["epoch"],
                batch_id=data["batch_id"],
                noise_amps=data["noise_amps"],
                disc_step_counter=data["disc_step_counter"],
                extra_disc_step_counter=data["extra_disc_step_counter"],
                grad_scaler_g=data.get("grad_scaler_g"),
                rec_noise=data.get("rec_noise", []),
                rng_state=data.get("rng_state", {}),
                seen_indices=data.get("seen_indices", []),
                last_gp_value={int(s): v for s, v in data.get("last_gp_value", {}).items()},
                scales=scales,
            )

        # Legacy format conversion
        disc_states = data.get("discriminator_states", {})
        gen_opts = data.get("generator_optimizers", {})
        disc_opts = data.get("discriminator_optimizers", {})
        gen_schs = data.get("generator_schedulers", {})
        disc_schs = data.get("discriminator_schedulers", {})

        # Note: legacy format might have a single generator_state_dict if it was pre-multiscale
        # or it might have them indexed. Here we try to reconstruct per-scale states.
        all_scales = (
            set(disc_states.keys()) | set(gen_opts.keys()) | set(disc_opts.keys())
        )

        scales: dict[int, ScaleCheckpoint] = {}
        for s in all_scales:
            scales[int(s)] = ScaleCheckpoint(
                generator=data.get(
                    "generator_state_dict", {}
                ),  # Legacy usually had one shared G
                discriminator=disc_states.get(s, {}),
                opt_g=gen_opts.get(s, {}),
                opt_d=disc_opts.get(s, {}),
                sch_g=gen_schs.get(s, {}),
                sch_d=disc_schs.get(s, {}),
            )

        return cls(
            epoch=data.get("epoch", 0),
            batch_id=data.get("batch_id", 0),
            noise_amps=data.get("noise_amps", []),
            disc_step_counter=data.get("disc_step_counter", 0),
            extra_disc_step_counter=data.get("extra_disc_step_counter", 0),
            grad_scaler_g=data.get("grad_scaler_g"),
            rec_noise=data.get("rec_noise", []),
            rng_state=data.get("rng_state", {}),
            seen_indices=data.get("seen_indices", []),
            last_gp_value={int(s): v for s, v in data.get("last_gp_value", {}).items()},
            scales=scales,
        )
