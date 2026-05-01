"""Generator network and supporting modules for FaciesGAN.

This module provides the multi-scale ``Generator`` used to synthesize
facies images from per-scale noise tensors, along with a simple
``FaciesQuantization`` module used to snap outputs to one-hot vectors.
"""

import math
from typing import Callable, Self, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt_utils

from . import utils
from .custom_layer import ConvBlock, FaciesQuantization, SPADEGenerator


class Generator(nn.Module):
    """Multi-scale progressive generator for FaciesGAN.

    The generator is composed of a sequence of per-scale modules appended
    with ``create_scale``. It synthesizes images from a list of per-scale
    noise tensors ``z`` and amplitude scalars ``amp``. The network supports
    optional conditioning channels (wells/seismic) concatenated to the
    noise tensor.
    """

    def __init__(
        self,
        num_layer: int,
        kernel_size: int,
        padding_size: int,
        input_channels: int,
        output_channels: int = 4,
        num_facies_classes: int = 4,
        noise_channels: int | None = None,
    ) -> None:
        """Initialize the multi-scale Generator."""
        super().__init__()

        # Base configuration logic (inlined)
        self.num_layer = num_layer
        self.kernel_size = kernel_size
        self.padding_size = padding_size
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.noise_channels = (
            noise_channels if noise_channels is not None else output_channels
        )
        self.cond_channels = self.input_channels - self.noise_channels
        self.has_cond_channels = self.cond_channels > 0
        self.zero_padding = self.num_layer * (math.floor(self.kernel_size / 2))
        self.full_zero_padding = self.zero_padding * 2
        self.num_facies_classes = num_facies_classes

        # Initialize the facies quantizer with the number of facies classes.
        # In multi-channel mode (Rock Physics), quantization is only applied
        # to the facies subset (first N channels).
        self.facies_quantizer = FaciesQuantization(num_classes=self.num_facies_classes)

        self.spade_scales: set[int] = set()
        self.use_gradient_checkpointing: bool = False

        # Framework-specific containers
        self.gens = nn.ModuleList()

        # Residual add + clamp fused into a single callable
        self._residual_clamp: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
            Generator.residual_clamp_fn
        )

    @staticmethod
    def residual_clamp_fn(gen_out: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        """Add residual and clamp to [-1, 1] in one pass."""
        return (gen_out + prev).clamp(-1, 1)

    def eval(self) -> Self:
        """Set the module in evaluation mode."""
        return super().eval()

    def forward(
        self,
        z: list[torch.Tensor],
        amp: list[float],
        in_noise: torch.Tensor | None = None,
        start_scale: int = 0,
        stop_scale: int | None = None,
    ) -> torch.Tensor:
        """Generate facies through progressive pyramid synthesis."""
        if in_noise is None:
            channels = self.output_channels
            batch_size = z[start_scale].shape[0]
            height, width = tuple(
                dim - self.full_zero_padding for dim in z[start_scale].shape[2:]
            )
            device = z[start_scale].device
            out_facie: torch.Tensor = torch.zeros(
                (batch_size, channels, height, width),
                device=device,
            )
            if device.type == "cuda":
                out_facie = out_facie.to(memory_format=torch.channels_last)
        else:
            out_facie = in_noise

        stop_scale = stop_scale if stop_scale is not None else len(self.gens) - 1

        noise_C: int = self.noise_channels
        cond_C: int = self.input_channels - noise_C

        for index in range(start_scale, stop_scale + 1):
            # 1. Upsample the previous scale's output to match current spatial dimensions
            target_h = z[index].shape[2] - self.full_zero_padding
            target_w = z[index].shape[3] - self.full_zero_padding
            if target_h <= 0 or target_w <= 0:
                raise ValueError(
                    f"Invalid interpolation size ({target_h}, {target_w}) at scale {index}. "
                    f"Noise shape: {z[index].shape}, full_zero_padding: {self.full_zero_padding}"
                )
            out_facie = utils.interpolate(
                out_facie,
                (target_h, target_w),
            )

            # 2. Inject noise and conditioning (Wells/Seismic)
            if cond_C > 0:
                # Conditioning mode: Concatenate noise and previous output (repeated)
                # to the current noise tensor.
                padded_facie = F.pad(
                    out_facie[:, :noise_C, ...], [self.zero_padding] * 4, value=0
                )
                repeats = (cond_C + noise_C - 1) // noise_C
                padded_facie = padded_facie.repeat(1, repeats, 1, 1)[:, :cond_C, :, :]
                z_in = torch.cat(
                    [
                        amp[index] * z[index][:, :noise_C, :, :],
                        z[index][:, noise_C:, :, :] + padded_facie,
                    ],
                    dim=1,
                )
            else:
                # Standard mode: Simple residual injection
                facie_for_residual = out_facie[:, :noise_C, ...]
                z_in = amp[index] * z[index] + F.pad(
                    facie_for_residual, [self.zero_padding] * 4, value=0
                )

            # 3. Forward through the current scale's ConvBlock
            if self.use_gradient_checkpointing and self.training and z_in.requires_grad:
                gen_out: torch.Tensor = cast(
                    torch.Tensor,
                    ckpt_utils.checkpoint(self.gens[index], z_in, use_reentrant=False),  # type: ignore
                )
            else:
                gen_out = self.gens[index](z_in)

            # 4. Add residual contribution from the lower resolution
            out_facie = self._residual_clamp(gen_out, out_facie)

        # Final Polish: Split channels and apply one-hot quantization to facies only
        orig_C = getattr(self, "num_facies_classes", None)
        if orig_C is not None and orig_C < out_facie.shape[1]:
            # Multi-channel pipeline: [Facies(orig_C) | Rock Physics(rest)]
            facies = out_facie[:, :orig_C, ...]
            rp = out_facie[:, orig_C:, ...]
            facies_q = self.facies_quantizer(facies)
            out_facie = torch.cat([facies_q, rp], dim=1)
        else:
            # Standard facies-only pipeline
            out_facie = self.facies_quantizer(out_facie)

        return out_facie

    def create_scale(
        self, scale: int, num_features: int, min_num_features: int
    ) -> None:
        """Create and append a new scale block to the generator pyramid."""
        if scale == 0:
            spade_gen = SPADEGenerator(
                num_layer=self.num_layer,
                kernel_size=self.kernel_size,
                padding_size=self.padding_size,
                num_features=num_features,
                min_num_features=min_num_features,
                output_channels=self.output_channels,
                input_channels=self.input_channels,
            )
            self.gens.append(spade_gen)
            self.spade_scales.add(scale)
        else:
            head = ConvBlock(
                self.input_channels,
                num_features,
                self.kernel_size,
                self.padding_size,
                1,
            )
            body = nn.Sequential()

            block_features = min_num_features
            for i in range(self.num_layer - 2):
                block_features = int(num_features / pow(2, (i + 1)))
                block = ConvBlock(
                    max(2 * block_features, min_num_features),
                    max(block_features, min_num_features),
                    self.kernel_size,
                    self.padding_size,
                    1,
                )
                body.add_module(f"block{i + 1}", block)

            tail = nn.Sequential(
                nn.Conv2d(
                    max(block_features, min_num_features),
                    self.output_channels,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=self.padding_size,
                ),
                nn.Tanh(),
            )

            self.gens.append(nn.Sequential(head, body, tail))
