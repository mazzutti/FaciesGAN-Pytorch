"""PyTorch multiscale Generator implementation for FaciesGAN.

This module provides a concrete `Generator` class implementing the
multiscale progressive generator used by the project. The class
encapsulates per-scale construction, forward synthesis, and utility
helpers (quantization, SPADE-based coarse generation).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Self, cast

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt_utils

from config import DomainConfig
from enums import DeviceType

from . import utils
from .custom_layer import ConvBlock, FaciesQuantization, SPADEGenerator


class Generator(nn.Module):
    """Multiscale progressive generator for FaciesGAN.

    The generator is composed of a sequence of per-scale modules appended
    with ``create_scale``. It synthesizes images from a list of per-scale
    noise tensors ``z`` and amplitude scalars ``amp``. The network supports
    optional conditioning channels (wells/seismic) concatenated to the
    noise tensor.

    Parameters
    ----------
    num_layer : int
        Number of convolutional layers in each scale block.
    kernel_size : int
        Size of convolutional kernels.
    padding_size : int
        Padding size for convolutions.
    input_channels : int
        Number of input channels (noise + conditioning channels).

    Attributes
    ----------
    gens : nn.ModuleList
        List of per-scale generator modules (SPADE or ConvBlock stacks).
    zero_padding : int
        Padding applied per side to keep spatial alignment across scales.
    full_zero_padding : int
        Total padding applied (2 * zero_padding) used to compute output sizes.
    spade_scales : set[int]
        Set of scales that use SPADE-based generation (usually coarse scales).
    color_quantizer : ColorQuantization
        Module used to quantize outputs to a small palette of colors.

    Methods
    -------
    __call__(z, amp, in_noise=None, start_scale=0, stop_scale
    ) -> torch.Tensor
        Calls the generator's forward method.
    eval() -> Self
        Sets the module in evaluation mode.
    forward(z, amp, in_noise=None, start_scale=0, stop_scale=None) -> torch.Tensor
        Forward pass through the multiscale generator.
    """

    def __init__(
        self,
        num_layer: int,
        kernel_size: int,
        padding_size: int,
        padding_value: float,
        input_channels: int,
        output_channels: int = 3,
        num_facies: int = DomainConfig.NUM_FACIES_CHANNELS,
        normalization_range: tuple[float, ...] = (-1.0, 1.0),
        device: torch.device = torch.device(DeviceType.CPU),
    ) -> None:
        """Initialize the multiscale Generator.

        Parameters
        ----------
        num_layer : int
            Number of convolutional layers per scale block.
        kernel_size : int
            Convolution kernel size.
        padding_size : int
            Padding size used for convolutions.
        padding_value : float
            Padding value used for convolutions.
        input_channels : int
            Number of input channels (noise plus optional conditioning).
        output_channels : int
            Number of output color channels.
        device : torch.device
            Device for computation.
        num_facies : int, optional
            Number of facies output channels (default DomainConfig.NUM_FACIES_CHANNELS).
        """
        # Initialize generator configuration used throughout the class.
        self.spade_scales: set[int] = set()

        # Callbacks for monitoring internal state (e.g., torch.compile
        # first-use compile progress (label -> progress tick).
        self.compile_progress_callback: Callable[[str], None] | None = None
        self._compile_progress_seen: set[str] = set()

        # Gradient (activation) checkpointing flag.
        self.use_gradient_checkpointing: bool = False

        # stored configuration fields used in the generator
        self.num_layer = num_layer

        # convolution parameters used in the generator
        self.kernel_size = kernel_size

        # padding size used in convolutions
        self.padding_size = padding_size

        # padding value used in convolutions
        self.padding_value = padding_value

        # channel counts used in the generator
        self.input_channels = input_channels

        # output channel count (e.g., RGB)
        self.output_channels = output_channels

        # conditional channel count (e.g., segmentation map)
        self.cond_channels = self.input_channels - self.output_channels

        # Number of pure facies output channels before any extra channels.
        self.num_facies = num_facies

        # Normalization range for rock physics channels.
        self.normalization_range = normalization_range
        self.norm_min = float(min(normalization_range))
        self.norm_max = float(max(normalization_range))

        # flag indicating whether conditional channels are used
        self.has_cond_channels = self.cond_channels > 0

        # zero padding values used to align spatial sizes across scales
        self.zero_padding = self.num_layer * (math.floor(self.kernel_size / 2))

        # full padding applied to input tensors at each scale
        self.full_zero_padding = 2 * self.zero_padding

        # Initialize the nn.Module base.
        nn.Module.__init__(self)

        # Use nn.ModuleList so that .train()/.eval(), .to(device),
        # and .state_dict() propagate to all per-scale gen blocks.
        self.gens = nn.ModuleList()  # type: ignore[assignment]

        # Color quantization layer (framework-specific)
        self.color_quantizer = FaciesQuantization(temperature=0.5, device=device)

        # Residual add + clamp fused into a single callable so that
        # torch.compile can merge them into one Inductor kernel,
        # eliminating a separate clamp kernel launch per scale.
        self._residual_clamp: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
            self._residual_clamp_method
        )

    def _residual_clamp_method(
        self, gen_out: torch.Tensor, prev: torch.Tensor
    ) -> torch.Tensor:
        """Instance method wrapper for residual_clamp_fn to support torch.compile."""
        return Generator.residual_clamp_fn(
            gen_out, prev, self.num_facies, self.normalization_range
        )

    def _mark_compile_progress(self, label: str) -> None:
        """Emit a one-time compile progress event for ``label``."""
        if label in self._compile_progress_seen:
            return
        self._compile_progress_seen.add(label)
        cb = self.compile_progress_callback
        if cb is not None:
            cb(label)

    @staticmethod
    def residual_clamp_fn(
        gen_out: torch.Tensor,
        prev: torch.Tensor,
        num_facies: int,
        normalization_range: tuple[float, ...],
    ) -> torch.Tensor:
        """Add residual and clamp appropriately for facies and rock physics.

        Facies (first num_facies channels): Unconstrained addition (logits).
        Rock Physics (remaining channels): Addition clamped to normalization_range.
        """
        if gen_out.shape[1] <= num_facies:
            # Only facies channels present (or no rock physics)
            return gen_out + prev

        norm_min = float(min(normalization_range))
        norm_max = float(max(normalization_range))

        # Split into facies and rock physics
        facies_out = gen_out[:, :num_facies, ...]
        facies_prev = prev[:, :num_facies, ...]
        facies_next = facies_out + facies_prev

        rp_out = gen_out[:, num_facies:, ...]
        rp_prev = prev[:, num_facies:, ...]
        rp_next = (rp_prev + rp_out).clamp(norm_min, norm_max)

        return torch.cat([facies_next, rp_next], dim=1)

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Call the generator's forward method.

        Parameters
        ----------
        *args : Any
            Positional arguments for the `forward` method.
        **kwargs : Any
            Keyword arguments for the `forward` method.

        Returns
        -------
        torch.Tensor
            Output of the `forward` method.
        """
        return nn.Module.__call__(self, *args, **kwargs)

    def eval(self) -> Self:
        """Set the module in evaluation mode.

        Returns
        -------
        Self
            The generator instance in evaluation mode.
        """
        return cast(Self, nn.Module.eval(self))

    def forward(
        self,
        z: list[torch.Tensor],
        amp: list[float],
        in_noise: torch.Tensor | None = None,
        start_scale: int = 0,
        stop_scale: int | None = None,
    ) -> torch.Tensor:
        """Generate facies through progressive pyramid synthesis.

        Parameters
        ----------
        z : list[torch.Tensor]
            Noise tensors for each pyramid scale.
        amp : list[float]
            Noise amplitudes for each scale.
        in_noise : torch.Tensor | None, optional
            Initial facies tensor to start from. If None, starts with zeros.
            Defaults to None.
        start_scale : int, optional
            Pyramid scale to start generation from. Defaults to 0.
        stop_scale : int | None, optional
            Final pyramid scale (inclusive). If None, uses all available scales.
            Defaults to None.

        Returns
        -------
        torch.Tensor
            Generated facies tensor at the finest requested scale.
        """
        if in_noise is None:
            channels = self.output_channels
            batch_size = z[start_scale].shape[0]
            height, width = tuple(
                dim - self.full_zero_padding for dim in z[start_scale].shape[2:]
            )
            device = z[start_scale].device
            # Allocate in channels_last on CUDA to match conv weight layout
            # and avoid implicit format conversions during forward.
            out_facie: torch.Tensor = torch.zeros(
                (batch_size, channels, height, width),
                device=device,
            )
            if device.type == DeviceType.CUDA:
                out_facie = out_facie.to(memory_format=torch.channels_last)
        else:
            out_facie = in_noise

        stop_scale = stop_scale if stop_scale is not None else len(self.gens) - 1

        for index in range(start_scale, stop_scale + 1):
            self._mark_compile_progress(f"gen_scale_{index}")

            out_facie = utils.interpolate(
                out_facie,
                (
                    z[index].shape[2] - self.full_zero_padding,
                    z[index].shape[3] - self.full_zero_padding,
                ),
            )

            # Number of input channels to receive residual addition.
            # This matches the number of channels in the generated facies/RP output.
            n_in = self.output_channels

            # Build z_in: combine noise (scaled by amp) with padded previous scale output.
            # Conditioning channels (if any) are appended without scaling or residual.
            # Replace F.pad with manual slice assignment to support torch.Tensor padding_value
            p = self.zero_padding
            base_out = out_facie[:, :n_in, ...]
            if p > 0:
                padded_out = torch.empty(
                    (
                        base_out.shape[0],
                        base_out.shape[1],
                        base_out.shape[2] + 2 * p,
                        base_out.shape[3] + 2 * p,
                    ),
                    dtype=base_out.dtype,
                    device=base_out.device,
                ).fill_(self.padding_value)
                padded_out[..., p:-p, p:-p] = base_out
            else:
                padded_out = base_out

            z_in = amp[index] * z[index][:, :n_in, ...] + padded_out
            if self.cond_channels > 0:
                z_in = torch.cat([z_in, z[index][:, n_in:, ...]], dim=1)

            if self.use_gradient_checkpointing and self.training and z_in.requires_grad:
                # Recompute this block's activations during backward.
                # use_reentrant=False is the recommended mode (no
                # nesting caveats, compatible with compiled models).
                gen_out = cast(
                    torch.Tensor,
                    ckpt_utils.checkpoint(  # type: ignore[misc]
                        self.gens[index], z_in, use_reentrant=False
                    ),
                )
            else:
                gen_out = self.gens[index](z_in)

            # Fused residual add + clamp to [-1, 1].
            # When compiled, Inductor merges the add and clamp into a
            # single pointwise kernel, saving one kernel launch per scale.
            self._mark_compile_progress("residual_clamp")
            out_facie = self._residual_clamp(gen_out, out_facie)

        if self.num_facies < out_facie.shape[1]:
            self._mark_compile_progress("facies_quantizer")
            facies = out_facie[:, : self.num_facies, ...]
            imp = out_facie[:, self.num_facies :, ...]
            facies_q = self.color_quantizer(facies)
            out_facie = torch.cat([facies_q, imp], dim=1)
        else:
            self._mark_compile_progress("facies_quantizer")
            out_facie = self.color_quantizer(out_facie)

        return out_facie  # type: ignore[return-value]

    def create_scale(
        self, scale: int, num_features: int, min_num_features: int
    ) -> None:
        """Create and append a new scale block to the generator pyramid.

        Constructs a ConvBlock sequence with progressively decreasing channel
        counts from num_features down to min_num_features.

        At scale 0, uses SPADEGenerator for noise-modulated generation.
        At higher scales, uses standard ConvBlock architecture.

        Parameters
        ----------
        scale : int
            Pyramid scale index.
        num_features : int
            Number of features in the first convolutional layer.
        min_num_features : int
            Minimum number of features (floor for channel reduction).
        """

        if scale == 0:
            # Use SPADE-based generator at the coarsest scale
            # This allows noise to modulate features via learned gamma/beta
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
            # Standard ConvBlock-based generator for finer scales
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
