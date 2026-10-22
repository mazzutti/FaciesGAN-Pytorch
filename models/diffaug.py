"""Differentiable augmentation (DiffAugment) for discriminator inputs.

Reference: Zhao et al., "Differentiable Augmentation for Data-Efficient GAN
Training", NeurIPS 2020.

Every transform is differentiable with respect to its input, so the generator
still receives gradients through the augmented critic input. The same spatial
transform is applied to every channel of a sample, which keeps conditioning
channels (wells, masks, seismic) aligned with the facies channels.

Vertical flips are intentionally not offered: they would break the
stratigraphic (top/bottom) ordering of geological sections.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

DEFAULT_POLICY = "hflip,translation,cutout"
SUPPORTED_POLICIES = ("hflip", "translation", "cutout")


def diff_augment(
    x: torch.Tensor,
    policy: str = DEFAULT_POLICY,
    pad_value: float = 0.0,
    translation_ratio: float = 0.125,
    cutout_ratio: float = 0.5,
) -> torch.Tensor:
    """Apply the comma-separated ``policy`` to a ``(B, C, H, W)`` tensor.

    Parameters
    ----------
    x : torch.Tensor
        Batch of critic inputs (facies plus optional conditioning channels).
    policy : str
        Comma-separated subset of ``hflip``, ``translation`` and ``cutout``.
    pad_value : float
        Value used for exposed borders (translation) and erased regions
        (cutout). Use the data padding midpoint so the critic sees the same
        "empty" value the generator pads with.
    translation_ratio : float
        Maximum shift as a fraction of the spatial size.
    cutout_ratio : float
        Side of the erased square as a fraction of the spatial size.
    """
    if not policy:
        return x
    for name in policy.split(","):
        name = name.strip().lower()
        if not name:
            continue
        if name == "hflip":
            x = _rand_hflip(x)
        elif name == "translation":
            x = _rand_translation(x, translation_ratio, pad_value)
        elif name == "cutout":
            x = _rand_cutout(x, cutout_ratio, pad_value)
        else:
            raise ValueError(
                f"Unknown DiffAugment policy entry {name!r}; "
                f"supported: {', '.join(SUPPORTED_POLICIES)}"
            )
    return x


def _rand_hflip(x: torch.Tensor) -> torch.Tensor:
    """Mirror each sample horizontally with probability 0.5."""
    b = x.shape[0]
    flip = (torch.rand(b, device=x.device) < 0.5).view(b, 1, 1, 1)
    return torch.where(flip, x.flip(3), x)


def _rand_translation(x: torch.Tensor, ratio: float, pad_value: float) -> torch.Tensor:
    """Shift each sample by a random integer offset, filling with ``pad_value``."""
    b, _, h, w = x.shape
    sh = int(h * ratio + 0.5)
    sw = int(w * ratio + 0.5)
    if sh == 0 and sw == 0:
        return x
    dev = x.device
    tx = torch.randint(-sh, sh + 1, (b, 1, 1), device=dev)
    ty = torch.randint(-sw, sw + 1, (b, 1, 1), device=dev)
    gb, gx, gy = torch.meshgrid(
        torch.arange(b, device=dev),
        torch.arange(h, device=dev),
        torch.arange(w, device=dev),
        indexing="ij",
    )
    gx = torch.clamp(gx + tx + 1, 0, h + 1)
    gy = torch.clamp(gy + ty + 1, 0, w + 1)
    x_pad = F.pad(x, (1, 1, 1, 1), value=pad_value)
    out = x_pad.permute(0, 2, 3, 1)[gb, gx, gy]
    return out.permute(0, 3, 1, 2).contiguous()


def _rand_cutout(x: torch.Tensor, ratio: float, pad_value: float) -> torch.Tensor:
    """Erase one random square per sample, filling with ``pad_value``."""
    b, _, h, w = x.shape
    ch = int(h * ratio + 0.5)
    cw = int(w * ratio + 0.5)
    if ch == 0 or cw == 0:
        return x
    dev = x.device
    ox = torch.randint(0, h + (1 - ch % 2), (b, 1, 1), device=dev)
    oy = torch.randint(0, w + (1 - cw % 2), (b, 1, 1), device=dev)
    gb, gx, gy = torch.meshgrid(
        torch.arange(b, device=dev),
        torch.arange(ch, device=dev),
        torch.arange(cw, device=dev),
        indexing="ij",
    )
    gx = torch.clamp(gx + ox - ch // 2, 0, h - 1)
    gy = torch.clamp(gy + oy - cw // 2, 0, w - 1)
    mask = torch.ones(b, h, w, dtype=x.dtype, device=dev)
    mask[gb, gx, gy] = 0
    mask = mask.unsqueeze(1)
    if pad_value == 0.0:
        return x * mask
    return x * mask + pad_value * (1 - mask)
