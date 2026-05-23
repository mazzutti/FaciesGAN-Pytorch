"""Neural interpolation utilities and compact Residual MLP architecture.

This module implements the ``NeuralSmoother`` renderer which evaluates a
per-coordinate ResidualMLP to produce smoothed facies images, plus a small
training helper and the ``ResidualMLP`` model used for coordinate-based
interpolation. Utility helpers such as ``get_mgrid`` and a lightweight
``FourierFeatureTransform`` are also provided.
"""

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from apex_utils import FusedLayerNorm
from config import DomainConfig
from device import device_manager
from interpolators.base import BaseInterpolator
from interpolators.color_encoder import ColorEncoder
from interpolators.config import InterpolatorConfig
from typedefs import FileLike

# Module logger
logger = logging.getLogger(__name__)


def _load_image(image_path: FileLike) -> np.ndarray:
    import datasets.utils as data_utils

    return data_utils.load_image(image_path)


def get_mgrid(height: int, width: int) -> torch.Tensor:
    """Create a flattened meshgrid of normalized coordinates in [-1, 1].

    This is the module-level variant extracted from `NeuralSmoother.get_mgrid`.
    Keeping a top-level function makes it easier for other modules to import
    and reuse the grid logic without constructing a `NeuralSmoother`.
    """
    tensors = (
        torch.linspace(-1, 1, steps=height),
        torch.linspace(-1, 1, steps=width),
    )
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
    return mgrid.reshape((-1, 2))


class NeuralSmoother(BaseInterpolator):
    """Encapsulates utility helpers and the main training/rendering engine.

    The original module-level functions `set_seed`, `get_mgrid` and
    `train` have been moved here. Thin module-level wrappers
    below keep API compatibility.
    """

    def __init__(
        self,
        model_path: Path,
        config: InterpolatorConfig,
    ) -> None:
        """Initialize a NeuralSmoother instance.

        Parameters
        ----------
        model_path : Path
                Filesystem path to a model checkpoint. If the file exists it will be
                loaded and the model weights restored. If the path does not point to
                an existing checkpoint the current implementation raises
                ``FileNotFoundError`` and training must be performed separately.
        config : InterpolatorConfig
                Configuration object containing interpolator parameters such as
                ``scale``, ``upsample`` and ``chunk_size`` which influence model
                construction and inference behavior.

        Notes
        -----
        - The constructor resolves a runtime device and constructs a
            :class:`ResidualMLP` moved to that device.
        - Model compilation via ``torch.compile`` is attempted in
            :meth:`_compile_model` where supported; failures are logged and
            execution continues with the uncompiled model.
        - Side effects: sets ``self.device``, ``self.num_classes`` and
            ``self.model``, and attempts to load weights from ``model_path``.
        """
        super().__init__(config)
        # resolved device for the instance (CUDA/CPU)
        self.device: torch.device = device_manager.device
        self.num_classes: int = int(config.num_classes or DomainConfig.NUM_FACIES)
        self.model = ResidualMLP(
            num_classes=self.num_classes,
            scale=config.scale,
        ).to(self.device)
        self._load_model(model_path)

    @staticmethod
    def _state_from_checkpoint(state: Mapping[str, Any]) -> dict[str, Any] | None:
        """Normalize common checkpoint shapes into a dict[str, Any].

        This is intentionally short and permissive: we accept Mapping checkpoints
        that contain `model_state` or `state_dict`, mapping-like state dicts
        (name->tensor), or any iterable convertible to `dict()`.
        """
        try:
            ms = state.get("model_state", state.get("state_dict", state))
            normalized = {str(k): v for k, v in dict(ms).items()}
            return normalized
        except (AttributeError, TypeError, ValueError):
            return None

    def _compile_model(self) -> None:
        """Attempt to JIT/compile the model when supported."""
        try:
            self.model = torch.compile(  # pyright: ignore
                self.model,
                mode="reduce-overhead",
                fullgraph=True,
                dynamic=True,
            )
            logger.debug("Model compiled with torch.compile()")
        except (AttributeError, RuntimeError, TypeError):
            logger.debug("torch.compile() not available or failed; continuing")

    def _load_model(self, model_path: Path) -> None:
        """Orchestrate model compilation, optional restore, and optimizer setup.

        This method delegates detailed work to small helpers to keep the
        responsibilities clear and testable.
        """
        self._compile_model()
        if model_path.exists():
            state = torch.load(
                str(model_path), map_location=self.device, weights_only=False
            )
            ms: dict[str, Any] = cast(
                dict[str, Any], self._state_from_checkpoint(state)
            )

            # Normalize checkpoint keys for cases where the model was
            # compiled with torch.compile(). Compiled modules are wrapped
            # (OptimizedModule) and their state dict keys may be prefixed
            # with "_orig_mod.". Conversely, checkpoints saved from the
            # original model won't have that prefix. Detect the pattern
            # and rewrite keys to match `self.model`'s expected keys.
            model_keys = set(self.model.state_dict().keys())  # type: ignore
            ck_keys = set(ms.keys())

            def any_prefixed(keys: set[str], any_prefix: str) -> bool:
                return any(k.startswith(any_prefix) for k in keys)

            prefix = "_orig_mod."
            # If model expects prefixed keys but checkpoint doesn't, add prefix
            if any_prefixed(model_keys, prefix) and not any_prefixed(ck_keys, prefix):
                ms = {f"{prefix}{k}": v for k, v in ms.items()}
                print(
                    "Adjusted checkpoint keys: added _orig_mod. prefix to match compiled model"
                )
            # If checkpoint has prefix but model does not, strip it
            elif any_prefixed(ck_keys, prefix) and not any_prefixed(model_keys, prefix):
                ms = {
                    (k[len(prefix) :] if k.startswith(prefix) else k): v
                    for k, v in ms.items()
                }
                print(
                    "Adjusted checkpoint keys: removed _orig_mod. prefix to match model"
                )

            try:
                self.model.load_state_dict(ms)  # type: ignore
            except RuntimeError as exc:
                # Fall back to non-strict load to allow minor key mismatches
                logger.warning(
                    "Strict load_state_dict failed: %s; retrying with strict=False", exc
                )
                self.model.load_state_dict(ms, strict=False)  # type: ignore[arg-type]
            print(
                f"Loaded model checkpoint from {model_path}; skipping training."
            )
        else:
            raise FileNotFoundError("No checkpoint found; training model from scratch.")

    @classmethod
    def train_on_image(
        cls,
        image_path: Path,
        out_model_path: Path,
        config: "InterpolatorConfig | None" = None,
        epochs: int = 2000,
        lr: float = 3e-4,
    ) -> None:
        """Train a NeuralSmoother on a single facies PNG and save the checkpoint.

        The model is trained to predict the palette-class label for every
        (x, y) coordinate in the image.  After training the model state dict
        is saved to *out_model_path* so that :class:`NeuralSmoother` can load
        it later via ``_load_model``.

        Parameters
        ----------
        image_path : Path
            Path to the input PNG image (e.g. a facies crossline).
        out_model_path : Path
            Destination ``.pt`` file for the trained model state dict.
        config : InterpolatorConfig, optional
            Interpolator configuration controlling geometry, scale, etc.
            Defaults to ``InterpolatorConfig()``.
        epochs : int
            Number of gradient-descent steps.
        lr : float
            Adam learning rate.
        """
        if config is None:
            config = InterpolatorConfig()

        device = device_manager.device
        model = ResidualMLP(
            num_classes=config.num_classes,
            scale=config.scale,
        ).to(device)

        native_h, native_w = config.geometry

        img_np = _load_image(image_path)  # (H, W, 3) float32 [0,1]
        encoder = ColorEncoder(img_np)

        # Build coordinate grid for the native resolution
        coords = get_mgrid(height=native_h, width=native_w).to(device)  # (H*W, 2)

        # Ground-truth class labels (H*W,)
        img_tensor = torch.from_numpy(img_np.reshape(-1, 3)).float().to(device)  # type: ignore
        labels = encoder.rgb_to_labels(img_tensor)  # (H*W,) long
        class_counts = torch.bincount(labels, minlength=config.num_classes).float()
        class_weights = torch.where(
            class_counts > 0,
            class_counts.sum() / (class_counts * class_counts.numel()),
            torch.zeros_like(class_counts),
        )

        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            logits = model(coords)  # (H*W, num_classes)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()  # type: ignore
            scheduler.step()
            if (epoch + 1) % 500 == 0:
                print(f"  epoch {epoch + 1}/{epochs}  loss={loss.item():.4f}")

        out_model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict()}, str(out_model_path))
        print(f"Saved checkpoint to {out_model_path}")

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Any],
        config: InterpolatorConfig,
    ) -> "NeuralSmoother":
        """Construct a ``NeuralSmoother`` from an in-memory state dict.

        Bypasses all file I/O — useful when checkpoints are stored in a
        consolidated archive (e.g. a ``.ptz`` bundle) rather than individual
        per-model files on disk.

        Parameters
        ----------
        state : Mapping[str, Any]
            Raw checkpoint mapping as returned by ``torch.load``.  Accepted
            formats are the same as those handled by :meth:`_load_model`:
            ``{"model_state": …}``, ``{"state_dict": …}``, or a bare
            name → tensor mapping.
        config : InterpolatorConfig
            Interpolator configuration used to size the model and grid.

        Returns
        -------
        NeuralSmoother
            A fully initialized instance ready for inference.
        """
        # Bypass __init__ to avoid requiring a model_path
        instance: "NeuralSmoother" = cls.__new__(cls)
        from interpolators.base import (
            BaseInterpolator,
        )  # avoid circular at module level

        BaseInterpolator.__init__(instance, config)
        instance.device = device_manager.device
        instance.num_classes = DomainConfig.NUM_FACIES
        instance.model = ResidualMLP(
            num_classes=instance.num_classes,
            scale=config.scale,
        ).to(instance.device)
        instance._compile_model()

        ms: dict[str, Any] | None = cls._state_from_checkpoint(state)
        if ms is None:
            raise RuntimeError(
                "Unable to interpret checkpoint state as model state dict."
            )

        # Reuse the same key-normalization logic as _load_model
        model_keys = set(instance.model.state_dict().keys())  # type: ignore
        ck_keys = set(ms.keys())
        prefix = "_orig_mod."

        def _any_prefixed(keys: set[str], pfx: str) -> bool:
            return any(k.startswith(pfx) for k in keys)

        if _any_prefixed(model_keys, prefix) and not _any_prefixed(ck_keys, prefix):
            ms = {f"{prefix}{k}": v for k, v in ms.items()}
        elif _any_prefixed(ck_keys, prefix) and not _any_prefixed(model_keys, prefix):
            ms = {
                (k[len(prefix) :] if k.startswith(prefix) else k): v
                for k, v in ms.items()
            }

        try:
            instance.model.load_state_dict(ms)
        except RuntimeError as exc:
            logger.warning(
                "Strict load_state_dict failed: %s; retrying with strict=False", exc
            )
            instance.model.load_state_dict(ms, strict=False)  # type: ignore[arg-type]

        return instance

    def interpolate(
        self,
        npy_path: Path,
        resolutions: tuple[tuple[int, ...], ...],
    ) -> list[torch.Tensor]:
        """Render smoothed facies images at multiple resolutions using neural interpolation.

        This method performs inference with the trained ResidualMLP model at an
        upsampled super-resolution, evaluating coordinates in chunks to control
        memory usage. It loads the specified image to build a ColorEncoder palette,
        computes class probabilities via softmax, then bilinearly interpolates
        these probabilities to each requested resolution and converts them to RGB.

        Parameters
        ----------
        npy_path : Path
            Filesystem path to the input image file. This image is loaded to
            construct a ColorEncoder that provides the color palette for
            converting model predictions (class labels) to RGB values.
        resolutions : tuple[tuple[int, ...], ...]
            Sequence of (height, width) tuples specifying the desired output
            resolutions. Each resolution produces an interpolated image by
            bilinearly resampling the model's probability maps.

        Returns
        -------
        list[torch.Tensor]
            A list of smoothed images as CPU ``torch.Tensor`` objects, one per
            requested resolution. Each tensor has shape ``(H, W, 3)`` and dtype
            ``torch.float32`` with values in ``[0, 1]`` representing RGB color
            intensities.

        Notes
        -----
        - The model is set to evaluation mode and inference runs under
          ``torch.inference_mode`` for efficiency and reproducibility.
        - The coordinate grid spans the super-resolution dimensions computed
          from ``self.config.geometry`` and ``self.config.upsample``.
        - Coordinates are processed in batches of size ``self.config.chunk_size``
          to limit peak GPU/CPU memory. Increase ``chunk_size`` for better
          throughput at the cost of higher memory usage.
        - A ``ColorEncoder`` is created from ``npy_path`` during this call and
          stored in ``self.encoder`` for palette-based RGB conversion.
        """
        return self._render(_load_image(npy_path), resolutions)

    def interpolate_from_array(
        self,
        img_np: np.ndarray,
        resolutions: tuple[tuple[int, ...], ...],
    ) -> list[torch.Tensor]:
        """Render smoothed facies images from an in-memory numpy array.

        Identical to :meth:`interpolate` but accepts a pre-loaded image array
        instead of a filesystem path, avoiding redundant disk reads when images
        are sourced from a consolidated ``.npz`` bundle.

        Parameters
        ----------
        img_np : np.ndarray
            RGB image array with shape ``(H, W, 3)`` and values in ``[0, 1]``
            (``float32``).  If the array has ``uint8`` values they will be
            normalized automatically.
        resolutions : tuple[tuple[int, ...], ...]
            Sequence of scale descriptors (same format as :meth:`interpolate`).

        Returns
        -------
        list[torch.Tensor]
            Smoothed images at the requested resolutions, each with shape
            ``(H, W, 3)`` and values in ``[0, 1]``.
        """
        # Normalize uint8 → float32 [0, 1] when needed
        img_f32: np.ndarray
        if img_np.dtype != np.float32 or float(img_np.max()) > 1.0:
            img_f32 = img_np.astype(np.float32, copy=False)
            if img_f32.max() > 1.0:
                img_f32 = img_f32 / 255.0
        else:
            img_f32 = img_np

        return self._render(img_f32, resolutions)

    def _render(
        self,
        img_np: np.ndarray,
        resolutions: tuple[tuple[int, ...], ...],
    ) -> list[torch.Tensor]:
        """Core rendering logic shared by ``interpolate`` and ``interpolate_from_array``."""
        logger.debug("Rendering facies pyramid...")
        native_h, native_w = self.config.geometry
        upsample: int | tuple[int, int] = self.config.upsample
        if isinstance(upsample, tuple):
            up_h, up_w = int(upsample[0]), int(upsample[1])  # type: ignore
        else:
            up_h = up_w = int(upsample)
        super_height: int = int(native_h * up_h)
        super_width: int = int(native_w * up_w)

        self.model.eval()  # pyright: ignore

        with torch.inference_mode():
            coords = get_mgrid(height=super_height, width=super_width).to(self.device)

            logits_chunks: list[torch.Tensor] = []
            smooth_imgs: list[torch.Tensor] = []

            for i in range(0, coords.shape[0], self.config.chunk_size):
                chunk = coords[i : i + self.config.chunk_size]
                logits_chunks.append(self.model(chunk).clone())

            logits = torch.cat(logits_chunks, dim=0)
            probs = torch.softmax(logits, dim=1)
            probs = (
                probs.reshape((super_height, super_width, probs.shape[1]))
                .permute(2, 0, 1)
                .unsqueeze(0)
            )

            encoder = ColorEncoder(img_np)
            palette = encoder.palette_tensor.to(self.device).float()

            for resolution in resolutions:
                if self.config.channels_last:
                    _, new_h, new_w, _ = resolution
                else:
                    _, _, new_h, new_w = resolution

                inter_probs = F.interpolate(
                    probs,
                    size=(new_h, new_w),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                inter_probs = (
                    inter_probs.squeeze(0)
                    .permute(1, 2, 0)
                    .reshape((-1, inter_probs.shape[1]))
                )
                inter_probs = inter_probs.to(self.device)
                pred_rgb = torch.matmul(inter_probs, palette)
                smooth_img = (  # pyright: ignore
                    pred_rgb.detach()
                    .cpu()
                    .reshape((new_h, new_w, 3))  # pyright: ignore
                )
                smooth_imgs.append(smooth_img)  # pyright: ignore

        return smooth_imgs


# ==========================================
# 2. IMPROVED ARCHITECTURE: Residual MLP
# ==========================================
class FourierFeatureTransform(nn.Module):
    """Fourier feature mapping used to embed 2D coordinates.

    This module projects 2D coordinates into a higher-dimensional
    sinusoidal feature space using a random Gaussian mapping matrix.

    Parameters
    ----------
    mapping_size : int
        Number of Fourier features per sine/cosine pair.
    scale : float
        Frequency scaling (sigma) applied to the random projection.
    """

    B: (
        torch.Tensor
    )  # shape (2, mapping_size) - random projection matrix stored as a buffer

    def __init__(self, mapping_size: int = 256, scale: float = 10.0) -> None:
        """Initialize Fourier feature projection and register buffers."""
        super().__init__()  # pyright: ignore[reportUnknownMemberType]

        # 'scale' is the "sigma". Higher = sharper/noisier. Lower = smoother/blurrier.
        # store as a buffer (not a trainable parameter) to avoid showing up in optimizer
        B_init = torch.randn(2, mapping_size) * scale
        self.register_buffer("B", B_init, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fourier features to input coordinates.

        Parameters
        ----------
        x : torch.Tensor
            Input coordinates of shape (..., 2).

        Returns
        -------
        torch.Tensor
            Concatenated sin/cos feature tensor of shape (..., mapping_size*2).
        """
        # Ensure numeric constant is a Python float so the result of the
        # static type-checkers resolve the expression type (avoid 'Unknown').
        factor: float = float(2.0 * np.pi)
        # Use a locally-cast buffer to satisfy static type-checkers which may
        # infer an incorrect union type for registered buffers (e.g. Tensor | Module)
        # Use torch.matmul to make the tensor operation explicit for static type checkers
        x_proj = torch.matmul(x * factor, self.B)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ResidualMLP(nn.Module):
    """Residual MLP with Fourier features and skip connections.

    This architecture uses a Fourier feature embedding, residual/skip
    connections and LayerNorm to produce stable per-coordinate class
    predictions.
    """

    def __init__(
        self,
        num_classes: int,
        mapping_size: int = 128,
        scale: float = 1.0,
        hidden_dim: int = 256,
    ) -> None:
        """Initialize the ResidualMLP network components.

        Parameters
        ----------
        num_classes : int
            Number of output classes.
        mapping_size : int
            Size of Fourier mapping features.
        scale : float
            Frequency scaling for Fourier features.
        hidden_dim : int
            Hidden layer dimension size.
        """
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.fourier = FourierFeatureTransform(mapping_size, scale)
        input_dim = mapping_size * 2

        # Standard layers
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            FusedLayerNorm(hidden_dim),
            nn.GELU(),  # Smoother than ReLU
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            FusedLayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Skip connection point: We concat input_dim + hidden_dim
        self.skip_layer = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            FusedLayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            FusedLayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute per-coordinate class logits from input coordinates.

        Parameters
        ----------
        coords : torch.Tensor
            Input coordinate tensor of shape (..., 2).

        Returns
        -------
        torch.Tensor
            Unnormalized class logits for each input coordinate.
        """
        # Embed coordinates
        x_emb = self.fourier(coords)

        # First block
        h = self.layer1(x_emb)
        h = self.layer2(h)

        # Skip connection
        h = torch.cat([h, x_emb], dim=-1)
        h = self.skip_layer(h)

        # Final block
        h = self.layer3(h)
        return self.output(h)
