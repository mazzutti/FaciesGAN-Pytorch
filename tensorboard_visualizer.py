"""TensorBoard-based training visualizer for parallel FACIESGAN training.

This provides a clean, real-time, non-blocking visualization using TensorBoard.
Much more responsive than matplotlib with better interactivity.
"""

from __future__ import annotations

# pyright: reportUnknownMemberType=false
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.cm as cm
import numpy as np
import torch
from tensorboardX import SummaryWriter  # pyright: ignore

import utils
from models.utils import split_facies_rp

if TYPE_CHECKING:
    from metrics import ScaleMetrics


@dataclass
class SeismicPhysicsInfo:
    """Metadata required for synthetic seismic modeling in TensorBoard."""

    ip_min: float | torch.Tensor
    ip_max: float | torch.Tensor
    seis_min: float | torch.Tensor
    seis_max: float | torch.Tensor
    vp_min: float | torch.Tensor
    vp_max: float | torch.Tensor
    vp_ref: float | torch.Tensor
    rho_mean: float | torch.Tensor
    dz_pyramid: dict[int, float] | torch.Tensor
    wavelet_t: torch.Tensor
    dt_wavelet: float


class TensorBoardVisualizer:
    """Real-time training visualization using TensorBoard.

    This helper writes per-scale scalar metrics and generated sample images to
    a `tensorboardX.SummaryWriter`. It accepts either a `ScaleMetrics`
    dataclass (with tensor-valued fields) or a pre-flattened mapping of
    floats and converts values appropriately before logging.

    Attributes
    ----------
    num_scales : int
        Number of scales being trained in parallel.
    output_dir : str
        Directory where visualization images are saved.
    update_interval : int
        Frequency (in epochs) at which `update()` will emit logs.
    writer : SummaryWriter
        TensorBoard writer used to record scalars and images.
    start_time : float
        Timestamp when the visualizer was created (used to compute elapsed time).
    """

    def __init__(
        self,
        num_scales: int,
        output_dir: str,
        log_dir: str | None = None,
        update_interval: int = 1,
        image_log_interval: int = 100,
        dataset_info: str | None = None,
        purge_step: int | None = None,
        num_facies: int = 4,
        has_rp: bool = False,
        physics_info: SeismicPhysicsInfo | None = None,
    ):
        """Initialize the TensorBoard visualizer.

        Parameters
        ----------
        num_scales : int
            Number of scales being trained in parallel.
        output_dir : str
            Directory to save visualization images.
        log_dir : str, optional
            Directory for TensorBoard logs. If None, uses output_dir/tensorboard_logs
        update_interval : int
            How often to log metrics (in epochs).
        dataset_info : str, optional
            Information about the dataset being used.
        """
        self.num_scales = num_scales
        self.output_dir = output_dir
        self.update_interval = update_interval
        self.image_log_interval = image_log_interval
        self.dataset_info = dataset_info or "Unknown dataset"
        self.num_facies = num_facies
        self.has_rp = has_rp
        self.physics_info = physics_info

        # Setup TensorBoard logging
        if not log_dir:
            log_dir = os.path.join(output_dir, "tensorboard_logs")

        # Ensure directories exist; guard against empty strings and race conditions.
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        os.makedirs(output_dir or ".", exist_ok=True)

        # Create TensorBoard writer.  When resuming, purge_step tells
        # tensorboardX to invalidate stale events from a previous run so
        # the curves stay clean.
        self.writer = SummaryWriter(log_dir=log_dir, purge_step=purge_step)

        # Training timing
        self.start_time = time.time()
        self.last_update_time = self.start_time

        # Write dataset info as text
        self.writer.add_text("Dataset/Info", self.dataset_info, 0)
        self.writer.add_text(
            "Training/Scales", f"Training {num_scales} scales in parallel", 0
        )

        print(f"✅ TensorBoard initialized.")
        print(f"   tensorboard --logdir={log_dir} --port=6006 --bind_all")

    def update(
        self,
        epoch: int,
        scale_metrics: ScaleMetrics | dict[int, dict[str, float]],
        generated_samples: tuple[torch.Tensor, ...] | None = None,
        samples_processed: int = 0,
        scales: tuple[int, ...] | None = None,
        real_seismic: dict[int, torch.Tensor] | None = None,
    ) -> None:
        """Update TensorBoard with per-scale metrics and optional images.

        Parameters
        ----------
        epoch : int
            Current epoch number (used as the global step in TensorBoard).
        scale_metrics : ScaleMetrics or tuple[dict[str, float], ...]
            Either a `ScaleMetrics` dataclass (with tensor fields) or a
            pre-flattened tuple of metric-name->float dictionaries, one per scale.
            When a `ScaleMetrics` is provided the method will extract scalar
            values via `tensor.item()` before logging.
            Dictionary of generated sample tensors, one per scale. Expected
            tensor shapes: (B, C, H, W) or (C, H, W). The first sample in the
            batch is used for logging. Tensors are detached and moved to CPU
            before conversion to numpy arrays.
        samples_processed : int, optional
            Running count of processed samples used for progress logging.

        Notes
        -----
        This method normalizes images to the [0, 1] range when necessary and
        converts them to CHW format for TensorBoard. Mean scalars across
        scales are written under the `Mean/*` tags.
        """
        # Update at intervals
        if epoch % self.update_interval != 0 and epoch != 1:
            return

        # Use attribute detection to determine whether we have a typed
        # ScaleMetrics instance (isinstance checks won't work due to generics).
        if not isinstance(scale_metrics, dict):
            flat_metrics = scale_metrics.as_flat_dict()
        else:
            flat_metrics = scale_metrics

        # Calculate timing info
        current_time = time.time()
        elapsed = current_time - self.start_time

        # Define standard metric groups
        d_keys = ["d_total", "d_real", "d_fake", "d_gp"]
        g_keys = [
            "g_total",
            "g_fake",
            "g_facies_rec",
            "g_well",
            "g_div",
            "g_rec_rock_physics",
            "g_tv",
            "g_elastic",
            "g_physics",
        ]

        # Log individual scale metrics
        for scale, metrics in flat_metrics.items():
            for key in d_keys:
                self.writer.add_scalar(
                    f"Scale_{scale}/{key.upper()}", metrics.get(key, 0.0), epoch
                )
            for key in g_keys:
                # Map internal names to UI labels
                label = key.replace("g_fake", "g_adversarial").upper()
                self.writer.add_scalar(
                    f"Scale_{scale}/{label}", metrics.get(key, 0.0), epoch
                )

        # Compute and log mean losses across all scales
        if scale_metrics:
            active_scales = list(flat_metrics.keys())
            for key in d_keys:
                val = float(
                    np.mean([flat_metrics[s].get(key, 0.0) for s in active_scales])
                )
                self.writer.add_scalar(f"Mean/{key.upper()}", val, epoch)
            for key in g_keys:
                label = key.replace("g_fake", "g_adversarial").upper()
                val = float(
                    np.mean([flat_metrics[s].get(key, 0.0) for s in active_scales])
                )
                self.writer.add_scalar(f"Mean/{label}", val, epoch)

        # Log training progress
        self.writer.add_scalar("Training/Samples_Processed", samples_processed, epoch)
        self.writer.add_scalar("Training/Elapsed_Time_Minutes", elapsed / 60, epoch)

        # Log generated samples as images with color mapping
        if generated_samples and epoch % self.image_log_interval == 0:
            for i, sample in enumerate(generated_samples):
                scale = scales[i] if scales is not None else i
                # Convert to numpy (B, H, W, C) in [0, 1] range
                # tensor2np handles detach, cpu, and denormalization automatically.
                img_bhwc = utils.tensor2np(sample, denormalize=True)
                img_hwc = img_bhwc[0]

                # Use centralized logic to split facies and rock physics if present.
                # Since these are generated samples, they do not contain conditioning.
                split = split_facies_rp(
                    torch.from_numpy(img_hwc),
                    num_facies=self.num_facies,
                    has_rp=self.has_rp,
                    channels_last=True,
                )
                facies_img = split["facies"]
                rp_img = split["rock_physics"]

                # Convert one-hot facies to RGB using a high-contrast palette
                if facies_img is not None:
                    # facies_img is (H, W, C) due to channels_last=True
                    # We need to transpose back to (C, H, W) for facies_to_rgb
                    facies_chw = facies_img.permute(2, 0, 1)
                    facies_rgb_chw = utils.facies_to_rgb(facies_chw)
                    self.writer.add_image(
                        f"Samples_Facies/Scale_{scale}", facies_rgb_chw, epoch
                    )

                if rp_img is not None:
                    # Rock physics contains [Ip, Is, Vp/Vs]
                    # Transpose to (C, H, W) for easier slicing
                    rp_chw = rp_img.permute(2, 0, 1).cpu().numpy()

                    # Log individual RP attributes if they exist
                    names = ["Ip", "Is", "VpVs"]
                    cmaps = ["magma", "magma", "viridis"]
                    for i, name in enumerate(names):
                        if rp_chw.shape[0] > i:
                            attr_hw = np.clip(rp_chw[i], 0.0, 1.0)
                            self._add_image_with_cmap(
                                f"Samples_{name}/Scale_{scale}",
                                attr_hw,
                                cmaps[i],
                                epoch,
                            )

                # Log Synthetic Seismic if Rock Physics is enabled
                if self.has_rp and self.physics_info is not None:
                    self._log_seismic(scale, sample, real_seismic, epoch)

        self.last_update_time = current_time

        # Flush to ensure data is written
        self.writer.flush()

    def _log_seismic(
        self,
        scale: int,
        sample: torch.Tensor,
        real_seismic_dict: dict[int, torch.Tensor] | None,
        epoch: int,
    ) -> None:
        """Compute and log synthetic seismic vs real seismic."""
        if self.physics_info is None:
            return

        from models.utils import calculate_synthetic_seismic

        # 1. Extract Ip from generated sample (B, C, H, W)
        # generated sample contains [Facies | Ip, Is, VpVs]
        if sample.shape[1] <= self.num_facies:
            return

        ip_norm = sample[:, self.num_facies : self.num_facies + 1, ...]

        # 2. Compute Synthetic Seismic
        # We estimate vp_mean from Ip for dynamic resampling
        rho_mean = torch.as_tensor(self.physics_info.rho_mean, device=sample.device)
        vp_min = torch.as_tensor(self.physics_info.vp_min, device=sample.device)
        vp_max = torch.as_tensor(self.physics_info.vp_max, device=sample.device)

        ip_min = torch.as_tensor(self.physics_info.ip_min, device=sample.device)
        ip_max = torch.as_tensor(self.physics_info.ip_max, device=sample.device)

        ip_phys_approx = ((ip_norm + 1) / 2) * (ip_max - ip_min) + ip_min
        vp_phys = ip_phys_approx / rho_mean
        vp_mean = torch.mean(vp_phys).clamp(vp_min, vp_max)

        # Ensure wavelet/time parameters are tensors to match calculate_synthetic_seismic API
        device = ip_norm.device
        dtype = ip_norm.dtype
        wavelet_t = torch.as_tensor(
            self.physics_info.wavelet_t, device=device, dtype=dtype
        )
        dt_wavelet = torch.as_tensor(
            self.physics_info.dt_wavelet, device=device, dtype=dtype
        )
        dz_value = self.physics_info.dz_pyramid
        dz_value = dz_value[scale]
        dz = torch.as_tensor(float(dz_value), device=device, dtype=dtype)

        synth = calculate_synthetic_seismic(
            ip_norm,
            wavelet_t,
            dt_wavelet,
            dz,
            vp_mean,
            vp_min,
            ip_min,
            ip_max,
            torch.as_tensor(self.physics_info.seis_min, device=device),
            torch.as_tensor(self.physics_info.seis_max, device=device),
        )

        # 3. Log Generated Seismic
        # Convert to RGB with RdBu colormap (Standard for seismic)
        synth_np = synth[0, 0].detach().cpu().numpy()

        # Dynamic symmetric normalization to ensure "punchy" colors like matplotlib's imshow
        # We scale by the maximum absolute value to keep 0 at the center (white)
        v_max_gen = max(abs(synth_np.min()), abs(synth_np.max()))
        synth_norm = synth_np / (v_max_gen + 1e-6)
        synth_mapped = (synth_norm + 1.0) / 2.0

        self._add_image_with_cmap(
            f"Samples_Seismic/Scale_{scale}", synth_mapped, "RdBu", epoch
        )

    def _add_image_with_cmap(
        self, tag: str, data_hw: np.ndarray, cmap_name: str, epoch: int
    ) -> None:
        """Apply colormap to HxW data and log as CHW image."""
        # Ensure data is within [0, 1] for colormap
        data_norm = np.clip(data_hw, 0.0, 1.0)
        # Apply colormap (returns HxWx4)
        rgb_hwc = np.asarray(
            cm.get_cmap(cmap_name)(data_norm)[..., :3], dtype=np.float32
        )
        # Transpose to CHW for TensorBoard
        rgb_chw = np.transpose(rgb_hwc, (2, 0, 1))
        self.writer.add_image(tag, rgb_chw, epoch)

    def close(self):
        """Close the TensorBoard writer."""
        if hasattr(self, "writer"):
            self.writer.close()

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()
