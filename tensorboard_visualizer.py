"""TensorBoard-based training visualizer for parallel FACIESGAN training.

This provides a clean, real-time, non-blocking visualization using TensorBoard.
Much more responsive than matplotlib with better interactivity.
"""

from __future__ import annotations

# pyright: reportUnknownMemberType=false
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.cm as cm
import numpy as np
import torch
from tensorboardX import SummaryWriter  # pyright: ignore

import utils
from background_workers import submit_save_image
from config import DirectoryConfig, DomainConfig, LoggingConfig
from device import device_manager
from enums import MetricKey
from models.utils import SplitKey, split_facies_rp
from physics.seismic import calculate_synthetic_seismic

if TYPE_CHECKING:
    from physics.physics import PhysicsState
    from training.metrics import ScaleMetrics


logger = logging.getLogger(__name__)


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
        update_interval: int = LoggingConfig.SCALAR_LOG_INTERVAL,
        image_log_interval: int = LoggingConfig.IMAGE_LOG_INTERVAL,
        dataset_info: str | None = None,
        purge_step: int | None = None,
        num_facies: int = DomainConfig.NUM_FACIES_CHANNELS,
        has_rp: bool = False,
        physics_state: "PhysicsState | None" = None,
        normalization_range: tuple[float, float] = (-1.0, 1.0),
        seismic_stretch_percentile: int = 98,  # kept for call-site compatibility, no longer used
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
        seismic_stretch_percentile : int, optional
            Deprecated. Percentile parameter is no longer used; the
            visualizer applies a fixed 2/98 robust stretch to the synthetic
            seismic's own distribution. Kept for backwards-compatible call
            sites but has no effect.
        """
        self.num_scales = num_scales
        self.output_dir = output_dir
        self.update_interval = update_interval
        self.image_log_interval = image_log_interval
        self.dataset_info = dataset_info or "Unknown dataset"
        self.num_facies_channels = num_facies
        self.has_rp = has_rp
        self.physics_state = physics_state
        self.normalization_range = (
            float(normalization_range[0]),
            float(normalization_range[1]),
        )

        # Setup TensorBoard logging
        if not log_dir:
            log_dir = str(Path(output_dir) / DirectoryConfig.TENSORBOARD_LOGS)

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

        print("TensorBoard initialized.")
        print(f"tensorboard --logdir={log_dir} --port=6006 --bind_all")

    def update(
        self,
        epoch: int,
        scale_metrics: ScaleMetrics | dict[int, dict[str, float]],
        generated_samples: tuple[torch.Tensor, ...] | None = None,
        samples_processed: int = 0,
        scales: tuple[int, ...] | None = None,
        real_seismic: dict[int, torch.Tensor] | None = None,
        force_update: bool = False,
    ) -> None:
        """Update TensorBoard with per-scale metrics and optional images.

        Parameters
        ----------
        epoch : int
            Current epoch number (used as the global step in TensorBoard).
        scale_metrics : ScaleMetrics or dict[int, dict[str, float]]
            Either a `ScaleMetrics` dataclass (with tensor fields) or a
            dictionary of scale -> metric-key->float dictionaries.
            When a `ScaleMetrics` is provided the method will extract scalar
            values via `tensor.item()` before logging.
        generated_samples : tuple[torch.Tensor, ...], optional
            Tuple of generated sample tensors, one per scale. Expected
            tensor shapes: (B, C, H, W) or (C, H, W). The first sample in the
            batch is used for logging. Tensors are detached and moved to CPU
            before conversion to numpy arrays.
        samples_processed : int, optional
            Running count of processed samples used for progress logging.

        Notes
        -----
        This method normalizes images to the configured normalization range
        when necessary and
        converts them to CHW format for TensorBoard. Mean scalars across
        scales are written under the `Mean/*` tags.
        """
        # Update at intervals, or if forced (e.g. final epoch)
        if not force_update and epoch % self.update_interval != 0 and epoch != 1:
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
        d_keys = [
            MetricKey.D_TOTAL,
            MetricKey.D_REAL,
            MetricKey.D_FAKE,
            MetricKey.D_GP,
        ]
        g_keys = [
            MetricKey.G_TOTAL,
            MetricKey.G_FAKE,
            MetricKey.G_REC_FACIES,
            MetricKey.G_WELL,
            MetricKey.G_DIV,
            MetricKey.G_REC_ROCK_PHYSICS,
            MetricKey.G_TV,
            MetricKey.G_ELASTIC,
            MetricKey.G_SEISMIC,
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
        if generated_samples and (epoch % self.image_log_interval == 0 or force_update):
            for i, sample in enumerate(generated_samples):
                scale = scales[i] if scales is not None else i
                # Convert to numpy (B, H, W, C) in the normalized domain
                # tensor2np handles detach, cpu, and denormalization automatically.
                img_bhwc = utils.tensor2np(
                    sample,
                    denormalize=False,
                    normalization_range=self.normalization_range,
                )
                img_hwc = img_bhwc[0]

                # Use centralized logic to split facies and rock physics if present.
                # Since these are generated samples, they do not contain conditioning.
                split = split_facies_rp(
                    torch.from_numpy(img_hwc),
                    num_facies=self.num_facies_channels,
                    has_rp=self.has_rp,
                    channels_last=True,
                )
                facies_img = split[SplitKey.FACIES]
                rp_img = split[SplitKey.ROCK_PHYSICS]

                # Convert one-hot facies to RGB using a high-contrast palette
                if facies_img is not None:
                    # facies_img is (H, W, C) due to channels_last=True
                    # We need to transpose back to (C, H, W) for facies_to_rgb
                    facies_chw = facies_img.permute(2, 0, 1)
                    # Quantize continuous [-1,1] generator output to nearest
                    # palette colour (same as plot_pyramids.py: rgb_to_facies
                    # then facies_to_rgb), so TensorBoard shows discrete
                    # palette colours instead of a blurry continuous image.
                    facies_idx = utils.rgb_to_facies(facies_chw)
                    facies_rgb_chw = utils.facies_to_rgb(facies_idx)
                    self.writer.add_image(
                        f"Samples_Facies/Scale_{scale}", facies_rgb_chw, epoch
                    )

                    # Also save to disk
                    try:
                        # Organize by scale
                        scale_dir = Path(self.output_dir) / f"Scale_{scale}"
                        scale_dir.mkdir(parents=True, exist_ok=True)

                        out_path = scale_dir / f"Facies_epoch_{epoch:05d}.png"
                        # facies_rgb_chw is already a numpy array (3, H, W) from utils.facies_to_rgb
                        rgb_hwc = facies_rgb_chw.transpose(1, 2, 0)
                        submit_save_image(rgb_hwc, str(out_path))
                    except Exception:
                        logger.warning(
                            "Could not submit facies image for background save",
                            exc_info=True,
                        )

                if rp_img is not None and self.physics_state is not None:
                    # Rock physics contains [Ip, Is, Vp/Vs].
                    # Use raw tensor channels (without tensor2np default
                    # clipping to [0, 1]) so diagnostics and plots reflect
                    # the true normalization domain (e.g. [-1, 1]).
                    sample_chw = device_manager.to_cpu(sample[0])
                    rp_chw_t = sample_chw[
                        self.num_facies_channels : self.num_facies_channels + 3, ...
                    ]
                    rp_chw = rp_chw_t.numpy()

                    # Log individual RP attributes if they exist
                    # Keep legacy TensorBoard tag spelling for VP/VS so
                    # dashboards and historical runs stay comparable.
                    names = ["Ip", "Is", "VpVs"]
                    cmaps = ["magma", "magma", "viridis"]
                    # Batch-read normalization scalars from the PhysicsState
                    # to avoid multiple GPU->CPU synchronizations in the loop.
                    norm_min_t, norm_max_t = device_manager.to_cpu(
                        [self.physics_state.norm_min, self.physics_state.norm_max],
                        non_blocking=True,
                    )
                    norm_min_f = float(norm_min_t.item())
                    norm_max_f = float(norm_max_t.item())
                    lo = float(min(norm_min_f, norm_max_f))
                    hi = float(max(norm_min_f, norm_max_f))
                    span = hi - lo
                    for ch_idx, name in enumerate(names):
                        if rp_chw.shape[0] > ch_idx:
                            channel_raw = np.asarray(rp_chw[ch_idx], dtype=np.float32)
                            channel_raw = np.nan_to_num(
                                channel_raw, nan=lo, posinf=hi, neginf=lo
                            )
                            # Percentile contrast stretch for display only.
                            # RP channels can live in a narrow sub-range of the
                            # configured normalization domain
                            # (e.g. VP/VS normalises into [0.017, 0.36] due to
                            # stats.json being wider than the actual data range).
                            # Without stretching, all values map to the dark end
                            # of the colormap and spatial variation is invisible.
                            p1 = float(np.percentile(channel_raw, 1))
                            p99 = float(np.percentile(channel_raw, 99))
                            if p99 - p1 > DomainConfig.EPSILON:
                                attr_hw = np.clip(
                                    (channel_raw - p1) / (p99 - p1), 0.0, 1.0
                                )
                            else:
                                if span > DomainConfig.EPSILON:
                                    attr_hw = np.clip(
                                        (channel_raw - lo) / span, 0.0, 1.0
                                    )
                                else:
                                    attr_hw = np.zeros_like(
                                        channel_raw, dtype=np.float32
                                    )

                            self._add_image_with_cmap(
                                f"Samples_{name}/Scale_{scale}",
                                attr_hw,
                                cmaps[ch_idx],
                                epoch,
                            )

                # Log Synthetic Seismic if Rock Physics is enabled
                if self.has_rp and self.physics_state is not None:
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
        if self.physics_state is None:
            return

        # 1. Extract Ip from generated sample (B, C, H, W)
        # generated sample contains [Facies | Ip, Is, VpVs]
        if sample.shape[1] <= self.num_facies_channels:
            return

        ip_norm = sample[
            :, self.num_facies_channels : self.num_facies_channels + 1, ...
        ]

        # 2. Compute Synthetic Seismic
        # We estimate vp_mean from Ip for dynamic resampling
        rho_mean = self.physics_state.rho_mean
        vp_min = self.physics_state.vp_min
        vp_max = self.physics_state.vp_max
        ip_min = self.physics_state.ip_min
        ip_max = self.physics_state.ip_max

        norm_min = self.physics_state.norm_min
        norm_max = self.physics_state.norm_max
        ip_phys_approx = (ip_norm - norm_min) / (
            norm_max - norm_min + DomainConfig.EPSILON
        ) * (ip_max - ip_min) + ip_min
        vp_phys = ip_phys_approx / rho_mean
        vp_mean = torch.mean(vp_phys).clamp(vp_min, vp_max)
        dz = self.physics_state.dz_pyramid[scale]

        synth = calculate_synthetic_seismic(
            ip_norm,
            vp_mean,
            dz,
            self.physics_state,
        )

        # 3. Log Generated Seismic
        # Normalize the synthetic seismic using its own robust percentile
        # statistics so the full colormap range is used at every training stage,
        # regardless of how the generator's amplitude scale compares to the
        # real-data statistics stored in physics_state.
        #
        # The physical sign convention is preserved:
        #   RC > 0  (impedance increases downward) → red
        #   RC < 0  (impedance decreases downward) → blue
        #   RC = 0  (no interface)                 → white (centre of RdBu)
        #
        # Using dataset seis_min/seis_max as a normalisation reference fails
        # when the synthetic amplitudes are much smaller than the real-data
        # range, causing all values to cluster near the centre and produce a
        # "ghost / translucent" appearance.
        synth_np = np.asarray(device_manager.to_numpy(synth[0, 0]), dtype=np.float32)

        # Remove DC bias so the zero amplitude is perfectly centered
        synth_np = synth_np - np.mean(synth_np)

        # Symmetric stretch around zero — 2nd/98th percentile gives robustness
        # against outliers while fully utilising the diverging colour range.
        p_lo = float(np.percentile(synth_np, 2))
        p_hi = float(np.percentile(synth_np, 98))
        max_abs = max(abs(p_lo), abs(p_hi), DomainConfig.EPSILON)
        synth_norm = np.clip(synth_np / max_abs, -1.0, 1.0)

        # Map [-1, 1] → [0, 1]: zero amplitude lands at 0.5 (white in RdBu).
        synth_mapped = ((synth_norm + 1.0) / 2.0).astype(np.float32)

        self._add_image_with_cmap(
            f"Samples_Seismic/Scale_{scale}", synth_mapped, "RdBu", epoch
        )

    def _add_image_with_cmap(
        self, tag: str, data_hw: np.ndarray, cmap_name: str, epoch: int
    ) -> None:
        """Apply colormap to HxW data and log as CHW image."""
        # Ensure data is within the unit interval for colormap lookup
        data_norm = np.clip(data_hw, 0.0, 1.0)
        # Apply colormap (returns HxWx4)
        rgb_hwc = np.asarray(
            cm.get_cmap(cmap_name)(data_norm)[..., :3], dtype=np.float32
        )
        # Transpose to CHW for TensorBoard
        rgb_chw = np.transpose(rgb_hwc, (2, 0, 1))
        self.writer.add_image(tag, rgb_chw, epoch)

        # Also save to disk as a PNG file in training_visualizations
        try:
            # tag is e.g. "Samples_Facies/Scale_0" or "Samples_Ip/Scale_0"
            parts = tag.split("/")
            category = parts[0].replace("Samples_", "")
            scale_name = parts[1] if len(parts) > 1 else "Global"

            # Create scale-specific directory
            scale_dir = Path(self.output_dir) / scale_name
            scale_dir.mkdir(parents=True, exist_ok=True)

            out_path = scale_dir / f"{category}_epoch_{epoch:05d}.png"

            # Offload to background thread
            submit_save_image(rgb_hwc, str(out_path))
        except Exception:
            # Don't crash training if image saving fails
            logger.warning(
                "Could not submit visualization for background save",
                exc_info=True,
            )

    def close(self):
        """Close the TensorBoard writer."""
        if hasattr(self, "writer"):
            self.writer.close()

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()
