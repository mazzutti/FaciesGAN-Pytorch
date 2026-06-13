from typing import Any

import torch
import torch.nn as nn

from config import DomainConfig, PhysicsConfig
from device import device_manager
from enums import DataFiles, StatKey


class PhysicsState(nn.Module):
    """
    Manages physics-related buffers and logic such as denormalization
    and computing physical ranges for rock physics models.
    """

    def __init__(
        self,
        options: Any,
        shapes: tuple[tuple[int, ...], ...],
    ) -> None:
        super().__init__()
        self.options = options
        self.shapes = shapes
        self.phys_names: list[str] = []
        self.fixed_kernel_size: int = PhysicsConfig.FIXED_KERNEL_SIZE

        # Will be registered as buffers:
        self.norm_min: torch.Tensor
        self.norm_max: torch.Tensor
        self.padding_value: torch.Tensor
        self.phys_diff: torch.Tensor
        self.phys_min: torch.Tensor

        self.seis_min: torch.Tensor
        self.seis_max: torch.Tensor
        self.rho_mean: torch.Tensor
        self.ip_min: torch.Tensor
        self.ip_max: torch.Tensor
        self.vp_min: torch.Tensor
        self.vp_max: torch.Tensor
        self.vp_ref: torch.Tensor
        self.dz_pyramid: torch.Tensor
        self.wavelet_dt: torch.Tensor
        self.wavelet_t: torch.Tensor

        self._register_physics_buffers()

    def _register_physics_buffers(self) -> None:
        """Pre-compute and register tensors for fast denormalization on GPU."""
        from datasets.utils import get_effective_global_stats
        from physics.seismic import torch_ricker_wavelet
        from utils import get_padding_value

        # Load stats for Elastic Consistency Loss denormalization.
        # Keep VP/VS range consistent with dataset normalization when
        # robust VP/VS normalization is enabled.
        stats = get_effective_global_stats(
            self.options.input_path,
            vp_vs_robust_range=bool(getattr(self.options, "vp_vs_robust_range", False)),
            vp_vs_robust_percentiles=(
                float(
                    getattr(self.options, "vp_vs_robust_percentiles", (1.0, 99.0))[0]
                ),
                float(
                    getattr(self.options, "vp_vs_robust_percentiles", (1.0, 99.0))[1]
                ),
            ),
        )
        phys_min, phys_diff, phys_mean = self._compute_phys_ranges(stats)

        self.phys_names = []
        if getattr(self.options, "use_ip", True):
            self.phys_names.append("Ip")
        if getattr(self.options, "use_is", True):
            self.phys_names.append("Is")
        if getattr(self.options, "use_vpvs", True):
            self.phys_names.append("VP_VS")

        phys_diff_tensor = torch.stack([phys_diff[n] for n in self.phys_names])
        phys_min_tensor = torch.stack([phys_min[n] for n in self.phys_names])
        self.register_buffer("phys_diff", phys_diff_tensor.view(1, -1, 1, 1))
        self.register_buffer("phys_min", phys_min_tensor.view(1, -1, 1, 1))

        # Register normalization range for synthetic seismic generation
        norm_range_min = float(self.options.normalization_range[0])
        norm_range_max = float(self.options.normalization_range[1])
        self.register_buffer(
            "norm_min",
            torch.tensor(
                norm_range_min,
                dtype=torch.float32,
                device=device_manager.device,
            ),
        )
        self.register_buffer(
            "norm_max",
            torch.tensor(
                norm_range_max,
                dtype=torch.float32,
                device=device_manager.device,
            ),
        )

        # --- Compute Facies-Specific Rock Physics Means ---
        import numpy as np

        # Default fallback values in [0, 1] range (will be remapped to norm_range):
        # Class 0: Floodplain, Class 1: Point bar, Class 2: Channel, Class 3: Boundary
        default_means_01 = torch.tensor([
            [0.25, 0.20, 0.90], # Floodplain (0) - Low Ip, low Is, High VpVs
            [0.55, 0.50, 0.40], # Point Bar (1)   - Mid values
            [0.85, 0.80, 0.15], # Channel (2)     - High Ip, high Is, Low VpVs
            [0.50, 0.50, 0.50], # Boundary (3)    - Mid transition
        ], dtype=torch.float32)

        means_tensor_01 = None
        if "facies_rp_means" in stats:
            try:
                # Load raw means and normalize them on the fly using stats min/max bounds to [0, 1]
                computed_means_01 = np.zeros((4, 3), dtype=np.float32)
                for c in range(4):
                    c_str = str(c)
                    if c_str in stats["facies_rp_means"]:
                        c_stats = stats["facies_rp_means"][c_str]
                        ip_val = float(c_stats["Ip"])
                        is_val = float(c_stats["Is"])
                        vpvs_val = float(c_stats["VP_VS"])

                        ip_01 = (ip_val - stats["Ip"]["min"]) / (stats["Ip"]["max"] - stats["Ip"]["min"] + 1e-10)
                        is_01 = (is_val - stats["Is"]["min"]) / (stats["Is"]["max"] - stats["Is"]["min"] + 1e-10)
                        vpvs_01 = (vpvs_val - stats["VP_VS"]["min"]) / (stats["VP_VS"]["max"] - stats["VP_VS"]["min"] + 1e-10)

                        computed_means_01[c, 0] = ip_01
                        computed_means_01[c, 1] = is_01
                        computed_means_01[c, 2] = vpvs_01
                    else:
                        computed_means_01[c] = default_means_01[c].numpy()
                means_tensor_01 = torch.from_numpy(computed_means_01)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Error parsing facies RP means from stats: {e}. Using fallback defaults.")

        if means_tensor_01 is None:
            means_tensor_01 = default_means_01

        # Remap [0, 1] means to norm_range
        means_norm = means_tensor_01 * (norm_range_max - norm_range_min) + norm_range_min
        self.register_buffer("facies_rp_means", means_norm.view(1, 4, 3, 1, 1))

        seis_min = phys_min[DataFiles.SEISMIC.name]
        self.register_buffer("seis_min", seis_min)
        self.register_buffer("seis_max", seis_min + phys_diff[DataFiles.SEISMIC.name])

        self.register_buffer(
            "padding_value",
            torch.tensor(
                get_padding_value(self.options.normalization_range),
                device=device_manager.device,
            ),
        )

        # Fixed kernel size for Wavelet Resampling — plain int, not a buffer,
        # since it is a compile-time constant that never changes.
        self.fixed_kernel_size = PhysicsConfig.FIXED_KERNEL_SIZE

        # Register rho_mean as a buffer for fast access in physics loss
        rho_mean = float(stats[DataFiles.RHO.name][StatKey.MEAN])
        self.register_buffer(
            "rho_mean",
            torch.tensor(rho_mean, device=device_manager.device, dtype=torch.float32),
        )

        ip_min = phys_min[DataFiles.Ip.name]
        self.register_buffer("ip_min", ip_min)
        self.register_buffer("ip_max", ip_min + phys_diff[DataFiles.Ip.name])

        vp_min = phys_min[DataFiles.VP.name]
        self.register_buffer("vp_min", vp_min)
        self.register_buffer("vp_max", vp_min + phys_diff[DataFiles.VP.name])

        # Reference velocity for Wavelet Resampling (avoids per-batch sync)
        ip_mean = float(phys_mean[DataFiles.Ip.name])
        vp_ref = torch.tensor(
            ip_mean / rho_mean, device=device_manager.device, dtype=torch.float32
        )
        self.register_buffer("vp_ref", vp_ref)

        # Pre-calculate dz for every scale to avoid redundant float math in G-loop
        dz_pyramid: list[float] = []
        h_target = float(
            self.shapes[getattr(self.options, "stop_scale", len(self.shapes) - 1)][2]
        )
        for s in range(len(self.shapes)):
            h_scale = float(self.shapes[s][2])
            ratio = h_target / h_scale
            dz_pyramid.append(self.options.dz_pixel * ratio)
        self.register_buffer(
            "dz_pyramid",
            torch.tensor(
                dz_pyramid,
                device=device_manager.device,
                dtype=torch.float32,
            ),
        )

        # 2. Pre-generate time-domain wavelet for Physics Loss
        self.register_buffer(
            "wavelet_dt",
            torch.tensor(
                self.options.wavelet_dt,
                device=device_manager.device,
                dtype=torch.float32,
            ),
        )

        wavelet_t = torch_ricker_wavelet(
            self.options.wavelet_f_peak,
            self.options.wavelet_dt,
            self.options.wavelet_length,
        )
        self.register_buffer("wavelet_t", wavelet_t)

    def _compute_phys_ranges(self, stats: dict[str, dict[str, Any]]) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        """Compute the physical minimum and span (diff) for all rock physics components."""
        phys_min: dict[str, torch.Tensor] = {}
        phys_diff: dict[str, torch.Tensor] = {}
        phys_mean: dict[str, torch.Tensor] = {}

        for comp in DataFiles.all_rock_physics() + [DataFiles.SEISMIC]:
            key = comp.name
            s = stats[key]

            # Note: VP/VS in stats.json are in Km/s, convert to m/s
            scale = (
                PhysicsConfig.VP_MS_SCALE
                if comp in [DataFiles.VP, DataFiles.VS]
                else 1.0
            )

            phys_min[key] = torch.tensor(
                s[StatKey.MIN] * scale,
                device=device_manager.device,
                dtype=torch.float32,
            )
            phys_diff[key] = torch.tensor(
                (s[StatKey.MAX] - s[StatKey.MIN]) * scale,
                device=device_manager.device,
                dtype=torch.float32,
            )
            phys_mean[key] = torch.tensor(
                s[StatKey.MEAN] * scale,
                device=device_manager.device,
                dtype=torch.float32,
            )
        return phys_min, phys_diff, phys_mean

    def denormalize_rock_physics(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        """Denormalize a rock physics tensor (B, 3, H, W) to physical units.

        Channels follow the order from DataFiles.generator_output_rock_physics():
        [Ip, Is, Vp/Vs]. Returns a dictionary keyed by component name.
        """
        # Single fused multiply-add using (1, C, 1, 1) registered buffers.
        # Avoids the per-channel loop + repeated dict lookups in the hot path.
        # Convert tensor from normalization_range to [0, 1]
        # This prevents negative physical values when normalization_range is [-1, 1]
        t_0_1 = (tensor - self.norm_min) / (
            self.norm_max - self.norm_min + DomainConfig.EPSILON
        )
        phys = t_0_1 * self.phys_diff + self.phys_min
        return {n: phys[:, i : i + 1, ...] for i, n in enumerate(self.phys_names)}
