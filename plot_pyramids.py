import logging
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

import utils
from datasets.dataset import PyramidsDataset
from device import device_manager
from models.palette import PALETTE_RGB
from options import TrainingOptions

logger = logging.getLogger(__name__)


def main():
    opt = TrainingOptions()
    opt.use_rock_physics = True
    opt.use_seismic = True
    opt.use_wells = True
    opt.num_facies = 3
    # Load dataset
    print("Loading dataset...")
    dataset = PyramidsDataset(opt)

    num_scales = len(dataset.scales)
    num_facies_ch = opt.num_facies  # account for background

    _, axes = plt.subplots(num_scales, 6, figsize=(18, 3 * num_scales))  # type: ignore

    for scale in range(num_scales):
        # get_scale_data returns (facies, wells, masks, seismic)
        # Note: 'facies' actually contains [Facies | Ip | Is | VpVs] if use_rock_physics=True
        facies_batch, wells_batch, _, seismic_batch = dataset.get_scale_data(scale)

        idx = 0
        if facies_batch.shape[0] == 0:
            continue

        # Move batch tensors to CPU using non-blocking transfers when possible
        f = device_manager.to_cpu(facies_batch[idx], non_blocking=True)
        w = (
            device_manager.to_cpu(wells_batch[idx], non_blocking=True)
            if wells_batch.shape[0] > idx
            else None
        )
        s = (
            device_manager.to_cpu(seismic_batch[idx], non_blocking=True)
            if seismic_batch.shape[0] > idx
            else None
        )

        # Split channels based on configuration
        ip = f[num_facies_ch] if f.shape[0] > num_facies_ch else None
        is_ = f[num_facies_ch + 1] if f.shape[0] > num_facies_ch + 1 else None
        vpvs = f[num_facies_ch + 2] if f.shape[0] > num_facies_ch + 2 else None

        # 1. Facies (RGB to discrete indices for plotting)
        ax = axes[scale, 0]
        # Get continuous RGB and map to closest discrete palette colors
        f_rgb = f[:num_facies_ch]
        f_idx = utils.rgb_to_facies(f_rgb)

        print(
            f"Scale {scale} Facies raw range (normalized): [{float(f.min()):.4f}, {float(f.max()):.4f}]"
        )
        # facies_to_rgb maps back to RGB [0,1] using PALETTE_RGB
        ax.imshow(utils.facies_to_rgb(f_idx).transpose(1, 2, 0))
        ax.set_title(f"Scale {scale} Facies")
        ax.axis("off")

        # 2. Rock Physics: Ip
        ax = axes[scale, 1]
        if ip is not None:
            print(
                f"Scale {scale} Ip range (normalized): [{float(ip.min()):.4f}, {float(ip.max()):.4f}]"
            )
            ax.imshow(ip, cmap="magma")
        ax.set_title(f"Scale {scale} Ip")
        ax.axis("off")

        # 3. Rock Physics: Is
        ax = axes[scale, 2]
        if is_ is not None:
            print(
                f"Scale {scale} Is range (normalized): [{float(is_.min()):.4f}, {float(is_.max()):.4f}]"
            )
            ax.imshow(is_, cmap="magma")
        ax.set_title(f"Scale {scale} Is")
        ax.axis("off")

        # 4. Rock Physics: Vp/Vs
        ax = axes[scale, 3]
        if vpvs is not None:
            print(
                f"Scale {scale} Vp/Vs range (normalized): [{float(vpvs.min()):.4f}, {float(vpvs.max()):.4f}]"
            )
            ax.imshow(vpvs, cmap="viridis")
        ax.set_title(f"Scale {scale} Vp/Vs")
        ax.axis("off")

        # 5. Wells
        ax = axes[scale, 4]
        if w is not None:
            print(
                f"Scale {scale} Wells raw range (normalized): [{float(w.min()):.4f}, {float(w.max()):.4f}]"
            )
            well_cmap = mcolors.ListedColormap(PALETTE_RGB)
            w_plot = w  # (3, H, W) RGB

            # Map RGB to discrete indices (background [-1, -1, -1] automatically maps to 0 / Black)
            w_idx = utils.rgb_to_facies(w_plot).astype(float)
            ax.imshow(w_idx, cmap=well_cmap, vmin=0, vmax=len(PALETTE_RGB) - 1)
        ax.set_title(f"Scale {scale} Wells")
        ax.axis("off")

        # 6. Seismic (display-only symmetric stretch around zero)
        ax = axes[scale, 5]
        if s is not None:
            print(
                f"Scale {scale} Seismic range (normalized): [{float(s.min()):.4f}, {float(s.max()):.4f}]"
            )
            s_np = s[0].numpy() if s.ndim == 3 else s.numpy()

            # Robust symmetric stretch (2nd/98th percentiles) for visualization
            p_lo = float(np.percentile(s_np, 2))
            p_hi = float(np.percentile(s_np, 98))
            max_abs = max(abs(p_lo), abs(p_hi), np.finfo(np.float32).eps)
            s_norm = np.clip(s_np / max_abs, -1.0, 1.0)

            # Show diverging map centered at zero with fixed vmin/vmax
            ax.imshow(s_norm, cmap="seismic", vmin=-1.0, vmax=1.0)
        ax.set_title(f"Scale {scale} Seismic")
        ax.axis("off")

    plt.tight_layout()
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pyramid_verification.png"
    plt.savefig(out_path, dpi=150)  # type: ignore
    plt.close()
    print(f"Pyramid verification plot saved to {out_path}")


if __name__ == "__main__":
    main()
