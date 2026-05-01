import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch

import utils
from datasets.dataset import TorchPyramidsDataset
from models.palette import PALETTE_RGB
from options import TrainingOptions
from utils import denorm


def main():
    opt = TrainingOptions()
    opt.use_rock_physics = True
    opt.use_seismic = True
    opt.use_wells = True
    opt.num_facies_classes = 4

    # Load dataset
    print("Loading dataset...")
    dataset = TorchPyramidsDataset(opt)

    num_scales = len(dataset.scales)
    num_facies = opt.num_facies_classes

    _, axes = plt.subplots(num_scales, 6, figsize=(18, 3 * num_scales))  # type: ignore

    for scale in range(num_scales):
        # get_scale_data returns (facies, wells, masks, seismic)
        # Note: 'facies' actually contains [Facies | Ip | Is | VpVs] if use_rock_physics=True
        facies_batch, wells_batch, _, seismic_batch = dataset.get_scale_data(scale)

        idx = 0
        if facies_batch.shape[0] == 0:
            continue

        f = facies_batch[idx].detach().cpu()
        w = wells_batch[idx].detach().cpu() if wells_batch.shape[0] > idx else None
        s = seismic_batch[idx].detach().cpu() if seismic_batch.shape[0] > idx else None

        # Split channels based on configuration
        ip = f[num_facies] if f.shape[0] > num_facies else None
        is_ = f[num_facies + 1] if f.shape[0] > num_facies + 1 else None
        vpvs = f[num_facies + 2] if f.shape[0] > num_facies + 2 else None

        # 1. Facies (One-hot argmax for plotting)
        ax = axes[scale, 0]
        f_idx = torch.argmax(f[:num_facies], dim=0).numpy()
        print(
            f"Scale {scale} Facies raw range (normalized): [{f.min():.4f}, {f.max():.4f}]"
        )
        ax.imshow(utils.facies_to_rgb(f_idx).transpose(1, 2, 0))
        ax.set_title(f"Scale {scale} Facies")
        ax.axis("off")

        # 2. Rock Physics: Ip
        ax = axes[scale, 1]
        if ip is not None:
            print(
                f"Scale {scale} Ip range (normalized): [{ip.min():.4f}, {ip.max():.4f}]"
            )
            ax.imshow(denorm(ip), cmap="magma")
        ax.set_title(f"Scale {scale} Ip")
        ax.axis("off")

        # 3. Rock Physics: Is
        ax = axes[scale, 2]
        if is_ is not None:
            print(
                f"Scale {scale} Is range (normalized): [{is_.min():.4f}, {is_.max():.4f}]"
            )
            ax.imshow(denorm(is_), cmap="magma")
        ax.set_title(f"Scale {scale} Is")
        ax.axis("off")

        # 4. Rock Physics: Vp/Vs
        ax = axes[scale, 3]
        if vpvs is not None:
            print(
                f"Scale {scale} Vp/Vs range (normalized): [{vpvs.min():.4f}, {vpvs.max():.4f}]"
            )
            ax.imshow(denorm(vpvs), cmap="viridis")
        ax.set_title(f"Scale {scale} Vp/Vs")
        ax.axis("off")

        # 5. Wells
        ax = axes[scale, 4]
        if w is not None:
            print(
                f"Scale {scale} Wells raw range (normalized): [{w.min():.4f}, {w.max():.4f}]"
            )
            well_cmap = mcolors.ListedColormap(PALETTE_RGB)
            w_denorm = denorm(w)  # (4, H, W)
            # Collapse wells to indices
            if isinstance(w_denorm, np.ndarray):
                w_denorm = torch.tensor(np.asarray(w_denorm), dtype=torch.float32)
            w_idx = torch.argmax(w_denorm, dim=0).numpy().astype(float)
            # Mask where no well exists (sum of channels is 0 in denorm space? No, in norm space is -1)
            # In denorm space, it's 0.
            mask = (w_denorm.sum(dim=0) == 0).numpy()
            w_idx[mask] = np.nan
            ax.imshow(w_idx, cmap=well_cmap)
        ax.set_title(f"Scale {scale} Wells")
        ax.axis("off")

        # 6. Seismic
        ax = axes[scale, 5]
        if s is not None:
            print(
                f"Scale {scale} Seismic range (normalized): [{s.min():.4f}, {s.max():.4f}]"
            )
            s_np = s[0].numpy() if s.ndim == 3 else s.numpy()
            ax.imshow(s_np, cmap="seismic")
        ax.set_title(f"Scale {scale} Seismic")
        ax.axis("off")

    plt.tight_layout()
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "pyramid_verification.png")
    plt.savefig(out_path, dpi=150)  # type: ignore
    plt.close()
    print(f"✅ Pyramid verification plot saved to {out_path}")


if __name__ == "__main__":
    main()
