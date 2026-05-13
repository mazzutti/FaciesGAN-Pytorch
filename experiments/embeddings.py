"""Latent space embedding computation and caching for experiments."""

import os
import warnings
from typing import Any, Callable

import numpy as np
import torch

# Suppress annoying sparse warnings from Isomap
from scipy.sparse import SparseEfficiencyWarning  # type: ignore[import]
from sklearn.manifold import MDS, TSNE, Isomap
from umap import UMAP  # type: ignore[import]

warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

from dataclasses import dataclass

from constants import EMBEDDINGS_FILE, EmbeddingMethod
from datasets import PyramidsDataset

# Type alias for shared embeddings: method -> (real_reduced, {variant: fake_reduced})
_SharedEmbeddings = dict[str, tuple[np.ndarray, dict[str, np.ndarray]]]


@dataclass
class ExperimentCache:
    """Encapsulates cached embeddings and generated samples across all variants."""

    shared: _SharedEmbeddings
    all_facies: dict[str, list[np.ndarray]]
    all_mask_indexes: dict[str, torch.Tensor]
    all_ip: dict[str, list[np.ndarray]]
    all_seismic: dict[str, list[np.ndarray]]


def load_shared_embeddings(base_output: str, num_iter: int) -> ExperimentCache | None:
    """Load cached embeddings and samples from disk if they exist."""
    path = os.path.join(base_output, EMBEDDINGS_FILE)
    if not os.path.isfile(path):
        return None

    try:
        # allow_pickle is needed because we save complex dicts of numpy arrays
        data = np.load(path, allow_pickle=True)

        # Check if the cache matches the requested epoch
        cached_epoch = data.get("num_iter", -1)
        if cached_epoch != num_iter:
            print(
                f"    [Cache] Epoch mismatch (cached: {cached_epoch}, requested: {num_iter}). Recomputing..."
            )
            return None

        # Reconstruct components from the NPZ archive
        # Dictionaries are stored as 0-d object arrays, so .item() extracts them
        shared = data["shared"].item()
        all_facies = data["all_facies"].item()
        all_mi = data["all_mi"].item()
        all_ip = data.get("all_ip", np.array({})).item()
        all_seismic = data.get("all_seismic", np.array({})).item()

        print(f"    [Cache] Successfully loaded embeddings from {path}")
        return ExperimentCache(
            shared=shared,
            all_facies=all_facies,
            all_mask_indexes=all_mi,
            all_ip=all_ip,
            all_seismic=all_seismic,
        )

    except Exception as e:
        print(f"    [Cache] Warning: Could not load {path} ({e}). Recomputing...")
        return None


def save_shared_embeddings(
    cache: ExperimentCache,
    base_output: str,
    num_iter: int,
) -> None:
    """Cache embeddings and generated samples to disk."""
    path = os.path.join(base_output, EMBEDDINGS_FILE)
    np.savez_compressed(
        path,
        shared=np.array(cache.shared, dtype=object),
        all_facies=np.array(cache.all_facies, dtype=object),
        all_mi=np.array(cache.all_mask_indexes, dtype=object),
        all_ip=np.array(cache.all_ip, dtype=object),
        all_seismic=np.array(cache.all_seismic, dtype=object),
        num_iter=num_iter,
    )


def _flatten_data(data: np.ndarray | list[np.ndarray]) -> np.ndarray:
    """Convert (N, H, W, C) or (N, H, W) data to (N, H*W*C) flat features."""
    arr = np.array(data) if isinstance(data, list) else data
    return arr.reshape(arr.shape[0], -1)


def prepare_features(
    dataset: PyramidsDataset,
    generated_samples: list[np.ndarray] | np.ndarray,
    rock_physics_only: bool = False,
    seismic_only: bool = False,
) -> np.ndarray:
    """Extract and flatten features from either real or generated samples."""
    # 1. Convert to array and handle layout
    arr = np.array(generated_samples)
    if arr.ndim == 3:
        # (N, H, W) -> (N, 1, H, W)
        arr = np.expand_dims(arr, axis=1)

    # 2. Extract relevant channels
    num_facies_channels = dataset.options.num_facies_channels
    if seismic_only:
        # Expecting (N, 1, H, W) or (N, C, H, W) - take last channel if multiple
        if arr.shape[1] > 1:
            arr = arr[:, -1:, ...]
    elif rock_physics_only:
        # If combined [F+RP], extract Ip (at num_facies_channels)
        if arr.shape[1] > num_facies_channels:
            arr = arr[:, num_facies_channels : num_facies_channels + 1, ...]
        else:
            # Already RP-only, take Ip (first channel)
            arr = arr[:, 0:1, ...]
    else:
        # Facies: if combined, extract only facies channels
        if arr.shape[1] > num_facies_channels:
            arr = arr[:, :num_facies_channels, ...]

    # 3. Ensure spatial dimensions match the expected scale (robustness against mixed resolutions)
    # We use the stop_scale size as the reference
    target_h = dataset.options.crop_size
    target_w = dataset.options.crop_size
    if arr.shape[2] != target_h or arr.shape[3] != target_w:
        import torch.nn.functional as F

        # Use torch.tensor to ensure a well-typed float tensor (avoids from_numpy typing issues)
        arr_t = torch.tensor(arr, dtype=torch.float32)
        arr_t = F.interpolate(arr_t, size=(target_h, target_w), mode="bilinear")
        arr = arr_t.numpy()

    return _flatten_data(arr)


def compute_shared_embeddings(
    all_generated: dict[str, list[np.ndarray]],
    dataset: PyramidsDataset,
    methods: list[str] | None = None,
    rock_physics_only: bool = False,
    seismic_only: bool = False,
) -> _SharedEmbeddings:
    """Fit dimensionality reduction on real data + ALL variants simultaneously."""

    if methods is None:
        methods = [m.value for m in EmbeddingMethod]

    # 1. Prepare Real Data
    real_tensor, _, _, real_seismic_tensor = dataset.get_scale_data(-1)
    if seismic_only:
        real_flat = _flatten_data(real_seismic_tensor.cpu().numpy())
    else:
        real_np = real_tensor.cpu().numpy()
        real_flat = prepare_features(dataset, real_np, rock_physics_only, False)

    # 2. Prepare Fake Data from all variants
    all_fakes_list: list[np.ndarray] = []
    variant_offsets: dict[str, tuple[int, int]] = {}
    current_off = len(real_flat)

    for name, samples in all_generated.items():
        f_flat = prepare_features(dataset, samples, rock_physics_only, seismic_only)
        all_fakes_list.append(f_flat)
        variant_offsets[name] = (current_off, current_off + len(f_flat))
        current_off += len(f_flat)

    if not all_fakes_list:
        return {}

    # Combine all into one large matrix for a shared fitting
    print(f"    Concatenating features: real_flat={real_flat.shape}")
    for i, f in enumerate(all_fakes_list):
        print(f"      f_flat[{i}]={f.shape}")

    all_data = np.concatenate([real_flat] + all_fakes_list, axis=0)

    # Model factory to avoid if/elif chain
    # Note: Using n_jobs=1 to avoid issues with debugger/multiprocessing on some systems
    model_factories: dict[str, Callable[[], Any]] = {
        EmbeddingMethod.MDS: lambda: MDS(
            n_components=2,
            n_init=1,
            max_iter=100,
            n_jobs=1,
        ),
        EmbeddingMethod.TSNE: lambda: TSNE(
            n_components=2, n_jobs=1, init="pca", learning_rate="auto"
        ),
        EmbeddingMethod.ISOMAP: lambda: Isomap(n_components=2, n_jobs=1),
        EmbeddingMethod.UMAP: lambda: UMAP(n_components=2, n_jobs=1) if UMAP else None,
    }

    results: _SharedEmbeddings = {}
    for m in methods:
        factory = model_factories.get(m.lower())
        if not factory:
            continue

        model = factory()
        if not model:
            print(f"    [Warning] {m.upper()} not available (check dependencies).")
            continue

        print(f"    Fitting {m.upper()} on {len(all_data)} samples ...", flush=True)
        try:
            reduced = model.fit_transform(all_data)
        except Exception as e:
            print(f"    [Error] {m.upper()} fitting failed: {e}")
            continue

        # Split back into real and per-variant fakes
        results[m] = (
            reduced[: len(real_flat)],
            {
                name: reduced[start:end]
                for name, (start, end) in variant_offsets.items()
            },
        )

    return results
