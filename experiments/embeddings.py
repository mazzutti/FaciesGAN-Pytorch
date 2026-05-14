"""Manifold learning pipeline for latent space visualization.

This module provides a robust, CPU-based pipeline for dimensionality reduction 
and manifold embedding (t-SNE, UMAP, MDS, Isomap). It enforces absolute 
reproducibility through deterministic seeding and optimizes performance for 
high-dimensional facies volumes using a 100-component PCA pre-reduction step.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    cast,
    runtime_checkable,
)

import numpy as np
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed  # type: ignore[import]
from scipy.sparse import SparseEfficiencyWarning
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE, Isomap
from umap import UMAP  # type: ignore[import]

from config import CheckpointFilenames
from datasets.dataset import PyramidsDataset
from device import device_manager
from enums import EmbeddingMethod

# Suppress specific library warnings that clutter the console
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

# --- Type Aliases ---
# method_name -> (real_embedding, {variant_name: generated_embedding})
_SharedEmbeddings = Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray]]]

logger = logging.getLogger(__name__)


@dataclass
class ExperimentCache:
    """Container for cached embeddings and samples across all ablation variants.

    Attributes:
        shared: Manifold results (2D coordinates) for various methods.
        all_facies: Generated facies volumes (N, C, H, W) per variant.
        all_mask_indexes: Ground-truth conditioning indices used per variant.
        all_ip: Acoustic Impedance volumes per variant.
        all_is: Shear Impedance volumes per variant.
        all_vpvs: Vp/Vs ratio volumes per variant.
        all_seismic: Synthetic seismic volumes per variant.
    """

    shared: _SharedEmbeddings
    all_facies: Dict[str, List[np.ndarray]]
    all_mask_indexes: Dict[str, torch.Tensor]
    all_ip: Dict[str, List[np.ndarray]]
    all_is: Dict[str, List[np.ndarray]]
    all_vpvs: Dict[str, List[np.ndarray]]
    all_seismic: Dict[str, List[np.ndarray]]


@runtime_checkable
class ManifoldEstimator(Protocol):
    """Structural protocol for Scikit-learn style manifold estimators."""

    def fit_transform(self, X: np.ndarray, y: Any = None) -> np.ndarray: ...


def get_manifold_estimator(
    method: str, n_components: int = 2, random_state: int = 42
) -> Optional[ManifoldEstimator]:
    """Factory to initialize CPU-based manifold estimators with consistent parameters.

    Args:
        method: The embedding method (tsne, umap, mds, or isomap).
        n_components: Target dimensionality (usually 2 for visualization).
        random_state: Seed for deterministic results.

    Returns:
        An initialized estimator instance, or None if the method is unknown.
    """
    m = method.lower()

    # Configuration rationale:
    # - t-SNE: PCA init is more stable and preserves global structure better.
    # - UMAP: Spectral init is faster and provides good initial layout.
    # - MDS: n_init=1 since we are already in a low-dimensional PCA subspace.
    # - Isomap: 30 neighbors provides a good balance between local/global connectivity.

    estimator: Any = None
    if m == EmbeddingMethod.TSNE:
        estimator = TSNE(
            n_components=n_components,
            init="pca",
            random_state=random_state,
            n_jobs=1,
        )
    elif m == EmbeddingMethod.UMAP:
        estimator = UMAP(
            n_components=n_components,
            init="spectral",
            random_state=random_state,
            n_jobs=1,
        )
    elif m == EmbeddingMethod.MDS:
        estimator = MDS(
            n_components=n_components,
            n_init=1,
            max_iter=100,
            random_state=random_state,
            normalized_stress="auto",
            init="random",  # type: ignore[call-arg]
        )
    elif m == EmbeddingMethod.ISOMAP:
        estimator = Isomap(n_neighbors=30, n_components=n_components, n_jobs=1)

    return cast(Optional[ManifoldEstimator], estimator)


def load_shared_embeddings(
    base_output: str, num_iter: int
) -> Optional[ExperimentCache]:
    """Loads cached embeddings from disk if they match the current experiment state.

    Args:
        base_output: Directory containing the experiment results.
        num_iter: The current training epoch to validate cache staleness.

    Returns:
        The cached ExperimentCache object, or None if missing or stale.
    """
    path = Path(base_output) / CheckpointFilenames.EMBEDDINGS
    if not path.is_file():
        return None

    try:
        data = np.load(path, allow_pickle=True)
        if data.get("num_iter", -1) != num_iter:
            return None

        return ExperimentCache(
            shared=data["shared"].item(),
            all_facies=data["all_facies"].item(),
            all_mask_indexes=data["all_mi"].item(),
            all_ip=data.get("all_ip", {}).item(),
            all_is=data.get("all_is", {}).item(),
            all_vpvs=data.get("all_vpvs", {}).item(),
            all_seismic=data.get("all_seismic", {}).item(),
        )
    except (IOError, KeyError, ValueError) as e:
        logger.warning(f"Failed to load embedding cache from {path}: {e}")
        return None


def save_shared_embeddings(
    cache: ExperimentCache, base_output: str, num_iter: int
) -> None:
    """Serializes the experiment cache to disk.

    Args:
        cache: The ExperimentCache instance to save.
        base_output: Directory where the .npz file will be created.
        num_iter: Metadata indicating the epoch at which this was saved.
    """
    path = Path(base_output) / CheckpointFilenames.EMBEDDINGS
    np.savez(
        path,
        shared=cache.shared,  # type: ignore
        all_facies=cache.all_facies,  # type: ignore
        all_mi=cache.all_mask_indexes,  # type: ignore
        all_ip=cache.all_ip,  # type: ignore
        all_is=cache.all_is,  # type: ignore
        all_vpvs=cache.all_vpvs,  # type: ignore
        all_seismic=cache.all_seismic,  # type: ignore
        num_iter=num_iter,
    )


def _flatten_data(data: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
    """Standardizes input data into a 2D float32 array for ML estimators."""
    arr = np.array(data) if isinstance(data, list) else data
    return arr.reshape(arr.shape[0], -1).astype(np.float32)


def prepare_features(
    dataset: PyramidsDataset,
    samples: Union[np.ndarray, List[np.ndarray]],
    rock_physics_only: bool = False,
    seismic_only: bool = False,
    channel_index: Optional[int] = None,
) -> np.ndarray:
    """Pre-processes multi-channel volumes into feature vectors for manifold learning.

    This function handles the complex channel mapping between Facies (categorical/RGB),
    Rock Physics (Ip, Is, VpVs), and Seismic (Greyscale) data.

    Args:
        dataset: The source dataset (provides channel configurations).
        samples: Input volumes of shape (N, C, H, W).
        rock_physics_only: If True, extracts only the rock physics channels.
        seismic_only: If True, extracts only the synthetic seismic channel.
        channel_index: Specific channel to extract (e.g., only Ip).

    Returns:
        A flattened (N, Features) array ready for PCA.
    """
    arr = np.array(samples)
    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=1)

    n_facies = int(dataset.options.num_facies_channels or 4)

    if seismic_only:
        # Seismic is typically the last channel (-1)
        if arr.shape[1] > 1:
            arr = arr[:, -1:, ...]
    elif rock_physics_only:
        # Rock physics channels follow the facies channels.
        if arr.shape[1] == n_facies + 3:
            idx = channel_index if channel_index is not None else n_facies
        elif arr.shape[1] == 4:
            idx = (channel_index - (n_facies - 1)) if channel_index is not None else 1
        elif arr.shape[1] == 3:
            idx = (
                (channel_index - n_facies)
                if (channel_index is not None and channel_index >= n_facies)
                else 0
            )
        else:
            idx = 0
        arr = arr[:, idx : idx + 1, ...]
    else:
        # Default: Extract only facies channels (structural features)
        if arr.shape[1] > n_facies:
            arr = arr[:, :n_facies, ...]

    # Ensure consistent spatial resolution (resize to crop_size if needed)
    target_h, target_w = int(dataset.options.crop_size or 128), int(
        dataset.options.crop_size or 128
    )
    if arr.shape[2] != target_h or arr.shape[3] != target_w:
        tensor = torch.tensor(arr, dtype=torch.float32)
        arr = F.interpolate(tensor, size=(target_h, target_w), mode="bilinear").numpy()

    return _flatten_data(arr)


def _fit_single_method(
    method: str, data: np.ndarray, seed: int
) -> Tuple[str, Optional[np.ndarray]]:
    """Helper for parallel execution. Fits a single manifold estimator to the data."""
    estimator = get_manifold_estimator(method, random_state=seed)
    if not estimator:
        return method, None
    try:
        embedding = estimator.fit_transform(data)
        return method, embedding
    except Exception as e:
        logger.error(f"Manifold fitting failed for {method.upper()}: {e}")
        return method, None


def compute_shared_embeddings(
    all_gen: Dict[str, List[np.ndarray]],
    dataset: PyramidsDataset,
    methods: Optional[List[str]] = None,
    rock_physics_only: bool = False,
    seismic_only: bool = False,
    channel_index: Optional[int] = None,
) -> _SharedEmbeddings:
    """Orchestrates the full manifold embedding pipeline.

    The pipeline consists of:
    1. Deterministic seeding (Torch & Numpy).
    2. Data flattening and channel extraction.
    3. PCA dimensionality reduction to 100 components.
    4. Parallel fitting of manifold estimators.

    Args:
        all_gen: Dictionary mapping variant names to lists of generated volumes.
        dataset: The ground-truth dataset for real-sample comparison.
        methods: List of manifold methods to compute.
        rock_physics_only: Filter for rock physics features.
        seismic_only: Filter for seismic features.
        channel_index: specific channel filter.

    Returns:
        A dictionary mapping method names to (real_embedding, {variant: fake_embedding}).
    """
    if methods is None:
        methods = [m.value for m in EmbeddingMethod]

    # Enforce absolute reproducibility across all libraries
    seed: int = int(dataset.options.manual_seed or 42)
    torch.manual_seed(seed)  # type: ignore
    np.random.seed(seed)

    # 1. Prepare Ground Truth (Real) Features
    real_tensor, _, _, real_seismic_tensor = dataset.get_scale_data(-1)
    if seismic_only:
        real_flat = _flatten_data(device_manager.to_numpy(real_seismic_tensor))
    else:
        real_flat = prepare_features(
            dataset,
            device_manager.to_numpy(real_tensor),
            rock_physics_only,
            False,
            channel_index,
        )

    # 2. Prepare Generated (Fake) Features
    f_list: List[np.ndarray] = []
    variant_offsets: Dict[str, Tuple[int, int]] = {}
    current_pos = len(real_flat)

    for name, samples in all_gen.items():
        f_flat = prepare_features(
            dataset, samples, rock_physics_only, seismic_only, channel_index
        )
        f_list.append(f_flat)
        variant_offsets[name] = (current_pos, current_pos + len(f_flat))
        current_pos += len(f_flat)

    if not f_list:
        return {}

    all_data = np.concatenate([real_flat] + f_list, axis=0)

    # 3. PCA Dimensionality Reduction (Stability Step)
    n_pca = 100
    if all_data.shape[1] > n_pca:
        print(
            f"    [PCA] Reducing {all_data.shape[1]} features → {n_pca} (seed={seed})",
            flush=True,
        )
        pca = PCA(n_components=n_pca, svd_solver="randomized", random_state=seed)
        all_data = pca.fit_transform(all_data)

    # 4. Parallel Manifold Fitting
    print(
        f"    Fitting {len(methods)} methods in parallel on {all_data.shape[0]} samples:",
        flush=True,
    )
    for m in methods:
        print(f"      → {m.upper()} ...", flush=True)

    # We use n_jobs=1 (sequential) to avoid deadlocks that occur with threaded backends
    # in some environments (e.g., when running inside a debugger).
    results_raw = cast(
        List[Tuple[str, Optional[np.ndarray]]],
        Parallel(n_jobs=1)(
            delayed(_fit_single_method)(m, all_data, seed) for m in methods
        ),
    )
    parallel_results = results_raw

    # 5. Assemble and Split Results (Real vs Variants)
    final_results: _SharedEmbeddings = {}
    for method, embedding in parallel_results:
        if embedding is not None:
            final_results[method] = (
                embedding[: len(real_flat)],
                {
                    name: embedding[start:end]
                    for name, (start, end) in variant_offsets.items()
                },
            )

    return final_results
