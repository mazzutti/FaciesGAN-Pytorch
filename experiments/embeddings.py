"""Manifold learning pipeline for latent space visualization.

This module provides a robust, CPU-based pipeline for dimensionality reduction
and manifold embedding (t-SNE, UMAP, MDS, Isomap). It enforces absolute
reproducibility through deterministic seeding.
"""

from __future__ import annotations

import logging
import math
import os
import traceback
import warnings
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
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
from scipy.sparse import SparseEfficiencyWarning, coo_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE, Isomap

# joblib removed: use ThreadPoolExecutor for parallelism to avoid multiprocessing issues
from sklearn.metrics import euclidean_distances
from umap import UMAP  # type: ignore[import]


def _get_numba_threads() -> int:
    try:
        import numba  # type: ignore

        _threads = getattr(numba.config, "NUMBA_NUM_THREADS", None)
        if _threads is not None:
            return int(_threads)
    except Exception:
        pass
    return os.cpu_count() or 1


_NUMBA_THREADS = _get_numba_threads()

_MAX_JOBS = min(12, _NUMBA_THREADS)

from config import CheckpointFilenames
from datasets.dataset import PyramidsDataset
from device import device_manager
from enums import EmbeddingMethod

# Suppress specific library warnings that clutter the console
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
# Suppress sklearn MDS FutureWarnings about default `init` and deprecated `dissimilarity`
warnings.filterwarnings(
    "ignore",
    message=r".*The default value of `init` will change.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*The `dissimilarity` parameter is deprecated.*",
    category=FutureWarning,
)

# --- Type Aliases ---
# method_name -> (real_embedding, {variant_name: generated_embedding})
_SharedEmbeddings = Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray]]]

# Future result type for threaded manifold fitting tasks
_FutureResult = Tuple[
    str,
    Optional[
        Union[
            np.ndarray,
            Tuple[Any, ...],
            coo_matrix,
        ]
    ],
]

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

    # Model factory to avoid if/elif chain
    # Note: Using n_jobs=1 to avoid issues with debugger/multiprocessing on some systems
    model_factories: dict[str, Callable[[], Any]] = {
        EmbeddingMethod.MDS: lambda: MDS(
            n_components=n_components,
            n_init=1,
            max_iter=100,
            random_state=random_state,
            n_jobs=_MAX_JOBS,
        ),
        EmbeddingMethod.TSNE: lambda: TSNE(
            n_components=n_components,
            n_jobs=_MAX_JOBS,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        ),
        EmbeddingMethod.ISOMAP: lambda: Isomap(
            n_components=n_components,
            n_jobs=_MAX_JOBS,
            eigen_solver="arpack",
        ),
        EmbeddingMethod.UMAP: lambda: (
            UMAP(
                n_components=n_components,
                random_state=None,
                n_jobs=_MAX_JOBS,
            )
            if UMAP
            else None
        ),
    }

    if m not in model_factories:
        return None

    estimator = model_factories[m]()
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

    n_facies = int(dataset.options.num_facies_channels or 3)

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
        # Perform interpolation on the active device for speed, then move to CPU
        tensor = torch.as_tensor(arr, dtype=torch.float32, device=device_manager.device)
        resized = F.interpolate(tensor, size=(target_h, target_w), mode="bilinear")
        arr = device_manager.to_cpu(resized, non_blocking=True).numpy()

    return _flatten_data(arr)


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
        if real_seismic_tensor.numel() == 0 or len(real_seismic_tensor.shape) < 4:
            target_h = int(dataset.options.crop_size or 128)
            target_w = int(dataset.options.crop_size or 128)
            real_flat = np.empty((0, target_h * target_w), dtype=np.float32)
        else:
            real_flat = prepare_features(
                dataset,
                device_manager.to_numpy(real_seismic_tensor),
                rock_physics_only=False,
                seismic_only=True,
            )
    else:
        if real_tensor.numel() == 0 or len(real_tensor.shape) < 4:
            target_h = int(dataset.options.crop_size or 128)
            target_w = int(dataset.options.crop_size or 128)
            n_channels = int(dataset.options.num_facies_channels or 3)
            if rock_physics_only:
                n_channels = 1
            real_flat = np.empty(
                (0, n_channels * target_h * target_w), dtype=np.float32
            )
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

    # Apply sample-wise Z-score normalization to seismic features to ensure
    # we compare structural geometry/phase independent of convolved amplitude/gain differences.
    if seismic_only:
        means = all_data.mean(axis=1, keepdims=True)
        stds = all_data.std(axis=1, keepdims=True) + 1e-6
        all_data = (all_data - means) / stds

    # 4. Parallel Manifold Fitting (aligned with remote `experiments.py`):
    # - Use PCA(2) init for MDS
    # - Precompute distances for MDS
    # - Dynamic parameter selection for UMAP/TSNE/Isomap based on sample count
    # - Run methods in parallel using threads to avoid multiprocessing deadlocks
    n_samples = all_data.shape[0]
    print(f"Fitting {len(methods)} methods in parallel on {n_samples} samples:")
    for m in methods:
        print(f" -> {m.upper()} ...")

    # PCA pre-reduction: for high-dimensional data (e.g. flattened 256×256 = 65536-dim
    # volumes) all manifold methods benefit from a dimensionality reduction step.
    # UMAP docs explicitly recommend this; it also makes MDS/Isomap/t-SNE faster.
    # Cap at min(50, n_samples-1) to stay well within scikit-learn's constraints.
    if all_data.shape[1] > 50:
        n_pca_pre = min(50, n_samples - 1)
        all_data = PCA(n_components=n_pca_pre, random_state=42).fit_transform(all_data)
        print(f"  PCA pre-reduction: -> {all_data.shape[1]} components", flush=True)

    # PCA init for MDS (deterministic seed chosen to match remote script)
    pca_init = PCA(n_components=2, random_state=3).fit_transform(all_data)

    # Precompute pairwise Euclidean distances only if MDS is requested
    distances = None
    if any(m.lower() == EmbeddingMethod.MDS for m in methods):
        distances = euclidean_distances(all_data)

    # Threaded execution with max_workers=1 to run sequentially and avoid heavy thread contention / OOM / deadlocks
    results_raw: List[Tuple[str, Optional[np.ndarray]]] = []
    with ThreadPoolExecutor(max_workers=1) as exe:
        futures: Dict[Future[_FutureResult], str] = {}

        def submit_method(method_name: str) -> Future[_FutureResult]:
            m_lower = method_name.lower()

            if m_lower == EmbeddingMethod.MDS:

                def run_mds():
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", FutureWarning)
                            mds = MDS(
                                n_components=2,
                                dissimilarity="precomputed",
                                init="random",  # type: ignore[assignment]
                                n_init=1,
                                max_iter=300,
                                eps=1e-6,
                                random_state=seed,
                                n_jobs=_MAX_JOBS,
                            )
                            emb = mds.fit_transform(distances, init=pca_init)  # type: ignore[arg-type]
                        return method_name, emb
                    except Exception as e:
                        msg = f"MDS fitting failed: {type(e).__name__}: {e}"
                        logger.error(msg)
                        traceback.print_exc()
                        print(f"[ERROR] {msg}", flush=True)
                        return method_name, None

                return cast(Future[_FutureResult], exe.submit(run_mds))

            if m_lower == EmbeddingMethod.UMAP:

                def run_umap():  # type: ignore
                    try:
                        # Scale n_neighbors with sqrt(n_samples) to maintain connectivity
                        # for large datasets (e.g. 4001 samples -> ~63 neighbors)
                        n_nb = max(5, int(math.sqrt(n_samples)))
                        try:
                            um = UMAP(
                                n_components=2,
                                n_neighbors=n_nb,
                                min_dist=0.05,
                                n_epochs=500,
                                metric="euclidean",
                                init="spectral",
                                random_state=None,
                                n_jobs=_MAX_JOBS,
                            )
                            emb = um.fit_transform(all_data)  # type: ignore
                        except Exception as spectral_err:
                            print(
                                f"[WARNING] UMAP spectral initialization failed: {spectral_err}. "
                                "Retrying with random initialization...",
                                flush=True,
                            )
                            um = UMAP(
                                n_components=2,
                                n_neighbors=n_nb,
                                min_dist=0.05,
                                n_epochs=500,
                                metric="euclidean",
                                init="random",
                                random_state=None,
                                n_jobs=_MAX_JOBS,
                            )
                            emb = um.fit_transform(all_data)  # type: ignore
                        return method_name, emb  # type: ignore
                    except Exception as e:
                        # Use print() as well: runner.py disables logging at CRITICAL
                        # so logger.error() would be silently swallowed.
                        msg = f"UMAP fitting failed: {type(e).__name__}: {e}"
                        logger.error(msg)
                        traceback.print_exc()
                        print(f"[ERROR] {msg}", flush=True)
                        return method_name, None

                return cast(Future[_FutureResult], exe.submit(run_umap))

            if m_lower == EmbeddingMethod.ISOMAP:

                def run_isomap():
                    try:
                        # Scale n_neighbors with sqrt(n_samples)/2 to maintain connectivity
                        # for large datasets (e.g. 4001 samples -> ~32 neighbors)
                        n_nb = max(5, int(math.sqrt(n_samples) / 2))
                        iso = Isomap(
                            n_components=2,
                            n_neighbors=n_nb,
                            eigen_solver="arpack",
                            neighbors_algorithm="kd_tree",
                            metric="euclidean",
                            n_jobs=_MAX_JOBS,
                        )
                        emb = iso.fit_transform(all_data)
                        return method_name, emb
                    except Exception as e:
                        msg = f"Isomap fitting failed: {type(e).__name__}: {e}"
                        logger.error(msg)
                        traceback.print_exc()
                        print(f"[ERROR] {msg}", flush=True)
                        return method_name, None

                return cast(Future[_FutureResult], exe.submit(run_isomap))

            if m_lower == EmbeddingMethod.TSNE:

                def run_tsne():
                    try:
                        perp = min(30.0, float(max(1, (n_samples - 1) / 3.0)))
                        ts = TSNE(
                            n_components=2,
                            max_iter=1500,
                            init="pca",
                            learning_rate="auto",
                            perplexity=perp,
                            random_state=seed,
                            n_jobs=_MAX_JOBS,
                        )
                        emb = ts.fit_transform(all_data)
                        return method_name, emb
                    except Exception as e:
                        msg = f"t-SNE fitting failed: {type(e).__name__}: {e}"
                        logger.error(msg)
                        traceback.print_exc()
                        print(f"[ERROR] {msg}", flush=True)
                        return method_name, None

                return cast(Future[_FutureResult], exe.submit(run_tsne))

            # Unknown method
            return cast(Future[_FutureResult], exe.submit(lambda: (method_name, None)))

        for m in methods:
            fut = submit_method(m)
            futures[fut] = m

        for fut in as_completed(futures.keys()):
            try:
                # cast the (heterogeneous) future result into the expected
                # Tuple[str, Optional[np.ndarray]] for downstream processing.
                results_raw.append(cast(Tuple[str, Optional[np.ndarray]], fut.result()))
            except Exception as e:
                m = futures.get(fut, "?")
                msg = f"Manifold fitting failed for {m}: {type(e).__name__}: {e}"
                logger.error(msg)
                print(f"[ERROR] {msg}", flush=True)
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
