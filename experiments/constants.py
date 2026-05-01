"""Shared constants for the experiments module."""

from enum import Enum


class EmbeddingMethod(str, Enum):
    """Available manifold learning methods for latent space analysis."""

    ISOMAP = "isomap"
    MDS = "mds"
    TSNE = "tsne"
    UMAP = "umap"


VARIANTS = {
    "wells_seismic": {"use_wells": True, "use_seismic": True, "use_rock_physics": True},
    "wells_only": {"use_wells": True, "use_seismic": False, "use_rock_physics": True},
    "seismic_only": {"use_wells": False, "use_seismic": True, "use_rock_physics": True},
    "unconditional": {
        "use_wells": False,
        "use_seismic": False,
        "use_rock_physics": True,
    },
}

VARIANT_NAMES = list(VARIANTS.keys())

VARIANT_LABELS = {
    "wells_seismic": "Wells + Seismic",
    "wells_only": "Wells Only",
    "seismic_only": "Seismic Only",
    "unconditional": "Unconditional",
}

EMBEDDINGS_FILE = "cached_embeddings.npz"
