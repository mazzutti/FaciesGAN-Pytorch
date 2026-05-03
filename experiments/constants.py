"""Shared constants for the experiments module."""

from enum import Enum


class EmbeddingMethod(str, Enum):
    """Available manifold learning methods for latent space analysis."""

    ISOMAP = "isomap"
    MDS = "mds"
    TSNE = "tsne"
    UMAP = "umap"


from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class VariantConfig:
    use_wells: bool
    use_seismic: bool
    use_rock_physics: bool
    label: str


class ExperimentVariant(Enum):
    WELLS_SEISMIC = VariantConfig(
        use_wells=True,
        use_seismic=True,
        use_rock_physics=True,
        label="Wells + Seismic",
    )
    WELLS_ONLY = VariantConfig(
        use_wells=True,
        use_seismic=False,
        use_rock_physics=True,
        label="Wells Only",
    )
    SEISMIC_ONLY = VariantConfig(
        use_wells=False,
        use_seismic=True,
        use_rock_physics=True,
        label="Seismic Only",
    )
    UNCONDITIONAL = VariantConfig(
        use_wells=False,
        use_seismic=False,
        use_rock_physics=True,
        label="Unconditional",
    )

    @property
    def id(self) -> str:
        return self.name.lower()


EMBEDDINGS_FILE = "cached_embeddings.npz"
