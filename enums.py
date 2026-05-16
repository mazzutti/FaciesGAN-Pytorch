"""Centralized string enumerations for FaciesGAN.

This module provides StrEnum classes to replace hardcoded strings
across the codebase, improving type safety and maintainability.
"""

from enum import Enum, IntEnum, StrEnum


class FaciesClass(IntEnum):
    """Numeric IDs for facies classes."""

    FLOODPLAIN = 0
    POINT_BAR = 1
    CHANNEL = 2
    BOUNDARY = 3


class MetricKey(StrEnum):
    """Keys used for logging discriminator and generator metrics."""

    D_TOTAL = "d_total"
    D_REAL = "d_real"
    D_FAKE = "d_fake"
    D_GP = "d_gp"

    G_TOTAL = "g_total"
    G_FAKE = "g_fake"
    G_REC_FACIES = "g_rec_facies"
    G_WELL = "g_well"
    G_DIV = "g_div"
    G_REC_ROCK_PHYSICS = "g_rec_rock_physics"
    G_TV = "g_tv"
    G_ELASTIC = "g_elastic"
    G_SEISMIC = "g_seismic"

    @classmethod
    def generator_keys(cls) -> list["MetricKey"]:
        return [
            cls.G_FAKE,
            cls.G_REC_FACIES,
            cls.G_WELL,
            cls.G_DIV,
            cls.G_REC_ROCK_PHYSICS,
            cls.G_TV,
            cls.G_ELASTIC,
            cls.G_SEISMIC,
        ]


class FeatureKey(StrEnum):
    """Keys for feature splitting and representation."""

    FACIES = "facies"
    ROCK_PHYSICS = "rock_physics"
    WELLS = "wells"
    SEISMIC = "seismic"
    IP = "Ip"
    IS = "Is"
    VPVS = "VpVs"


class DeviceType(StrEnum):
    """Supported torch device types."""

    CUDA = "cuda"
    CPU = "cpu"


class AmpDtype(StrEnum):
    """AMP (Automatic Mixed Precision) dtype choices."""

    FP16 = "fp16"
    BF16 = "bf16"


class DdpBackend(StrEnum):
    """Supported torch.distributed process-group backends."""

    NCCL = "nccl"
    GLOO = "gloo"


class LrDecayUnit(StrEnum):
    """Unit of time for learning-rate decay scheduling."""

    EPOCH = "epoch"
    BATCH = "batch"
    STEP = "step"


class InterpolationStrategy(StrEnum):
    """Interpolation strategies for multiscale pyramids."""

    CATEGORICAL = "categorical"
    CONTINUOUS = "continuous"


class LossFunction(StrEnum):
    """Supported loss functions for geophysical consistency."""

    HUBER = "huber"
    MSE = "mse"


class InterpolationMode(StrEnum):
    """PyTorch interpolation modes."""

    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"


class EmbeddingMethod(StrEnum):
    """Available manifold learning methods for latent space analysis."""

    ISOMAP = "isomap"
    MDS = "mds"
    TSNE = "tsne"
    UMAP = "umap"


from dataclasses import dataclass


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


class SplitKey(StrEnum):
    FACIES = "facies"
    ROCK_PHYSICS = "rock_physics"
    WELLS = "wells"
    SEISMIC = "seismic"
    MASKS = "masks"


class ChannelKey(StrEnum):
    FACIES = "facies"
    ROCK_PHYSICS = "rock_physics"
    GENERATOR_OUT = "generator_out"
    DISCRIMINATOR_IN = "discriminator_in"
    NOISE = "noise"


class LossFn(StrEnum):
    HUBER = "huber"
    MSE = "mse"


class DataFiles(IntEnum):
    FACIES = 1
    WELLS = 2
    MASKS = 3
    SEISMIC = 4
    Ip = 5
    Is = 6
    VP_VS = 7
    VP = 8
    VS = 9
    RHO = 10

    @classmethod
    def generator_output_rock_physics(cls):
        return [cls.Ip, cls.Is, cls.VP_VS]

    @classmethod
    def rock_physics_names(cls):
        return [c.name for c in cls.generator_output_rock_physics()]

    @classmethod
    def loss_only_rock_physics(cls):
        return [cls.VP, cls.VS, cls.RHO]

    @classmethod
    def all_rock_physics(cls):
        return cls.generator_output_rock_physics() + cls.loss_only_rock_physics()


class StatKey(StrEnum):
    """String keys for global statistics dictionaries (min/max/mean)."""

    MIN = "min"
    MAX = "max"
    MEAN = "mean"
