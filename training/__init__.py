from .metrics import DiscriminatorMetrics as DiscriminatorMetrics
from .metrics import GeneratorMetrics as GeneratorMetrics
from .metrics import IterableMetrics as IterableMetrics
from .metrics import MetricSmoother as MetricSmoother
from .metrics import ScaleMetrics as ScaleMetrics

__all__ = [
    "DiscriminatorMetrics",
    "GeneratorMetrics",
    "ScaleMetrics",
    "IterableMetrics",
    "MetricSmoother",
]


def __getattr__(name: str):
    if name == "Trainer":
        from .trainer import Trainer

        __all__.append("Trainer")

        return Trainer
    raise AttributeError(f"module 'training' has no attribute {name!r}")
