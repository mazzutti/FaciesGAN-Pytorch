"""Centralized string enumerations for FaciesGAN.

This module provides StrEnum classes to replace hardcoded strings 
across the codebase, improving type safety and maintainability.
"""

from enum import IntEnum, StrEnum

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
    G_PHYSICS = "g_physics"


class FeatureKey(StrEnum):
    """Keys for feature splitting and representation."""
    FACIES = "facies"
    ROCK_PHYSICS = "rock_physics"
    WELLS = "wells"
    SEISMIC = "seismic"
    IP = "Ip"
    IS = "Is"
    VPVS = "VpVs"


class InterpolationStrategy(StrEnum):
    """Interpolation strategies for multi-scale pyramids."""
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
