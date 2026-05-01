"""Helper enum describing dataset components and .npz archives.

This module centralizes dataset component names used by data-loading 
and interpolation utilities. All data is assumed to be stored in 
compressed NumPy (.npz) archives.
"""

from enum import IntEnum

# Default base directory for datasets
DEFAULT_DATA_DIR = "./data"


class DataFiles(IntEnum):
    """Constants for dataset components stored in the data directory.

    Each member name corresponds to the base name of the .npz archive 
    (e.g., FACIES -> facies.npz).
    """

    # Dataset components
    FACIES = 1
    WELLS = 2
    MASKS = 3
    SEISMIC = 4
    Ip = 5
    Is = 6
    VP_VS = 7

    # Rock physics components
    VP = 8
    VS = 9
    RHO = 10

    @classmethod
    def generator_output_rock_physics(cls) -> list["DataFiles"]:
        """Return rock physics components produced by the generator (Ip, Is, Vp/Vs)."""
        return [cls.Ip, cls.Is, cls.VP_VS]

    @classmethod
    def loss_only_rock_physics(cls) -> list["DataFiles"]:
        """Return rock physics components used only for loss calculation (Vp, Vs, Rho)."""
        return [cls.VP, cls.VS, cls.RHO]

    @classmethod
    def all_rock_physics(cls) -> list["DataFiles"]:
        """Return all rock physics components."""
        return cls.generator_output_rock_physics() + cls.loss_only_rock_physics()
