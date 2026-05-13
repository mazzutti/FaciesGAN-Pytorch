"""Common type definitions for the FaciesGAN project.

This module centralizes type aliases used for static analysis and runtime
type checking to ensure consistency across the codebase.
"""

from pathlib import Path
from typing import IO, Any, Union

# Filesystem paths or stream-like objects
FileLike = Union[str, Path, IO[Any]]
