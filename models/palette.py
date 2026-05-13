"""Centralized color palette definitions for the FaciesGAN project.

This module provides a single source of truth for the facies color palettes
used across the codebase, including RGB [0, 1] and
One-Hot representations.
"""

from __future__ import annotations

import numpy as np

from enums import FaciesClass

# ── Facies Class Names ───────────────────────────────────────────────
FACIES_NAMES: dict[FaciesClass, str] = {
    FaciesClass.FLOODPLAIN: "Floodplain",
    FaciesClass.POINT_BAR: "Point bar",
    FaciesClass.CHANNEL: "Channel",
    FaciesClass.BOUNDARY: "Boundary",
}

# ── Standard RGB Palette [0, 1] ──────────────────────────────────────
# High-contrast palette for visualization (TensorBoard, Matplotlib)
PALETTE_RGB: list[list[float]] = [
    [0.0, 0.0, 0.0],  # 0: Floodplain (Black)
    [1.0, 0.0, 0.0],  # 1: Point bar (Red)
    [0.0, 0.0, 1.0],  # 2: Channel (Blue)
    [0.0, 1.0, 0.0],  # 3: Boundary (Green)
]

# ── One-Hot Palette [0, 1] ─────────────────────────────────────────
PALETTE_ONE_HOT_4: list[list[float]] = np.eye(4, dtype=np.float32).tolist()

# ── Normalized Palette [-1, 1] ─────────────────────────────────────
# Used for mapping normalized facies tensors and exact value matching
PALETTE_NORMALIZED: list[list[float]] = [
    [x * 2.0 - 1.0 for x in color] for color in PALETTE_RGB
]
