"""
3-D voxel grid utilities used by VRP and visibility packages.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

# 26-connected spatial offsets (without (0,0,0))
OFFSETS_26: List[Tuple[int, int, int]] = [
    (di, dj, dk)
    for di in (-1, 0, 1)
    for dj in (-1, 0, 1)
    for dk in (-1, 0, 1)
    if not (di == 0 and dj == 0 and dk == 0)
]

# Corresponding Euclidean distances (weights) for each offset
WEIGHTS_26: np.ndarray = np.array(
    [math.sqrt(di * di + dj * dj + dk * dk) for di, dj, dk in OFFSETS_26],
    dtype=np.float64,
)

