"""
3-D voxel grid utilities used by VRP and visibility packages.
"""

from __future__ import annotations

import logging
import math
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

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


# ── Grid inflation ────────────────────────────────────────────────────────────

def inflate_grid(grid: np.ndarray, inflation_voxels: int) -> np.ndarray:
    """Morphological dilation of the obstacle grid by *inflation_voxels* voxels.

    Uses a spherical structuring element of radius ``inflation_voxels``.
    This expands every obstacle by the robot's collision radius so that
    path planners can treat the robot as a point.

    Using CuPy (GPU) for speed.
    """
    r = inflation_voxels
    coords = np.mgrid[-r:r+1, -r:r+1, -r:r+1]
    se = (coords[0]**2 + coords[1]**2 + coords[2]**2) <= r**2

    import cupy as cp
    from cupyx.scipy.ndimage import binary_dilation as gpu_dilation
    grid_gpu = cp.asarray(grid)
    se_gpu = cp.asarray(se)
    result = gpu_dilation(grid_gpu, structure=se_gpu)
    return cp.asnumpy(result)


# ── Grid down-sampling ───────────────────────────────────────────────────────

def downsample_occupancy_grid(
    fine_grid: np.ndarray,
    fine_origin: np.ndarray,
    fine_res: float,
    coarse_res: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Down-sample an occupancy grid.

    A coarse voxel is **occupied** if **any** of its constituent fine
    voxels is occupied (no false free-space).

    Vectorised with ``np.pad`` + ``reshape`` + ``any``.

    Returns ``(coarse_grid, coarse_origin, coarse_res)``.
    """
    factor = max(1, int(round(coarse_res / fine_res)))
    Fx, Fy, Fz = fine_grid.shape

    # Pad each axis to the next multiple of *factor* so reshape is exact.
    pad_x = (-Fx) % factor
    pad_y = (-Fy) % factor
    pad_z = (-Fz) % factor
    if pad_x or pad_y or pad_z:
        fine_padded = np.pad(
            fine_grid.astype(bool),
            [(0, pad_x), (0, pad_y), (0, pad_z)],
            constant_values=False,
        )
    else:
        fine_padded = fine_grid.astype(bool)

    Cx = fine_padded.shape[0] // factor
    Cy = fine_padded.shape[1] // factor
    Cz = fine_padded.shape[2] // factor

    coarse = (
        fine_padded
        .reshape(Cx, factor, Cy, factor, Cz, factor)
        .any(axis=(1, 3, 5))
    )

    coarse_origin = fine_origin.copy()
    actual_res = fine_res * factor
    logger.info(
        "[downsample] Grid: %s -> %s  (factor=%d, coarse_res=%.2fm)",
        fine_grid.shape, coarse.shape, factor, actual_res,
    )
    return coarse, coarse_origin, actual_res

