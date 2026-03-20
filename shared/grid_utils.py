"""
Shared 3-D voxel grid utilities used by both VRP and visibility packages.
"""

from __future__ import annotations

import math
from collections import deque
from typing import List, Tuple

import numpy as np

# 26-connected spatial offsets (excludes the identity (0,0,0))
OFFSETS_26: List[Tuple[int, int, int]] = [
    (di, dj, dk)
    for di in (-1, 0, 1)
    for dj in (-1, 0, 1)
    for dk in (-1, 0, 1)
    if not (di == 0 and dj == 0 and dk == 0)
]

WEIGHTS_26: np.ndarray = np.array(
    [math.sqrt(di * di + dj * dj + dk * dk) for di, dj, dk in OFFSETS_26],
    dtype=np.float64,
)


def snap_to_free(
    grid: np.ndarray,
    ijk_array: np.ndarray,
    max_radius: int = 10,
) -> np.ndarray:
    """BFS-snap each voxel index in *ijk_array* to the nearest free voxel.

    Parameters
    ----------
    grid : (Nx, Ny, Nz) bool – ``True`` = occupied.
    ijk_array : (N, 3) int – voxel indices to snap.
    max_radius : int – BFS depth limit (unused currently, kept for API compat).

    Returns
    -------
    (N, 3) int – snapped indices.
    """
    shape = np.array(grid.shape)
    result = ijk_array.copy()
    for n in range(len(ijk_array)):
        ijk = np.clip(ijk_array[n], 0, shape - 1)
        if not grid[tuple(ijk)]:
            result[n] = ijk
            continue
        queue = deque([tuple(ijk)])
        visited = {tuple(ijk)}
        found = False
        while queue:
            cur = queue.popleft()
            if not grid[cur]:
                result[n] = np.array(cur)
                found = True
                break
            ci, cj, ck = cur
            for di, dj, dk in OFFSETS_26:
                ni, nj, nk = ci + di, cj + dj, ck + dk
                if not (0 <= ni < shape[0] and 0 <= nj < shape[1] and 0 <= nk < shape[2]):
                    continue
                nb = (ni, nj, nk)
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        if not found:
            result[n] = ijk
    return result
