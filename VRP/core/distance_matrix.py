"""Collision-free distance matrix computation via cuGraph.

Computes the N x N pairwise shortest-path distance matrix between
waypoints on the inflated occupancy grid.

References:
    Davidson, A., Baxter, S., Garland, M. & Owens, J.D. (2014).
        Work-Efficient Parallel GPU Methods for Single-Source Shortest
        Paths. IPDPS.
"""

from __future__ import annotations

import logging
import os

import cupy as cp
import numpy as np

logger = logging.getLogger(__name__)


def compute_distance_matrix(
    occupancy_grid,
    waypoints_xyz: cp.ndarray,
    cache_path: str | None = None,
    force_rebuild: bool = False,
    rapids_python: str | None = None,
) -> cp.ndarray:
    """Compute the N x N collision-free distance matrix via cuGraph.

    Args:
        occupancy_grid: OccupancyGrid instance (GPU-resident).
        waypoints_xyz: (N, 3) CuPy world-frame waypoint positions.
        cache_path: optional path to cache the result as .npy.
        force_rebuild: ignore cache.
        rapids_python: path to the Python binary in the rapids_solver env.

    Returns:
        (N, N) float32 CuPy distance matrix in metres.

    Raises:
        RuntimeError: if the cuGraph subprocess fails.
    """
    if rapids_python is None:
        from .constants import RAPIDS_PYTHON
        rapids_python = RAPIDS_PYTHON

    if cache_path and not force_rebuild and os.path.exists(cache_path):
        logger.info("[DistMatrix] Loading cached matrix from %s", cache_path)
        return cp.asarray(np.load(cache_path))

    # Convert to numpy for subprocess
    waypoints_np = cp.asnumpy(waypoints_xyz).astype(np.float64)

    N = len(waypoints_np)
    logger.info("[DistMatrix] Computing %sx%s distance matrix via cuGraph...", N, N)

    from .cugraph_subprocess import rapids_env_has_cugraph, compute_via_subprocess

    if not os.path.exists(rapids_python):
        raise RuntimeError(
            f"[DistMatrix] RAPIDS Python not found at {rapids_python}. "
            "Set RAPIDS_PYTHON env var or install rapids_solver conda env."
        )

    if not rapids_env_has_cugraph(rapids_python):
        raise RuntimeError(
            "[DistMatrix] cuGraph not available in the RAPIDS environment. "
            "Install cuGraph in the rapids_solver conda env."
        )

    matrix_np = compute_via_subprocess(occupancy_grid, waypoints_np, rapids_python)
    matrix = cp.asarray(matrix_np)

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        np.save(cache_path, matrix_np)
        logger.info("[DistMatrix] Saved to %s", cache_path)

    return matrix
