"""Collision-free distance matrix computation via cuGraph.

Computes the N x N pairwise shortest-path distance matrix between
waypoints on the inflated occupancy grid using NVIDIA cuGraph's
all-pairs Dijkstra algorithm running on GPU.

References:
    Davidson, A., Baxter, S., Garland, M. & Owens, J.D. (2014).
        Work-Efficient Parallel GPU Methods for Single-Source Shortest
        Paths. IPDPS.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np

from .astar import astar_path

logger = logging.getLogger(__name__)


def compute_distance_matrix(
    occupancy_grid,
    waypoints_world: np.ndarray,
    cache_path: str | None = None,
    force_rebuild: bool = False,
    rapids_python: str | None = None,
) -> np.ndarray:
    """Compute the N x N collision-free distance matrix via cuGraph.

    Args:
        occupancy_grid: OccupancyGrid instance.
        waypoints_world: (N, 3) world-frame waypoint positions.
        cache_path: optional path to cache the result as .npy.
        force_rebuild: ignore cache.
        rapids_python: path to the Python binary in the rapids_solver env.

    Returns:
        (N, N) float32 distance matrix in metres.

    Raises:
        RuntimeError: if the cuGraph subprocess fails.
    """
    if rapids_python is None:
        from .constants import RAPIDS_PYTHON
        rapids_python = RAPIDS_PYTHON

    if cache_path and not force_rebuild and os.path.exists(cache_path):
        logger.info("[DistMatrix] Loading cached matrix from %s", cache_path)
        return np.load(cache_path)

    N = len(waypoints_world)
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

    matrix = compute_via_subprocess(occupancy_grid, waypoints_world, rapids_python)

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        np.save(cache_path, matrix)
        logger.info("[DistMatrix] Saved to %s", cache_path)

    return matrix


def build_route_path_cache(
    occupancy_grid,
    waypoints_world: np.ndarray,
    routes: List[List[int]],
    sub_sample_dist: float = 2.0,
) -> dict:
    """Compute A* paths for every (i, j) segment pair used by the routes.

    Only the unique directed pairs actually traversed are computed.
    Each path is sub-sampled to one point every ``sub_sample_dist`` metres.

    Returns:
        dict[(int, int), np.ndarray of shape (M, 3)]
    """
    pairs: set = set()
    for route in routes:
        for k in range(1, len(route)):
            pairs.add((route[k - 1], route[k]))

    xyz = np.asarray(waypoints_world)[:, :3]
    cache: dict = {}

    for (i, j) in sorted(pairs):
        path = astar_path(
            occupancy_grid.grid, xyz[i], xyz[j],
            occupancy_grid.origin, occupancy_grid.resolution,
            voxel_to_world_fn=occupancy_grid.voxel_to_world,
        )

        if len(path) == 0:
            logger.warning("[path_cache] No A* path %s->%s, using direct.", i, j)
            cache[(i, j)] = np.array([xyz[i], xyz[j]], dtype=np.float32)
            continue

        if len(path) <= 2:
            cache[(i, j)] = path
            continue

        seg_lens = np.linalg.norm(np.diff(path, axis=0), axis=1).astype(float)
        cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total = cum[-1]

        if total < sub_sample_dist:
            cache[(i, j)] = path[[0, -1]]
            continue

        targets = np.arange(0.0, total, sub_sample_dist)
        picked = np.unique(
            np.concatenate([[0], np.searchsorted(cum, targets), [len(path) - 1]])
        ).astype(int)
        cache[(i, j)] = path[picked]
        logger.info("[path_cache] %s->%s: %s A* pts -> %s sub-pts (%.1fm)",
                    i, j, len(path), len(cache[(i, j)]), total)

    logger.info("[path_cache] Built paths for %s route segments.", len(cache))
    return cache
