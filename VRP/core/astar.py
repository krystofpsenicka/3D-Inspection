"""A* path extraction on the fine-resolution occupancy grid.

Provides a single canonical A* implementation used by
``build_route_path_cache()`` for computing collision-free path
geometry between waypoint pairs.
"""

from __future__ import annotations

import heapq
import logging

import numpy as np

from shared.grid_utils import OFFSETS_26 as _OFFSETS_26, WEIGHTS_26 as _WEIGHTS_26

logger = logging.getLogger(__name__)


def astar_path(
    grid: np.ndarray,
    start_world: np.ndarray,
    goal_world: np.ndarray,
    origin: np.ndarray,
    resolution: float,
    voxel_to_world_fn=None,
) -> np.ndarray:
    """Run A* on the occupancy grid and return the world-frame path.

    Args:
        grid: (Nx, Ny, Nz) bool array (True = obstacle).
        start_world: (3,) start position in world frame.
        goal_world: (3,) goal position in world frame.
        origin: (3,) world origin of voxel (0,0,0).
        resolution: metres per voxel.
        voxel_to_world_fn: optional callable(ijk) -> xyz for voxel centres.
            If None, uses ``ijk * resolution + origin + resolution / 2``.

    Returns:
        (M, 3) world-frame path, or (0, 3) if no path exists.
    """
    Nx, Ny, Nz = grid.shape

    start_ijk = np.floor((start_world - origin) / resolution).astype(int)
    goal_ijk = np.floor((goal_world - origin) / resolution).astype(int)

    start = tuple(start_ijk)
    goal = tuple(goal_ijk)

    if start == goal:
        centre = _voxel_centre(start_ijk, origin, resolution, voxel_to_world_fn)
        return np.array([centre])

    if grid[start] or grid[goal]:
        return np.zeros((0, 3), dtype=np.float32)

    g_cost = {start: 0.0}
    came_from: dict = {}
    open_set = [(0.0, start)]

    def h(node):
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(node, goal))) * resolution

    while open_set:
        f, current = heapq.heappop(open_set)
        if current == goal:
            path_ijk = []
            node = current
            while node in came_from:
                path_ijk.append(node)
                node = came_from[node]
            path_ijk.append(start)
            path_ijk.reverse()
            path_world = np.array(
                [_voxel_centre(np.array(p), origin, resolution, voxel_to_world_fn)
                 for p in path_ijk],
                dtype=np.float32,
            )
            return path_world
        if g_cost.get(current, np.inf) < f - h(current) - 1e-9:
            continue
        ci, cj, ck = current
        for (di, dj, dk), w in zip(_OFFSETS_26, _WEIGHTS_26):
            ni, nj, nk = ci + di, cj + dj, ck + dk
            if not (0 <= ni < Nx and 0 <= nj < Ny and 0 <= nk < Nz):
                continue
            if grid[ni, nj, nk]:
                continue
            new_g = g_cost[current] + float(w) * resolution
            nbr = (ni, nj, nk)
            if new_g < g_cost.get(nbr, np.inf):
                g_cost[nbr] = new_g
                came_from[nbr] = current
                heapq.heappush(open_set, (new_g + h(nbr), nbr))

    return np.zeros((0, 3), dtype=np.float32)


def astar_distance(
    grid: np.ndarray,
    start_ijk: np.ndarray,
    goal_ijk: np.ndarray,
    resolution: float,
) -> float:
    """A* on a 26-connected 3D voxel grid, returning path length in metres.

    Returns ``np.inf`` if no path exists.
    """
    Nx, Ny, Nz = grid.shape
    start = tuple(start_ijk)
    goal = tuple(goal_ijk)

    if start == goal:
        return 0.0
    if grid[start] or grid[goal]:
        return np.inf

    g_cost = {start: 0.0}
    open_set = [(0.0, start)]

    def h(node):
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(node, goal))) * resolution

    while open_set:
        f, current = heapq.heappop(open_set)
        if current == goal:
            return g_cost[current]
        if g_cost.get(current, np.inf) < f - h(current) - 1e-9:
            continue
        ci, cj, ck = current
        for (di, dj, dk), w in zip(_OFFSETS_26, _WEIGHTS_26):
            ni, nj, nk = ci + di, cj + dj, ck + dk
            if not (0 <= ni < Nx and 0 <= nj < Ny and 0 <= nk < Nz):
                continue
            if grid[ni, nj, nk]:
                continue
            new_g = g_cost[current] + float(w) * resolution
            nbr = (ni, nj, nk)
            if new_g < g_cost.get(nbr, np.inf):
                g_cost[nbr] = new_g
                heapq.heappush(open_set, (new_g + h(nbr), nbr))
    return np.inf


def _voxel_centre(ijk, origin, resolution, voxel_to_world_fn=None):
    if voxel_to_world_fn is not None:
        return voxel_to_world_fn(np.array(ijk))
    return np.array(ijk, dtype=np.float64) * resolution + origin + resolution / 2
