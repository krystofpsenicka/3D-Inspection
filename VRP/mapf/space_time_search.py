"""GPU-accelerated Space-Time A* for multi-robot collision avoidance.

Implements priority-based sequential planning (Silver, 2005) where robots
are planned one at a time in priority order, each using A* on a 4D
space-time grid to avoid both static obstacles and previously committed
trajectories via a reservation table.

The A* search uses GPU-parallel frontier expansion following Zhou & Zeng
(2015), which expands multiple near-optimal nodes from the open list
simultaneously and processes their neighbors in parallel on GPU. The
heuristic-guided threshold selection (expanding nodes with f < f_min +
delta) preserves A*'s search efficiency while enabling massive GPU
parallelism, as demonstrated by the GATSA algorithm in Li et al. (2025).

The reservation table is stored as a GPU-resident CuPy array, enabling
vectorized batch queries and trajectory commits without CPU round-trips.

References:
    Silver, D. (2005). Cooperative Pathfinding. AIIDE.
    Erdmann, M. & Lozano-Perez, T. (1987). On Multiple Moving Objects.
        Algorithmica.
    Zhou, Y. & Zeng, J. (2015). Massively Parallel A* Search on a GPU.
        AAAI.
    Li, Z. et al. (2025). GPU-accelerated Conflict-based Search for
        Multi-agent Embodied Intelligence. Machine Intelligence Research.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import cupy as cp
import numpy as np

from shared.grid_utils import downsample_occupancy_grid, OFFSETS_26, WEIGHTS_26
from ..core.constants import (
    SPACE_TIME_DT,
    SPACE_TIME_MAX_HORIZON_S,
    ST_ASTAR_MAX_EXPANSIONS,
    GPU_SEARCH_MAX_ITERATIONS,
)

logger = logging.getLogger(__name__)

# 26-connected spatial offsets + wait action (0,0,0)
_SPATIAL_OFFSETS = OFFSETS_26
_SPATIAL_WEIGHTS = WEIGHTS_26
_OFFSETS_27 = cp.array(_SPATIAL_OFFSETS + [(0, 0, 0)], dtype=cp.int32)
_WEIGHTS_27 = cp.array(
    list(_SPATIAL_WEIGHTS) + [0.0], dtype=cp.float32,
)


class ReservationTable:
    """Dense 4D (T, Nx, Ny, Nz) reservation table on GPU.

    Every lookup and commit is a CuPy array operation — no Python dict
    or set overhead.

    Following Silver (2005), the reservation table records which
    space-time cells are occupied by previously planned robots, allowing
    subsequent robots to avoid collisions by construction.

    Args:
        grid_shape: (Nx, Ny, Nz) coarse spatial dimensions.
        max_time_steps: number of discrete time slots.
        robot_half_extents_voxels: (3,) per-axis half-width of robot AABB
            in coarse voxels. Commits inflate point trajectories by this box.
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int, int],
        max_time_steps: int,
        robot_half_extents_voxels: np.ndarray,
    ):
        self.Nx, self.Ny, self.Nz = grid_shape
        self.T = max_time_steps
        self._data = cp.zeros(
            (self.T, self.Nx, self.Ny, self.Nz), dtype=cp.uint8,
        )
        self._half = np.asarray(robot_half_extents_voxels, dtype=np.intp)

    def is_reserved(self, x: int, y: int, z: int, t: int) -> bool:
        """Scalar query — pulls one element to CPU for control flow."""
        if t >= self.T or t < 0:
            return False
        return bool(self._data[t, x, y, z])

    def is_reserved_batch(
        self, positions_ijk: cp.ndarray, time_steps: cp.ndarray,
    ) -> cp.ndarray:
        """Vectorized batch query. Returns (K,) bool CuPy array."""
        t = time_steps.astype(cp.intp)
        x = positions_ijk[:, 0].astype(cp.intp)
        y = positions_ijk[:, 1].astype(cp.intp)
        z = positions_ijk[:, 2].astype(cp.intp)
        valid = (
            (t >= 0) & (t < self.T)
            & (x >= 0) & (x < self.Nx)
            & (y >= 0) & (y < self.Ny)
            & (z >= 0) & (z < self.Nz)
        )
        result = cp.zeros(len(t), dtype=cp.bool_)
        if valid.any():
            vi = cp.where(valid)[0]
            result[vi] = self._data[t[vi], x[vi], y[vi], z[vi]].astype(cp.bool_)
        return result

    def commit_trajectory(
        self,
        positions_ijk: np.ndarray | cp.ndarray,
        time_steps: np.ndarray | cp.ndarray,
    ) -> None:
        """Mark all AABB voxels around each (t, cx, cy, cz) sample."""
        hx, hy, hz = int(self._half[0]), int(self._half[1]), int(self._half[2])

        if isinstance(positions_ijk, np.ndarray):
            positions_ijk = cp.asarray(positions_ijk)
        if isinstance(time_steps, np.ndarray):
            time_steps = cp.asarray(time_steps)

        K = len(positions_ijk)
        if K == 0:
            return

        ts = time_steps.astype(cp.intp)
        pos = positions_ijk.astype(cp.intp)

        valid = ts < self.T
        if not valid.any():
            return
        ts = ts[valid]
        pos = pos[valid]
        cx, cy, cz = pos[:, 0], pos[:, 1], pos[:, 2]

        for dx in range(-hx, hx + 1):
            xi = cx + dx
            mx = (xi >= 0) & (xi < self.Nx)
            for dy in range(-hy, hy + 1):
                yi = cy + dy
                mxy = mx & (yi >= 0) & (yi < self.Ny)
                for dz in range(-hz, hz + 1):
                    zi = cz + dz
                    m = mxy & (zi >= 0) & (zi < self.Nz)
                    if not m.any():
                        continue
                    self._data[ts[m], xi[m], yi[m], zi[m]] = 1


def world_to_coarse(
    xyz: np.ndarray | cp.ndarray,
    origin: np.ndarray | cp.ndarray,
    res: float,
) -> cp.ndarray:
    """World-frame XYZ -> coarse-grid voxel index (floor)."""
    xyz_g = cp.asarray(xyz, dtype=cp.float64)
    origin_g = cp.asarray(origin, dtype=cp.float64)
    return cp.floor((xyz_g - origin_g) / res).astype(cp.intp)


def coarse_to_world(
    ijk: np.ndarray | cp.ndarray,
    origin: np.ndarray | cp.ndarray,
    res: float,
) -> cp.ndarray:
    """Coarse voxel index -> world-frame XYZ (voxel centre)."""
    ijk_g = cp.asarray(ijk, dtype=cp.float64)
    origin_g = cp.asarray(origin, dtype=cp.float64)
    return ijk_g * res + origin_g + res * 0.5


def space_time_astar_gpu(
    grid: cp.ndarray,
    start_ijk: cp.ndarray,
    goal_ijk: cp.ndarray,
    t_start: int,
    reservation: ReservationTable,
    resolution: float,
    time_step_cost: float = 0.01,
    max_time_steps: int = 0,
    max_iterations: int = GPU_SEARCH_MAX_ITERATIONS,
    f_threshold_delta: float = 2.0,
) -> Optional[Tuple[cp.ndarray, cp.ndarray]]:
    """GPU parallel A* on a 4D space-time grid.

    Uses heuristic-guided parallel frontier expansion (Zhou & Zeng, 2015):
    each iteration expands ALL frontier cells whose f-value is within
    ``f_threshold_delta`` of the current minimum, enabling massive GPU
    parallelism while preserving the A* heuristic's search efficiency.

    This approach avoids the sequential priority-queue bottleneck of CPU A*
    (which expands one node at a time) by processing thousands of
    near-optimal frontier nodes in parallel each iteration.

    Args:
        grid: (Nx, Ny, Nz) bool CuPy array (True = obstacle).
        start_ijk: (3,) start voxel indices.
        goal_ijk: (3,) goal voxel indices.
        t_start: starting time step.
        reservation: GPU-resident ReservationTable.
        resolution: coarse grid resolution in metres.
        time_step_cost: small cost per time step to prefer shorter plans.
        max_time_steps: planning horizon (0 = use reservation.T).
        max_iterations: maximum wavefront iterations.
        f_threshold_delta: controls parallelism vs optimality tradeoff.
            Smaller = more A*-like (fewer cells expanded, more iterations).
            Larger = more Dijkstra-like (more cells per iteration, fewer
            iterations). 2.0 works well for coarse grids.

    Returns:
        (path_ijk, path_t) as CuPy arrays, or None on failure.
    """
    Nx, Ny, Nz = int(grid.shape[0]), int(grid.shape[1]), int(grid.shape[2])
    T_max = max_time_steps if max_time_steps > 0 else reservation.T

    sx, sy, sz = int(start_ijk[0]), int(start_ijk[1]), int(start_ijk[2])
    gx, gy, gz = int(goal_ijk[0]), int(goal_ijk[1]), int(goal_ijk[2])

    if bool(grid[sx, sy, sz]) or bool(grid[gx, gy, gz]):
        logger.warning("[ST-A*-GPU] Start or goal in obstacle.")
        return None

    if (sx, sy, sz) == (gx, gy, gz):
        return (
            cp.array([[sx, sy, sz]], dtype=cp.intp),
            cp.array([t_start], dtype=cp.intp),
        )

    # Heuristic: Euclidean distance to goal (spatial only)
    def _h_grid():
        """Precompute heuristic for the full spatial grid."""
        xx = cp.arange(Nx, dtype=cp.float32) - gx
        yy = cp.arange(Ny, dtype=cp.float32) - gy
        zz = cp.arange(Nz, dtype=cp.float32) - gz
        return cp.sqrt(
            xx[:, None, None] ** 2 + yy[None, :, None] ** 2 + zz[None, None, :] ** 2
        ) * resolution

    h_grid = _h_grid()  # (Nx, Ny, Nz)

    # g-cost and predecessor grids
    g_cost = cp.full((T_max, Nx, Ny, Nz), cp.inf, dtype=cp.float32)
    g_cost[t_start, sx, sy, sz] = 0.0

    # Predecessor: encode as flat index for path reconstruction
    pred = cp.full((T_max, Nx, Ny, Nz), -1, dtype=cp.int64)

    # Frontier mask: cells updated in the last iteration
    frontier = cp.zeros((T_max, Nx, Ny, Nz), dtype=cp.bool_)
    frontier[t_start, sx, sy, sz] = True

    offsets_27 = _OFFSETS_27  # (27, 3) int32
    weights_27 = _WEIGHTS_27  # (27,) float32

    for iteration in range(max_iterations):
        # Check if goal reached
        if g_cost[t_start:, gx, gy, gz].min() < cp.inf:
            # Find the time step with lowest g at goal
            goal_costs = g_cost[:, gx, gy, gz]
            t_goal = int(cp.argmin(goal_costs))
            if goal_costs[t_goal] < cp.inf:
                return _reconstruct_path(pred, g_cost, t_goal, gx, gy, gz,
                                         t_start, sx, sy, sz, Nx, Ny, Nz)

        if not frontier.any():
            break

        # Select expansion set: frontier cells where f < f_min + delta
        frontier_indices = cp.argwhere(frontier)  # (F, 4) = [t, x, y, z]
        if len(frontier_indices) == 0:
            break

        fi_t = frontier_indices[:, 0]
        fi_x = frontier_indices[:, 1]
        fi_y = frontier_indices[:, 2]
        fi_z = frontier_indices[:, 3]
        f_vals = g_cost[fi_t, fi_x, fi_y, fi_z] + h_grid[fi_x, fi_y, fi_z]
        f_min = float(f_vals.min())
        expand_mask = f_vals <= f_min + f_threshold_delta
        expand_idx = frontier_indices[expand_mask]  # (E, 4)

        if len(expand_idx) == 0:
            break

        # Clear expanded cells from frontier
        frontier[expand_idx[:, 0], expand_idx[:, 1],
                 expand_idx[:, 2], expand_idx[:, 3]] = False

        E = len(expand_idx)
        et, ex, ey, ez = expand_idx[:, 0], expand_idx[:, 1], expand_idx[:, 2], expand_idx[:, 3]
        cur_g = g_cost[et, ex, ey, ez]  # (E,)

        # Next time step
        nt = et + 1  # (E,)
        time_valid = nt < T_max

        # For each of 27 neighbors, compute in parallel
        new_frontier = cp.zeros((T_max, Nx, Ny, Nz), dtype=cp.bool_)

        for n_idx in range(27):
            dx, dy, dz = int(offsets_27[n_idx, 0]), int(offsets_27[n_idx, 1]), int(offsets_27[n_idx, 2])
            w = float(weights_27[n_idx])

            nx = ex + dx
            ny = ey + dy
            nz = ez + dz

            # Bounds check
            valid = (
                time_valid
                & (nx >= 0) & (nx < Nx)
                & (ny >= 0) & (ny < Ny)
                & (nz >= 0) & (nz < Nz)
            )
            if not valid.any():
                continue

            vi = cp.where(valid)[0]
            vnx, vny, vnz, vnt = nx[vi], ny[vi], nz[vi], nt[vi]

            # Obstacle check
            free = ~grid[vnx, vny, vnz]
            if not free.any():
                continue
            fi2 = cp.where(free)[0]
            vi = vi[fi2]
            vnx, vny, vnz, vnt = vnx[fi2], vny[fi2], vnz[fi2], vnt[fi2]

            # Reservation check
            pos_check = cp.stack([vnx, vny, vnz], axis=1)
            reserved = reservation.is_reserved_batch(pos_check, vnt)
            not_reserved = ~reserved
            if not not_reserved.any():
                continue
            fi3 = cp.where(not_reserved)[0]
            vi = vi[fi3]
            vnx, vny, vnz, vnt = vnx[fi3], vny[fi3], vnz[fi3], vnt[fi3]

            # Compute tentative g
            new_g = cur_g[vi] + w * resolution + time_step_cost

            # Relaxation: update if better
            old_g = g_cost[vnt, vnx, vny, vnz]
            improved = new_g < old_g
            if not improved.any():
                continue

            imp = cp.where(improved)[0]
            g_cost[vnt[imp], vnx[imp], vny[imp], vnz[imp]] = new_g[imp]

            # Store predecessor as flat index of the source cell
            src_flat = (
                et[vi[imp]] * (Nx * Ny * Nz)
                + ex[vi[imp]] * (Ny * Nz)
                + ey[vi[imp]] * Nz
                + ez[vi[imp]]
            )
            pred[vnt[imp], vnx[imp], vny[imp], vnz[imp]] = src_flat
            new_frontier[vnt[imp], vnx[imp], vny[imp], vnz[imp]] = True

        frontier = new_frontier

    # Final goal check
    goal_costs = g_cost[:, gx, gy, gz]
    t_goal = int(cp.argmin(goal_costs))
    if goal_costs[t_goal] < cp.inf:
        return _reconstruct_path(pred, g_cost, t_goal, gx, gy, gz,
                                 t_start, sx, sy, sz, Nx, Ny, Nz)

    logger.warning("[ST-A*-GPU] No path found (start=(%d,%d,%d) goal=(%d,%d,%d) "
                   "t=%d, %d iterations).",
                   sx, sy, sz, gx, gy, gz, t_start, iteration + 1)
    return None


def _reconstruct_path(
    pred: cp.ndarray, g_cost: cp.ndarray,
    t_goal: int, gx: int, gy: int, gz: int,
    t_start: int, sx: int, sy: int, sz: int,
    Nx: int, Ny: int, Nz: int,
) -> Tuple[cp.ndarray, cp.ndarray]:
    """Backtrack through predecessor grid to reconstruct path."""
    path = [(gx, gy, gz, t_goal)]
    ct, cx, cy, cz = t_goal, gx, gy, gz
    stride_xyz = Nx * Ny * Nz
    stride_x = Ny * Nz

    max_steps = 50000
    for _ in range(max_steps):
        if ct == t_start and cx == sx and cy == sy and cz == sz:
            break
        flat = int(pred[ct, cx, cy, cz])
        if flat < 0:
            break
        pt = flat // stride_xyz
        rem = flat % stride_xyz
        px = rem // stride_x
        rem2 = rem % stride_x
        py = rem2 // Nz
        pz = rem2 % Nz
        path.append((px, py, pz, pt))
        ct, cx, cy, cz = pt, px, py, pz

    path.reverse()
    path_arr = cp.array(path, dtype=cp.intp)
    return (
        path_arr[:, :3],  # (M, 3) ijk
        path_arr[:, 3],   # (M,) time steps
    )
