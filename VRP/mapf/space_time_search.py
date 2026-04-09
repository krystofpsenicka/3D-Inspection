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
from typing import Optional, Tuple

import cupy as cp

from ..core.constants import (
    OFFSETS_27,
    WEIGHTS_27,
    GPU_SEARCH_MAX_ITERATIONS,
)
from shared.occupancy_grid import OccupancyGrid

logger = logging.getLogger(__name__)

from .reservation_table import ReservationTable  # noqa: E402


def space_time_astar_gpu(
    coarse_og: OccupancyGrid,
    start_ijk: cp.ndarray,
    goal_ijk: cp.ndarray,
    t_offset: int,
    reservation: ReservationTable,
    time_step_cost: float = 0.01,
    max_time_steps: int = 0,
    max_iterations: int = GPU_SEARCH_MAX_ITERATIONS,
    f_threshold_delta: float = 2.0,
) -> Optional[Tuple[cp.ndarray, cp.ndarray]]:
    """GPU parallel A* on a 4D space-time grid with local time allocation.

    Uses heuristic-guided parallel frontier expansion (Zhou & Zeng, 2015):
    each iteration expands ALL frontier cells whose f-value is within
    ``f_threshold_delta`` of the current minimum, enabling GPU
    parallelism while preserving the A* heuristic's search efficiency.

    All 27 neighbors (26-connected + wait) are expanded in a single
    vectorized broadcast operation.

    The search allocates ``g_cost`` only for ``max_time_steps`` local
    time steps (not the full reservation horizon), starting at internal
    index 0.  Reservation table queries use ``t_offset + local_t`` to
    check absolute time.

    Args:
        coarse_og: OccupancyGrid for the coarse planning grid.
        start_ijk: (3,) start voxel indices.
        goal_ijk: (3,) goal voxel indices.
        t_offset: absolute time offset for reservation table queries.
        reservation: ReservationTable.
        time_step_cost: cost per time step.
        max_time_steps: local planning horizon. 0 = reservation.T - t_offset.
        max_iterations: maximum wavefront iterations.
        f_threshold_delta: controls parallelism vs optimality tradeoff.

    Returns:
        (path_ijk, path_t) as CuPy arrays with absolute time steps,
        or None on failure.
    """
    grid = coarse_og.grid
    resolution = coarse_og.resolution
    Nx, Ny, Nz = coarse_og.shape

    T_local = max_time_steps if max_time_steps > 0 else max(1, reservation.T - t_offset)

    sx, sy, sz = int(start_ijk[0]), int(start_ijk[1]), int(start_ijk[2])
    gx, gy, gz = int(goal_ijk[0]), int(goal_ijk[1]), int(goal_ijk[2])

    if bool(grid[sx, sy, sz]) or bool(grid[gx, gy, gz]):
        logger.warning("[ST-A*-GPU] Start or goal in obstacle.")
        return None

    if (sx, sy, sz) == (gx, gy, gz):
        return (
            cp.array([[sx, sy, sz]], dtype=cp.intp),
            cp.array([t_offset], dtype=cp.intp),
        )

    # Heuristic: spatial Euclidean + temporal Chebyshev (admissible)
    xx = cp.arange(Nx, dtype=cp.float32) - gx
    yy = cp.arange(Ny, dtype=cp.float32) - gy
    zz = cp.arange(Nz, dtype=cp.float32) - gz
    spatial_h = cp.sqrt(
        xx[:, None, None] ** 2 + yy[None, :, None] ** 2 + zz[None, None, :] ** 2
    ) * resolution
    chebyshev_h = cp.maximum(
        cp.abs(xx[:, None, None]),
        cp.maximum(cp.abs(yy[None, :, None]), cp.abs(zz[None, None, :]))
    )
    h_grid = spatial_h + chebyshev_h * time_step_cost  # (Nx, Ny, Nz)

    # g-cost and predecessor grids (local time: 0..T_local)
    g_cost = cp.full((T_local, Nx, Ny, Nz), cp.inf, dtype=cp.float32)
    g_cost[0, sx, sy, sz] = 0.0

    pred = cp.full((T_local, Nx, Ny, Nz), -1, dtype=cp.int64)

    # Index-based frontier: (F, 4) array of [t_local, x, y, z]
    frontier = cp.array([[0, sx, sy, sz]], dtype=cp.int32)

    for iteration in range(max_iterations):
        # Check if goal reached
        goal_costs = g_cost[:, gx, gy, gz]
        best_t = int(cp.argmin(goal_costs))
        if goal_costs[best_t] < cp.inf:
            return _reconstruct_path(
                pred, t_offset, best_t, gx, gy, gz, sx, sy, sz, Nx, Ny, Nz,
            )

        if len(frontier) == 0:
            break

        # Select expansion set: frontier cells where f < f_min + delta
        fi_t = frontier[:, 0]
        fi_x = frontier[:, 1]
        fi_y = frontier[:, 2]
        fi_z = frontier[:, 3]
        f_vals = g_cost[fi_t, fi_x, fi_y, fi_z] + h_grid[fi_x, fi_y, fi_z]
        f_min = float(f_vals.min())
        expand_mask = f_vals <= f_min + f_threshold_delta

        if not expand_mask.any():
            break

        expand_idx = frontier[expand_mask]   # (E, 4)
        remaining = frontier[~expand_mask]   # cells not expanded yet

        E = len(expand_idx)
        et = expand_idx[:, 0]
        ex = expand_idx[:, 1]
        ey = expand_idx[:, 2]
        ez = expand_idx[:, 3]
        cur_g = g_cost[et, ex, ey, ez]  # (E,)

        # All 27 neighbors of all E cells in one vectorized operation
        nt = (et + 1)[:, None].astype(cp.int32)       # (E, 1)
        spatial = expand_idx[:, 1:4]                    # (E, 3)
        nbr_pos = spatial[:, None, :] + OFFSETS_27[None, :, :]  # (E, 27, 3)

        # Flatten to (E*27,)
        flat_pos = nbr_pos.reshape(-1, 3)               # (E*27, 3)
        flat_t = cp.broadcast_to(nt, (E, 27)).reshape(-1)  # (E*27,)
        flat_w = cp.broadcast_to(WEIGHTS_27[None, :], (E, 27)).reshape(-1)
        flat_parent = cp.broadcast_to(
            cp.arange(E, dtype=cp.int32)[:, None], (E, 27),
        ).reshape(-1)

        nx, ny, nz = flat_pos[:, 0], flat_pos[:, 1], flat_pos[:, 2]

        # Bounds + time check
        valid = (
            (flat_t >= 0) & (flat_t < T_local)
            & (nx >= 0) & (nx < Nx)
            & (ny >= 0) & (ny < Ny)
            & (nz >= 0) & (nz < Nz)
        )
        if not valid.any():
            frontier = remaining
            continue

        vi = cp.where(valid)[0]
        nx, ny, nz = nx[vi], ny[vi], nz[vi]
        vt = flat_t[vi]
        vw = flat_w[vi]
        vp = flat_parent[vi]

        # Obstacle check
        free = ~grid[nx, ny, nz]
        if not free.any():
            frontier = remaining
            continue
        fi2 = cp.where(free)[0]
        nx, ny, nz, vt, vw, vp = nx[fi2], ny[fi2], nz[fi2], vt[fi2], vw[fi2], vp[fi2]

        # Reservation check (absolute time)
        abs_t = vt + t_offset
        pos_check = cp.stack([nx, ny, nz], axis=1)
        not_reserved = ~reservation.is_reserved_batch(pos_check, abs_t)
        if not not_reserved.any():
            frontier = remaining
            continue
        fi3 = cp.where(not_reserved)[0]
        nx, ny, nz, vt, vw, vp = nx[fi3], ny[fi3], nz[fi3], vt[fi3], vw[fi3], vp[fi3]

        # Compute tentative g
        new_g = cur_g[vp] + vw * resolution + time_step_cost

        # Relaxation: keep only candidates that improve on current g-cost
        old_g = g_cost[vt, nx, ny, nz]
        improved = new_g < old_g
        if not improved.any():
            frontier = remaining
            continue

        imp = cp.where(improved)[0]
        vt_imp, nx_imp, ny_imp, nz_imp = vt[imp], nx[imp], ny[imp], nz[imp]
        new_g_imp = new_g[imp]
        vp_imp = vp[imp]

        # Resolve duplicates: multiple parents may improve the same target
        # cell. Sort by (target_key, g) so the best-g candidate comes first
        # per target, then take only the first occurrence of each key.
        flat_keys = (
            vt_imp.astype(cp.int64) * (Nx * Ny * Nz)
            + nx_imp.astype(cp.int64) * (Ny * Nz)
            + ny_imp.astype(cp.int64) * Nz
            + nz_imp.astype(cp.int64)
        )
        sort_order = cp.lexsort(cp.stack([new_g_imp, flat_keys]))
        sorted_keys = flat_keys[sort_order]
        first_mask = cp.ones(len(sorted_keys), dtype=cp.bool_)
        first_mask[1:] = sorted_keys[1:] != sorted_keys[:-1]
        winners = sort_order[first_mask]

        # Write g-cost and predecessor only for the best candidate per cell
        g_cost[vt_imp[winners], nx_imp[winners], ny_imp[winners], nz_imp[winners]] = new_g_imp[winners]

        src_flat = (
            et[vp_imp[winners]] * (Nx * Ny * Nz)
            + ex[vp_imp[winners]] * (Ny * Nz)
            + ey[vp_imp[winners]] * Nz
            + ez[vp_imp[winners]]
        )
        pred[vt_imp[winners], nx_imp[winners], ny_imp[winners], nz_imp[winners]] = src_flat

        # Winners are already deduplicated — use directly as new frontier
        new_cells = cp.stack([vt_imp[winners], nx_imp[winners], ny_imp[winners], nz_imp[winners]], axis=1)
        frontier = cp.concatenate([remaining, new_cells], axis=0) if len(remaining) > 0 else new_cells

    # Final goal check
    goal_costs = g_cost[:, gx, gy, gz]
    best_t = int(cp.argmin(goal_costs))
    if goal_costs[best_t] < cp.inf:
        return _reconstruct_path(
            pred, t_offset, best_t, gx, gy, gz, sx, sy, sz, Nx, Ny, Nz,
        )

    logger.warning("[ST-A*-GPU] No path found (start=(%d,%d,%d) goal=(%d,%d,%d) "
                   "t_offset=%d, %d iterations).",
                   sx, sy, sz, gx, gy, gz, t_offset, iteration + 1)
    return None


def _reconstruct_path(
    pred: cp.ndarray,
    t_offset: int,
    t_goal_local: int, gx: int, gy: int, gz: int,
    sx: int, sy: int, sz: int,
    Nx: int, Ny: int, Nz: int,
) -> Tuple[cp.ndarray, cp.ndarray]:
    """Backtrack through predecessor grid to reconstruct path.

    Returns absolute time steps (local + t_offset).
    """
    path = [(gx, gy, gz, t_goal_local)]
    ct, cx, cy, cz = t_goal_local, gx, gy, gz
    stride_xyz = Nx * Ny * Nz
    stride_x = Ny * Nz

    for _ in range(GPU_SEARCH_MAX_ITERATIONS):
        if ct == 0 and cx == sx and cy == sy and cz == sz:
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
        path_arr[:, :3],              # (M, 3) ijk
        path_arr[:, 3] + t_offset,    # (M,) absolute time steps
    )
