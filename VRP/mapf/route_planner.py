"""Single-robot route planning via Space-Time A*.

Walks one robot through its VRP route leg-by-leg via the search in
``space_time_search.py``, smooths each leg with OMPL, then commits the
trajectory to the reservation table so later robots avoid it.
"""

from __future__ import annotations

import logging
import math

import cupy as cp

from shared.occupancy_grid import OccupancyGrid

from ..core.constants import (
    AUV_CRUISE_SPEED,
    GPU_SEARCH_MAX_ITERATIONS,
    OMPL_SIMPLIFY_MAX_TIME,
    ROBOT_RADIUS,
    SPACE_TIME_DT,
    SPACE_TIME_DWELL_S,
    SPACE_TIME_MIN_LEG_STEPS,
    SPACE_TIME_SAFETY_FACTOR,
)
from ..core.types import PlanningStats
from .path_smoother import arc_length_resample, simplify_path_ompl
from .reservation_table import ReservationTable
from .space_time_search import space_time_astar_gpu

logger = logging.getLogger(__name__)


def plan_robot_route_st(
    coarse_og: OccupancyGrid,
    reservation: ReservationTable,
    route: list[int],
    waypoint_positions: cp.ndarray,
    dwell_s: float = SPACE_TIME_DWELL_S,
    dt: float = SPACE_TIME_DT,
    fine_occupancy_grid: OccupancyGrid = None,
    robot_radius: float = ROBOT_RADIUS,
    cruise_speed: float = AUV_CRUISE_SPEED,
) -> tuple[cp.ndarray, cp.ndarray, list, PlanningStats]:
    """Plan one robot through its full VRP route using GPU Space-Time A*.

    Each leg is planned with a single A* call using a local time budget
    proportional to the leg distance.

    Returns:
        (world_xyz, coarse_time_steps, wp_schedule, stats) where
        world_xyz is (M, 3) CuPy, coarse_time_steps is (M,) CuPy,
        wp_schedule is list of (t_dwell_start, t_dwell_end, node_idx),
        stats is PlanningStats.
    """
    xyz_all = waypoint_positions

    coarse_positions: list = []
    world_positions: list = []
    coarse_times: list = []
    wp_schedule: list = []
    t_cursor = 0
    stats = PlanningStats()
    straight_line_total = 0.0

    hold_steps = max(1, int(round(dwell_s / dt)))
    grid_shape = coarse_og.shape

    for leg in range(1, len(route)):
        prev_node = route[leg - 1]
        curr_node = route[leg]

        # Convert waypoints to coarse voxel indices
        s_xyz = xyz_all[prev_node]
        g_xyz = xyz_all[curr_node]
        s_ijk = coarse_og.world_to_voxel(s_xyz.reshape(1, 3))[0]
        g_ijk = coarse_og.world_to_voxel(g_xyz.reshape(1, 3))[0]

        # Clip to grid bounds
        for d in range(3):
            s_ijk[d] = cp.clip(s_ijk[d], 0, grid_shape[d] - 1)
            g_ijk[d] = cp.clip(g_ijk[d], 0, grid_shape[d] - 1)

        dist = float(cp.linalg.norm(g_xyz - s_xyz))
        straight_line_total += dist

        if cp.array_equal(s_ijk, g_ijk):
            # Same coarse voxel  --  no A* needed, but still navigate to g_xyz
            # and record orientation.

            if not coarse_positions:
                # Bootstrap: seed arrays with the start position so subsequent
                # appends work correctly.
                coarse_positions.append(s_ijk.reshape(1, 3))
                world_positions.append(s_xyz.reshape(1, 3))
                coarse_times.append(cp.array([t_cursor], dtype=cp.intp))

            last_world = world_positions[-1][-1:]  # (1, 3) world
            g_world = g_xyz.reshape(1, 3)

            # Short linear interpolation from last_world -> g_xyz
            sub_dist = float(cp.linalg.norm(g_world - last_world))
            n_interp = max(1, int(math.ceil(sub_dist / (cruise_speed * dt))))
            if n_interp > 1:
                alphas = cp.linspace(0.0, 1.0, n_interp, dtype=cp.float64)[:, None]
                interp_world = last_world + alphas * (g_world - last_world)
                interp_ijk = cp.tile(g_ijk.reshape(1, 3), (n_interp, 1))
                interp_t = cp.arange(
                    t_cursor + 1,
                    t_cursor + n_interp + 1,
                    dtype=cp.intp,
                )
                coarse_positions.append(interp_ijk)
                world_positions.append(interp_world)
                coarse_times.append(interp_t)
                t_cursor += n_interp

            # Dwell at g_xyz
            dwell_ijk = cp.tile(g_ijk.reshape(1, 3), (hold_steps, 1))
            dwell_world = cp.tile(g_world, (hold_steps, 1))
            dwell_t = cp.arange(
                t_cursor + 1,
                t_cursor + hold_steps + 1,
                dtype=cp.intp,
            )
            coarse_positions.append(dwell_ijk)
            world_positions.append(dwell_world)
            coarse_times.append(dwell_t)
            t_dwell_start = int(t_cursor) + 1
            t_cursor += hold_steps
            wp_schedule.append((t_dwell_start, int(t_cursor), curr_node))
            continue

        # Local time budget for this leg
        t_leg = max(
            SPACE_TIME_MIN_LEG_STEPS,
            int(math.ceil(dist / cruise_speed / dt)) * SPACE_TIME_SAFETY_FACTOR,
        )

        result = space_time_astar_gpu(
            coarse_og,
            s_ijk,
            g_ijk,
            t_cursor,
            reservation,
            max_time_steps=t_leg,
        )

        if result is None:
            stats.astar_retries += 1
            result = space_time_astar_gpu(
                coarse_og,
                s_ijk,
                g_ijk,
                t_cursor,
                reservation,
                max_time_steps=t_leg * 2,
                max_iterations=2 * GPU_SEARCH_MAX_ITERATIONS,
            )

        if result is None:
            stats.astar_failures += 1
            logger.warning(
                "[route_planner] ST-A* failed leg %d->%d (t=%d).",
                prev_node,
                curr_node,
                t_cursor,
            )
            # Dwell at last position if possible
            if coarse_positions:
                last = coarse_positions[-1][-1:]
                last_world = world_positions[-1][-1:]
                dwell_ijk = cp.tile(last, (hold_steps, 1))
                dwell_world = cp.tile(last_world, (hold_steps, 1))
                dwell_t = cp.arange(
                    t_cursor + 1,
                    t_cursor + hold_steps + 1,
                    dtype=cp.intp,
                )
                coarse_positions.append(dwell_ijk)
                world_positions.append(dwell_world)
                coarse_times.append(dwell_t)
                t_dwell_start = int(t_cursor) + 1
                t_cursor += hold_steps
                wp_schedule.append((t_dwell_start, int(t_cursor), curr_node))
            continue

        seg_ijk, seg_t = result
        start_idx = 0 if not coarse_positions else 1
        if len(seg_ijk) <= start_idx:
            continue
        planned_ijk = seg_ijk[start_idx:]
        planned_t = seg_t[start_idx:]
        t_cursor = int(seg_t[-1])

        final_ijk = planned_ijk
        final_world = coarse_og.voxel_to_world(planned_ijk)

        if fine_occupancy_grid is not None and len(planned_ijk) >= 3:
            smoothed_world = simplify_path_ompl(
                final_world,
                fine_occupancy_grid,
                robot_radius,
                max_time=OMPL_SIMPLIFY_MAX_TIME,
                reservation=reservation,
                time_steps=planned_t,
                coarse_og=coarse_og,
            )

            if len(smoothed_world) >= 2:
                resampled_world = arc_length_resample(smoothed_world, len(planned_t))
                resampled_ijk = coarse_og.world_to_voxel(resampled_world)
                for d in range(3):
                    resampled_ijk[:, d] = cp.clip(resampled_ijk[:, d], 0, grid_shape[d] - 1)

                conflict = reservation.is_reserved_batch(resampled_ijk, planned_t).any()

                if not bool(conflict):
                    final_ijk = resampled_ijk
                    final_world = resampled_world
                    logger.info(
                        "[route_planner] leg %d->%d OMPL smoothed: %d->%d wpts",
                        prev_node,
                        curr_node,
                        len(planned_ijk),
                        len(smoothed_world),
                    )
                else:
                    logger.info(
                        "[route_planner] leg %d->%d OMPL path has reservation "
                        "conflict, using A* path.",
                        prev_node,
                        curr_node,
                    )

        coarse_positions.append(final_ijk)
        world_positions.append(final_world)
        coarse_times.append(planned_t)

        last = final_ijk[-1:]
        dwell_ijk = cp.tile(last, (hold_steps, 1))
        dwell_world = cp.tile(final_world[-1:], (hold_steps, 1))
        dwell_t = cp.arange(t_cursor + 1, t_cursor + hold_steps + 1, dtype=cp.intp)
        coarse_positions.append(dwell_ijk)
        world_positions.append(dwell_world)
        coarse_times.append(dwell_t)
        t_dwell_start = int(t_cursor) + 1
        t_cursor += hold_steps
        wp_schedule.append((t_dwell_start, int(t_cursor), curr_node))

    if not coarse_positions:
        return (cp.empty((0, 3), dtype=cp.float64), cp.empty((0,), dtype=cp.intp), [], stats)

    all_ijk = cp.concatenate(coarse_positions, axis=0)
    all_t = cp.concatenate(coarse_times, axis=0)

    reservation.commit_trajectory(all_ijk, all_t)

    # Reserve final position for all remaining time steps
    last_t = int(all_t[-1])
    if last_t + 1 < reservation.T:
        tail_steps = cp.arange(last_t + 1, reservation.T, dtype=cp.intp)
        tail_ijk = cp.tile(all_ijk[-1:], (len(tail_steps), 1))
        reservation.commit_trajectory(tail_ijk, tail_steps)

    world_xyz = cp.concatenate(world_positions, axis=0)

    if len(world_xyz) >= 2:
        actual_dist = float(cp.sum(cp.linalg.norm(cp.diff(world_xyz, axis=0), axis=1)))
        stats.detour_ratio = (
            actual_dist / straight_line_total if straight_line_total > 1e-9 else 1.0
        )
        diffs_norm = cp.linalg.norm(cp.diff(world_xyz, axis=0), axis=1)
        stats.wait_steps = int(cp.sum(diffs_norm < 1e-6))

    return world_xyz, all_t, wp_schedule, stats
