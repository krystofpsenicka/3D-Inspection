"""Single-robot route planning via Space-Time A* on GPU.

Plans one robot through its full VRP route using GPU-accelerated
Space-Time A* (see ``space_time_search.py``). For each leg the robot
navigates directly between waypoints, with OMPL smoothing applied
to the resulting coarse path.

After all legs, the trajectory is committed to the reservation table
so later robots (in priority order) avoid it.

References:
    Silver, D. (2005). Cooperative Pathfinding. AIIDE.
    Zhou, Y. & Zeng, J. (2015). Massively Parallel A* Search on a GPU. AAAI.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import cupy as cp
import numpy as np

from ..core.types import PlanningStats
from ..core.constants import (
    CAMERA_OFFSET_FORWARD,
    CAMERA_OFFSET_UP,
    OMPL_SIMPLIFY_MAX_TIME,
    ROBOT_RADIUS,
    SPACE_TIME_DT,
    SPACE_TIME_DWELL_S,
    SPACE_TIME_HOP_DISTANCE,
    ST_ASTAR_MAX_EXPANSIONS,
    GPU_SEARCH_MAX_ITERATIONS,
)
from .space_time_search import (
    ReservationTable,
    coarse_to_world,
    space_time_astar_gpu,
    world_to_coarse,
)
from .path_smoother import arc_length_resample, simplify_path_ompl

logger = logging.getLogger(__name__)


def _robot_xyz_from_waypoints(
    positions: cp.ndarray, rotmats: cp.ndarray, home_indices: set,
) -> cp.ndarray:
    """Return robot body-centre XYZ for all waypoints (vectorized GPU).

    Inspection waypoints encode the desired camera position and viewing
    direction. The robot body centre is offset backward so the camera
    arrives at the waypoint position.

    Home nodes are returned unchanged (no camera offset).

    Args:
        positions:    (N, 3) CuPy — waypoint positions.
        rotmats:      (N, 3, 3) CuPy — rotation matrices.
        home_indices: set of int — indices of home/depot nodes.

    Returns:
        (N, 3) CuPy — robot body-centre positions.
    """
    xyz = positions.copy()

    # Forward direction = column 0 of rotation matrix
    forwards = rotmats[:, :, 0]  # (N, 3)

    # Camera offset: subtract forward and up offsets
    xy_norm = cp.linalg.norm(forwards[:, :2], axis=1, keepdims=True)
    xy_norm = cp.maximum(xy_norm, 1e-9)
    xyz[:, 0] -= CAMERA_OFFSET_FORWARD * (forwards[:, 0] / xy_norm[:, 0])
    xyz[:, 1] -= CAMERA_OFFSET_FORWARD * (forwards[:, 1] / xy_norm[:, 0])
    xyz[:, 2] -= CAMERA_OFFSET_UP

    # Restore home positions (no camera offset)
    for idx in home_indices:
        xyz[idx] = positions[idx]

    return xyz


def plan_robot_route_st(
    coarse_grid: cp.ndarray,
    coarse_origin: cp.ndarray,
    coarse_res: float,
    reservation: ReservationTable,
    route: List[int],
    waypoint_positions: cp.ndarray,
    waypoint_rotmats: cp.ndarray,
    home_indices: set,
    dwell_s: float = SPACE_TIME_DWELL_S,
    dt: float = SPACE_TIME_DT,
    fine_occupancy_grid=None,
    robot_radius: float = ROBOT_RADIUS,
) -> Tuple[cp.ndarray, cp.ndarray, list, PlanningStats]:
    """Plan one robot through its full VRP route using GPU Space-Time A*.

    Returns:
        (world_xyz, coarse_time_steps, wp_schedule, stats) where
        world_xyz is (M, 3) CuPy, coarse_time_steps is (M,) CuPy,
        wp_schedule is list of (t_dwell_start, t_dwell_end, node_idx),
        stats is PlanningStats.
    """
    xyz_all = _robot_xyz_from_waypoints(waypoint_positions, waypoint_rotmats, home_indices)

    coarse_positions: list = []
    world_positions: list = []
    coarse_times: list = []
    wp_schedule: list = []
    t_cursor = 0
    stats = PlanningStats()
    straight_line_total = 0.0

    hold_steps = max(1, int(round(dwell_s / dt)))

    for leg in range(1, len(route)):
        prev_node = route[leg - 1]
        curr_node = route[leg]

        # Waypoint-to-waypoint
        leg_xyz = cp.stack([xyz_all[prev_node], xyz_all[curr_node]])

        leg_ijk = world_to_coarse(leg_xyz, coarse_origin, coarse_res)

        # Deduplicate consecutive identical voxels
        keep = cp.ones(len(leg_ijk), dtype=cp.bool_)
        keep[1:] = cp.any(leg_ijk[1:] != leg_ijk[:-1], axis=1)
        leg_ijk = leg_ijk[keep]

        grid_shape = (int(coarse_grid.shape[0]), int(coarse_grid.shape[1]), int(coarse_grid.shape[2]))
        for d, mx in enumerate(grid_shape):
            leg_ijk[:, d] = cp.clip(leg_ijk[:, d], 0, mx - 1)

        stride_voxels = max(1, int(round(SPACE_TIME_HOP_DISTANCE / coarse_res)))
        sub_idx = list(range(0, len(leg_ijk), stride_voxels))
        if sub_idx[-1] != len(leg_ijk) - 1:
            sub_idx.append(len(leg_ijk) - 1)
        sub_ijk = leg_ijk[sub_idx]

        straight_line_total += float(cp.linalg.norm(
            xyz_all[curr_node] - xyz_all[prev_node]))

        leg_plan_pos: list = []
        leg_plan_t: list = []

        for hop in range(1, len(sub_ijk)):
            s_ijk = sub_ijk[hop - 1]
            g_ijk = sub_ijk[hop]
            if cp.array_equal(s_ijk, g_ijk):
                continue

            result = space_time_astar_gpu(
                coarse_grid, s_ijk, g_ijk, t_cursor, reservation, coarse_res,
            )

            if result is None:
                stats.astar_retries += 1
                result = space_time_astar_gpu(
                    coarse_grid, s_ijk, g_ijk, t_cursor, reservation,
                    coarse_res, max_iterations=2 * GPU_SEARCH_MAX_ITERATIONS,
                )

            if result is None:
                stats.astar_failures += 1
                logger.warning(
                    "[route_planner] ST-A* failed hop %d->%d (leg %d->%d, t=%d).",
                    hop - 1, hop, prev_node, curr_node, t_cursor,
                )
                continue

            seg_ijk, seg_t = result
            start_idx = 0 if (not coarse_positions and not leg_plan_pos) else 1
            if len(seg_ijk) > start_idx:
                leg_plan_pos.append(seg_ijk[start_idx:])
                leg_plan_t.append(seg_t[start_idx:])
            t_cursor = int(seg_t[-1])

        if not leg_plan_pos:
            if coarse_positions:
                last = coarse_positions[-1][-1:]
                last_world = world_positions[-1][-1:]
                dwell_ijk = cp.tile(last, (hold_steps, 1))
                dwell_world = cp.tile(last_world, (hold_steps, 1))
                dwell_t = cp.arange(
                    t_cursor + 1, t_cursor + hold_steps + 1, dtype=cp.intp,
                )
                coarse_positions.append(dwell_ijk)
                world_positions.append(dwell_world)
                coarse_times.append(dwell_t)
                t_dwell_start = int(t_cursor) + 1
                t_cursor += hold_steps
                wp_schedule.append((t_dwell_start, int(t_cursor), curr_node))
            continue

        planned_ijk = cp.concatenate(leg_plan_pos, axis=0)
        planned_t = cp.concatenate(leg_plan_t, axis=0)

        final_ijk = planned_ijk
        final_world = coarse_to_world(planned_ijk, coarse_origin, coarse_res)

        if fine_occupancy_grid is not None and len(planned_ijk) >= 3:
            planned_world = coarse_to_world(planned_ijk, coarse_origin, coarse_res)
            smoothed_world = simplify_path_ompl(
                planned_world, fine_occupancy_grid, robot_radius,
                max_time=OMPL_SIMPLIFY_MAX_TIME,
            )

            if len(smoothed_world) >= 2:
                resampled_world = arc_length_resample(smoothed_world, len(planned_t))
                resampled_ijk = world_to_coarse(resampled_world, coarse_origin, coarse_res)
                for d, mx in enumerate(grid_shape):
                    resampled_ijk[:, d] = cp.clip(resampled_ijk[:, d], 0, mx - 1)

                pos_check = resampled_ijk
                conflict = reservation.is_reserved_batch(pos_check, planned_t).any()

                if not bool(conflict):
                    final_ijk = resampled_ijk
                    final_world = resampled_world
                    logger.info("[route_planner] leg %d->%d OMPL smoothed: %d->%d wpts",
                                prev_node, curr_node, len(planned_ijk), len(smoothed_world))
                else:
                    logger.info("[route_planner] leg %d->%d OMPL path has reservation "
                                "conflict, using A* path.", prev_node, curr_node)

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
        return (cp.empty((0, 3), dtype=cp.float64),
                cp.empty((0,), dtype=cp.intp), [], stats)

    all_ijk = cp.concatenate(coarse_positions, axis=0)
    all_t = cp.concatenate(coarse_times, axis=0)

    reservation.commit_trajectory(all_ijk, all_t)

    world_xyz = cp.concatenate(world_positions, axis=0)

    if len(world_xyz) >= 2:
        actual_dist = float(cp.sum(cp.linalg.norm(cp.diff(world_xyz, axis=0), axis=1)))
        stats.detour_ratio = (actual_dist / straight_line_total
                              if straight_line_total > 1e-9 else 1.0)
        diffs_norm = cp.linalg.norm(cp.diff(world_xyz, axis=0), axis=1)
        stats.wait_steps = int(cp.sum(diffs_norm < 1e-6))

    return world_xyz, all_t, wp_schedule, stats
