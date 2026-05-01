"""Collision detection for trajectory safety checks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import cupy as cp
import numpy as np

from shared.occupancy_grid import OccupancyGrid
from VRP.core.constants import ROBOT_RADIUS

ArrayLike2D = Union[cp.ndarray, np.ndarray]


def _pad_trajectories_to_tensor(
    all_traj_positions: Sequence[ArrayLike2D],
) -> tuple[cp.ndarray, list[int]]:
    """Pad ragged per-robot trajectories into a (R, T_max, 3) tensor.

    Each element of all_traj_positions is a (T_i, D) array (NumPy or
    CuPy) where D >= 3. Only the first 3 columns (XYZ) are used.

    Returns (stacked, lengths) where stacked is (R, T_max, 3) and
    lengths[i] is the original length of robot i's trajectory.
    """
    num_robots = len(all_traj_positions)
    lengths = [len(t) for t in all_traj_positions]
    max_T = max(lengths) if lengths else 0
    if max_T == 0:
        return cp.empty((num_robots, 0, 3), dtype=cp.float32), lengths

    stacked = cp.zeros((num_robots, max_T, 3), dtype=cp.float32)
    for i, traj in enumerate(all_traj_positions):
        n = len(traj)
        if n == 0:
            continue
        traj_gpu = cp.asarray(traj)
        stacked[i, :n] = traj_gpu[:, :3]
        if n < max_T:
            stacked[i, n:] = traj_gpu[-1, :3]

    return stacked, lengths


def find_trajectory_collisions(
    all_traj_positions: Sequence[ArrayLike2D],
    radius: float = ROBOT_RADIUS,
) -> list[tuple[int, int, int, float]]:
    """Scan replay trajectories for sphere-based inter-robot collisions.

    Vectorized on GPU: pads all trajectories to equal length, stacks
    into ``(R, T, 3)``, and broadcasts pairwise distance checks for all
    robot pairs and time-steps simultaneously.

    Args:
        all_traj_positions: Per-robot arrays (NumPy or CuPy), each (T_i, D) where D >= 3.
        radius: Bounding-sphere radius per robot. Defaults to ROBOT_RADIUS.

    Returns list of (step, robot_a, robot_b, penetration_depth).
    """
    num_robots = len(all_traj_positions)
    if num_robots < 2:
        return []

    stacked, lengths = _pad_trajectories_to_tensor(all_traj_positions)
    if stacked.shape[1] == 0:
        return []

    # Pairwise distances: (R, 1, T, 3) vs (1, R, T, 3)
    a = stacked[:, None, :, :]
    b = stacked[None, :, :, :]
    dist = cp.linalg.norm(a - b, axis=3)  # (R, R, T)

    # Collision when distance < 2 * radius
    collision_threshold = 2.0 * radius
    overlap = dist < collision_threshold

    # Upper triangle only (a < b)
    ri, rj, ti = cp.where(overlap)
    mask = ri < rj
    ri, rj, ti = ri[mask], rj[mask], ti[mask]

    if len(ri) == 0:
        return []

    penetrations = collision_threshold - dist[ri, rj, ti]

    ri_cpu = cp.asnumpy(ri)
    rj_cpu = cp.asnumpy(rj)
    ti_cpu = cp.asnumpy(ti)
    pen_cpu = cp.asnumpy(penetrations)

    return [
        (int(ti_cpu[k]), int(ri_cpu[k]), int(rj_cpu[k]), float(pen_cpu[k]))
        for k in range(len(ri_cpu))
    ]


def find_environment_collisions(
    all_traj_positions: Sequence[ArrayLike2D],
    occupancy_grid: OccupancyGrid,
) -> list[int]:
    """Count per-robot collisions with the occupancy grid.

    Flattens all robot positions into one batch, runs a single
    ``is_free_world_batch`` check, then splits results
    per robot.

    Args:
        all_traj_positions: Per-robot arrays (NumPy or CuPy), each (T_i, D) where D >= 3.
        occupancy_grid: Fine-resolution occupancy grid.

    Returns:
        Per-robot collision counts.
    """
    lengths = [len(t) for t in all_traj_positions]
    if sum(lengths) == 0:
        return [0] * len(all_traj_positions)

    all_xyz = cp.concatenate(
        [cp.asarray(t)[:, :3] for t in all_traj_positions if len(t) > 0],
        axis=0,
    )
    free_mask = occupancy_grid.is_free_world_batch(all_xyz)

    collision_counts: list[int] = []
    offset = 0
    for length in lengths:
        if length == 0:
            collision_counts.append(0)
            continue
        robot_collisions = int(cp.sum(~free_mask[offset : offset + length]))
        collision_counts.append(robot_collisions)
        offset += length

    return collision_counts
