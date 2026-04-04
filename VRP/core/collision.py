"""AABB inter-robot collision detection (GPU-vectorized)."""

from __future__ import annotations

from typing import List, Optional, Tuple

import cupy as cp
import numpy as np

from .constants import BROV_CUBOID_DIMS


def find_trajectory_collisions(
    all_traj_positions: List[List[np.ndarray | cp.ndarray]],
    dims: Optional[np.ndarray | cp.ndarray] = None,
) -> List[Tuple[int, int, int, float]]:
    """Scan replay trajectories for AABB inter-robot collisions.

    Vectorized on GPU: pads all trajectories to equal length, stacks
    into ``(R, T, 3)``, and broadcasts pairwise AABB overlap for all
    robot pairs and time-steps simultaneously.

    Returns list of (step, robot_a, robot_b, penetration_depth).
    """
    if dims is None:
        dims = np.array(BROV_CUBOID_DIMS, dtype=np.float32)
    half = cp.asarray(dims, dtype=cp.float32) / 2.0

    num_robots = len(all_traj_positions)
    if num_robots < 2:
        return []

    # Pad trajectories to equal length and stack into (R, T, 3)
    lengths = [len(t) for t in all_traj_positions]
    T = max(lengths)
    if T == 0:
        return []

    stacked = cp.empty((num_robots, T, 3), dtype=cp.float32)
    for i, traj in enumerate(all_traj_positions):
        n = len(traj)
        if n == 0:
            stacked[i] = 0.0
            continue
        # Stack XYZ only (first 3 components of each joint-space vector)
        arr = cp.array([cp.asarray(p[:3]) for p in traj], dtype=cp.float32)
        stacked[i, :n] = arr
        # Pad by repeating last position
        if n < T:
            stacked[i, n:] = arr[-1]

    # Pairwise AABB overlap: (R, 1, T, 3) vs (1, R, T, 3)
    a = stacked[:, None, :, :]  # (R, 1, T, 3)
    b = stacked[None, :, :, :]  # (1, R, T, 3)
    gap = cp.abs(a - b) - 2.0 * half  # (R, R, T, 3)

    # Collision where all 3 axes overlap (gap < 0)
    overlap = cp.all(gap < 0, axis=3)  # (R, R, T)

    # Only upper triangle (a < b)
    ri, rj, ti = cp.where(overlap)
    mask = ri < rj
    ri, rj, ti = ri[mask], rj[mask], ti[mask]

    if len(ri) == 0:
        return []

    # Compute penetration depths
    penetrations = -cp.max(gap[ri, rj, ti], axis=1)

    # Transfer to CPU and build result list
    ri_cpu = cp.asnumpy(ri)
    rj_cpu = cp.asnumpy(rj)
    ti_cpu = cp.asnumpy(ti)
    pen_cpu = cp.asnumpy(penetrations)

    return [
        (int(ti_cpu[k]), int(ri_cpu[k]), int(rj_cpu[k]), float(pen_cpu[k]))
        for k in range(len(ri_cpu))
    ]
