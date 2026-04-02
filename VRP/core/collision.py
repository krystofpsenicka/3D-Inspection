"""AABB inter-robot collision detection."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .constants import BROV_CUBOID_DIMS


def find_trajectory_collisions(
    all_traj_positions: List[List[np.ndarray]],
    dims: Optional[np.ndarray] = None,
) -> List[Tuple[int, int, int, float]]:
    """Scan replay trajectories for AABB inter-robot collisions.

    Returns list of (step, robot_a, robot_b, penetration_depth).
    """
    if dims is None:
        dims = np.array(BROV_CUBOID_DIMS, dtype=np.float32)
    dims = np.asarray(dims, dtype=np.float32)
    half = dims / 2.0

    num_robots = len(all_traj_positions)
    total_steps = max(len(t) for t in all_traj_positions)
    collisions: list = []

    for step in range(total_steps):
        for a in range(num_robots):
            if step >= len(all_traj_positions[a]):
                pos_a = all_traj_positions[a][-1][:3]
            else:
                pos_a = all_traj_positions[a][step][:3]
            for b in range(a + 1, num_robots):
                if step >= len(all_traj_positions[b]):
                    pos_b = all_traj_positions[b][-1][:3]
                else:
                    pos_b = all_traj_positions[b][step][:3]
                diff = np.abs(pos_a - pos_b)
                gap = diff - 2.0 * half
                if np.all(gap < 0):
                    penetration = float(-np.max(gap))
                    collisions.append((step, a, b, penetration))
    return collisions
