"""Dense 4D (T, Nx, Ny, Nz) reservation table on GPU.

Following Silver (2005), the reservation table records which
space-time cells are occupied by previously planned robots, allowing
subsequent robots to avoid collisions by construction.

The table uses spherical inflation: each committed trajectory sample is
expanded by a ball of the given collision radius (in voxel units).

References:
    Silver, D. (2005). Cooperative Pathfinding. AIIDE.
"""

from __future__ import annotations

import logging
import math
from typing import Tuple

import cupy as cp

logger = logging.getLogger(__name__)


class ReservationTable:
    """Dense 4D (T, Nx, Ny, Nz) reservation table on GPU.

    Args:
        grid_shape: (Nx, Ny, Nz) coarse spatial dimensions.
        max_time_steps: number of discrete time slots.
        robot_collision_radius: collision radius in **voxel units** (float).
            Commits inflate point trajectories by a ball of this radius.
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int, int],
        max_time_steps: int,
        robot_collision_radius: float,
    ):
        self.Nx, self.Ny, self.Nz = grid_shape
        self.T = max_time_steps
        self._data = cp.zeros(
            (self.T, self.Nx, self.Ny, self.Nz), dtype=cp.uint8,
        )
        self._radius = float(robot_collision_radius)
        self._radius_ceil = int(math.ceil(self._radius))

        # Precompute ball offsets: (B, 3) int32 array of (dx, dy, dz) where
        # dx^2 + dy^2 + dz^2 <= radius^2
        r = self._radius_ceil
        if r == 0:
            self._ball_offsets = cp.array([[0, 0, 0]], dtype=cp.int32)
        else:
            coords = cp.mgrid[-r:r + 1, -r:r + 1, -r:r + 1]
            dist_sq = (coords[0] ** 2 + coords[1] ** 2 + coords[2] ** 2).astype(cp.float32)
            mask = dist_sq <= self._radius ** 2
            self._ball_offsets = cp.stack(cp.where(mask), axis=1).astype(cp.int32) - r

    def is_reserved(self, x: int, y: int, z: int, t: int) -> bool:
        """Scalar query -- pulls one element to CPU for control flow."""
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
        positions_ijk: cp.ndarray,
        time_steps: cp.ndarray,
    ) -> None:
        """Mark all voxels within the collision sphere around each (t, x, y, z) sample."""
        K = len(positions_ijk)
        if K == 0:
            return

        ts = time_steps.astype(cp.intp)
        pos = positions_ijk.astype(cp.intp)

        valid = (ts >= 0) & (ts < self.T)
        if not valid.any():
            return
        ts = ts[valid]
        pos = pos[valid]
        K = len(pos)

        B = len(self._ball_offsets)

        # Broadcast: (K, 1, 3) + (1, B, 3) -> (K, B, 3)
        inflated = pos[:, None, :] + self._ball_offsets[None, :, :]  # (K, B, 3)
        inflated_t = cp.broadcast_to(ts[:, None], (K, B))  # (K, B)

        # Flatten to (K*B, 3) and (K*B,)
        flat_pos = inflated.reshape(-1, 3)
        flat_t = inflated_t.reshape(-1)

        # Bounds check
        valid = (
            (flat_pos[:, 0] >= 0) & (flat_pos[:, 0] < self.Nx)
            & (flat_pos[:, 1] >= 0) & (flat_pos[:, 1] < self.Ny)
            & (flat_pos[:, 2] >= 0) & (flat_pos[:, 2] < self.Nz)
            & (flat_t >= 0) & (flat_t < self.T)
        )
        if not valid.any():
            return
        flat_pos = flat_pos[valid]
        flat_t = flat_t[valid]

        self._data[flat_t, flat_pos[:, 0], flat_pos[:, 1], flat_pos[:, 2]] = 1
