"""
Shared Occupancy Grid
=====================

Core ``OccupancyGrid`` dataclass and pure-numpy utilities that are used by
both the VRP planner and the visibility/sampling pipeline.

Moved here from ``VRP/occupancy_grid.py`` and ``VRP/space_time_astar.py``
so that non-VRP code (e.g. viewpoint sampling) can use them without pulling
in VRP-specific configuration.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OccupancyGrid:
    """3D binary occupancy grid in voxel coordinates.

    Attributes
    ----------
    grid : np.ndarray, dtype=bool, shape (Nx, Ny, Nz)
        ``True`` = occupied / collision, ``False`` = free.
    origin : np.ndarray (3,)
        World position of voxel (0, 0, 0).
    resolution : float
        Metres per voxel edge.
    raw_grid : np.ndarray | None
        Pre-inflation obstacle grid (mesh + cuboids, no dilation).
        Used by the ESDF builder so cuRobo collision spheres are not
        double-counted with the inflation radius.
    mesh_scale : float | None
        Uniform scale factor applied to the mesh so its longest axis
        equals ``MESH_TARGET_LENGTH``.  Needed by visualisation so the
        rendered mesh matches the occupancy grid.
    """

    grid: np.ndarray
    origin: np.ndarray
    resolution: float
    raw_grid: Optional[np.ndarray] = None
    filled_raw_grid: Optional[np.ndarray] = None
    mesh_scale: Optional[float] = None

    # ── Coordinate transforms ─────────────────────────────────────────────────

    def world_to_voxel(self, world_xyz: np.ndarray) -> np.ndarray:
        """Convert world-frame ``(..., 3)`` coords → integer voxel indices.

        Returned indices are *not* clipped; callers should use
        ``is_valid_voxel`` before indexing into the grid.
        """
        return np.floor(
            (world_xyz - self.origin) / self.resolution
        ).astype(int)

    def voxel_to_world(self, voxel_ijk: np.ndarray) -> np.ndarray:
        """Convert integer voxel indices ``(..., 3)`` → world-frame centre (m)."""
        return voxel_ijk.astype(float) * self.resolution + self.origin + self.resolution * 0.5

    def is_valid_voxel(self, ijk: np.ndarray) -> bool:
        """Return True if ``ijk`` is inside the grid bounds."""
        ijk = np.asarray(ijk)
        return bool(
            np.all(ijk >= 0) and np.all(ijk < np.array(self.grid.shape))
        )

    def is_free_world(self, world_xyz: np.ndarray) -> bool:
        """Return True if the world-frame point is in a free voxel."""
        ijk = self.world_to_voxel(world_xyz)
        if not self.is_valid_voxel(ijk):
            return False
        return not bool(self.grid[tuple(ijk)])

    def is_free_world_batch(self, world_xyz: np.ndarray) -> np.ndarray:
        """Check (N, 3) world positions → (N,) bool. Out-of-bounds → False."""
        ijk = self.world_to_voxel(world_xyz)
        shape = np.array(self.grid.shape)
        in_bounds = np.all(ijk >= 0, axis=1) & np.all(ijk < shape, axis=1)
        result = np.zeros(len(ijk), dtype=bool)
        valid_ijk = ijk[in_bounds]
        result[in_bounds] = ~self.grid[valid_ijk[:, 0], valid_ijk[:, 1], valid_ijk[:, 2]]
        return result

    def world_to_flat_index(self, world_xyz: np.ndarray) -> int:
        """Return the flat (C-order) grid index for a world-frame point."""
        ijk = self.world_to_voxel(world_xyz)
        return int(np.ravel_multi_index(tuple(ijk), self.grid.shape))

    def flat_index_to_world(self, flat_idx: int) -> np.ndarray:
        """Inverse of ``world_to_flat_index``."""
        ijk = np.array(np.unravel_index(flat_idx, self.grid.shape))
        return self.voxel_to_world(ijk)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(self.grid.shape)  # type: ignore[return-value]

    @property
    def num_free(self) -> int:
        return int(np.sum(~self.grid))

    @property
    def num_occupied(self) -> int:
        return int(np.sum(self.grid))

    # ── Sampling ──────────────────────────────────────────────────────────────

    def sample_random_free_points(
        self, n: int, rng: Optional[np.random.RandomState] = None
    ) -> np.ndarray:
        """Return ``(n, 3)`` random world-frame points inside free voxels."""
        if rng is None:
            rng = np.random.RandomState()
        free_ijk = np.argwhere(~self.grid)           # (F, 3)
        if len(free_ijk) < n:
            raise ValueError(
                f"Grid has only {len(free_ijk)} free voxels; requested {n}."
            )
        chosen = free_ijk[rng.choice(len(free_ijk), n, replace=False)]
        # Random sub-voxel offset for variety
        offsets = rng.uniform(0.0, self.resolution, size=(n, 3))
        return chosen.astype(float) * self.resolution + self.origin + offsets

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Pickle the grid to disk for caching."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "OccupancyGrid":
        with open(path, "rb") as f:
            return pickle.load(f)


# ── Grid inflation ────────────────────────────────────────────────────────────

def inflate_grid(grid: np.ndarray, inflation_voxels: int) -> np.ndarray:
    """Morphological dilation of the obstacle grid by *inflation_voxels* voxels.

    Uses scipy's binary_dilation which is equivalent to a 3D sphere structuring
    element of radius ``inflation_voxels``.  This expands every obstacle by
    the robot's collision radius so that path planners can treat the robot
    as a point.
    """
    from scipy.ndimage import binary_dilation
    # Build a spherical structuring element
    r = inflation_voxels
    d = 2 * r + 1
    se = np.zeros((d, d, d), dtype=bool)
    cx, cy, cz = r, r, r
    for ix in range(d):
        for iy in range(d):
            for iz in range(d):
                if (ix - cx) ** 2 + (iy - cy) ** 2 + (iz - cz) ** 2 <= r ** 2:
                    se[ix, iy, iz] = True
    return binary_dilation(grid, structure=se)


# ── Grid down-sampling ───────────────────────────────────────────────────────

def downsample_occupancy_grid(
    fine_grid: np.ndarray,
    fine_origin: np.ndarray,
    fine_res: float,
    coarse_res: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Down-sample an occupancy grid.

    A coarse voxel is **occupied** if **any** of its constituent fine
    voxels is occupied (conservative -- no false free-space).

    Fully vectorised with ``np.pad`` + ``reshape`` + ``any``.

    Returns ``(coarse_grid, coarse_origin, coarse_res)``.
    """
    factor = max(1, int(round(coarse_res / fine_res)))
    Fx, Fy, Fz = fine_grid.shape

    # Pad each axis to the next multiple of *factor* so reshape is exact.
    pad_x = (-Fx) % factor
    pad_y = (-Fy) % factor
    pad_z = (-Fz) % factor
    if pad_x or pad_y or pad_z:
        fine_padded = np.pad(
            fine_grid.astype(bool),
            [(0, pad_x), (0, pad_y), (0, pad_z)],
            constant_values=False,
        )
    else:
        fine_padded = fine_grid.astype(bool)

    Cx = fine_padded.shape[0] // factor
    Cy = fine_padded.shape[1] // factor
    Cz = fine_padded.shape[2] // factor

    coarse = (
        fine_padded
        .reshape(Cx, factor, Cy, factor, Cz, factor)
        .any(axis=(1, 3, 5))
    )

    coarse_origin = fine_origin.copy()
    actual_res = fine_res * factor
    logger.info(
        "[downsample] Grid: %s → %s  (factor=%d, coarse_res=%.2fm)",
        fine_grid.shape, coarse.shape, factor, actual_res,
    )
    return coarse, coarse_origin, actual_res
