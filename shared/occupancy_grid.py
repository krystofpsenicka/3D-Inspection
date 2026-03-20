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
        """Save the grid to NPZ (arrays) + JSON (metadata)."""
        import json
        base = path.rsplit(".", 1)[0] if "." in path else path
        np.savez_compressed(
            base + ".npz",
            grid=self.grid,
            raw_grid=self.raw_grid if self.raw_grid is not None else np.empty(0),
            filled_raw_grid=self.filled_raw_grid if self.filled_raw_grid is not None else np.empty(0),
        )
        meta = {
            "resolution": self.resolution,
            "origin": self.origin.tolist(),
            "mesh_scale": self.mesh_scale,
        }
        with open(base + ".json", "w") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, path: str) -> "OccupancyGrid":
        import json
        import os
        base = path.rsplit(".", 1)[0] if "." in path else path
        npz_path = base + ".npz"
        json_path = base + ".json"
        if os.path.exists(npz_path) and os.path.exists(json_path):
            data = np.load(npz_path)
            with open(json_path) as f:
                meta = json.load(f)
            raw = data["raw_grid"] if data["raw_grid"].size > 0 else None
            filled = data["filled_raw_grid"] if data["filled_raw_grid"].size > 0 else None
            return cls(
                grid=data["grid"],
                origin=np.array(meta["origin"]),
                resolution=meta["resolution"],
                raw_grid=raw,
                filled_raw_grid=filled,
                mesh_scale=meta.get("mesh_scale"),
            )
        # Backward compat: try pickle
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


# ── Grid inflation ────────────────────────────────────────────────────────────

def inflate_grid(grid: np.ndarray, inflation_voxels: int) -> np.ndarray:
    """Morphological dilation of the obstacle grid by *inflation_voxels* voxels.

    Uses a spherical structuring element of radius ``inflation_voxels``.
    This expands every obstacle by the robot's collision radius so that
    path planners can treat the robot as a point.

    Tries CuPy (GPU) first for speed; falls back to scipy on CPU.
    """
    r = inflation_voxels
    coords = np.mgrid[-r:r+1, -r:r+1, -r:r+1]
    se = (coords[0]**2 + coords[1]**2 + coords[2]**2) <= r**2

    try:
        import cupy as cp
        from cupyx.scipy.ndimage import binary_dilation as gpu_dilation
        grid_gpu = cp.asarray(grid)
        se_gpu = cp.asarray(se)
        result = gpu_dilation(grid_gpu, structure=se_gpu)
        return cp.asnumpy(result)
    except (ImportError, Exception) as exc:
        logger.debug("GPU inflate_grid unavailable (%s), using scipy.", exc)
        from scipy.ndimage import binary_dilation
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
