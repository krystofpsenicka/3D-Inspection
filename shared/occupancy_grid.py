"""
Occupancy Grid
==============

``OccupancyGrid`` dataclass with GPU-resident CuPy arrays, used by
the VRP planner and the visibility/sampling pipeline.

All grid data lives on GPU. Conversion to CPU happens only at I/O
boundaries (save/load, OMPL).
"""

from __future__ import annotations

from dataclasses import dataclass

import cupy as cp
import numpy as np


@dataclass
class OccupancyGrid:
    """3D binary occupancy grid (GPU-resident).

    Attributes
    ----------
    grid : cp.ndarray, dtype=bool, shape (Nx, Ny, Nz)
        ``True`` = occupied / collision, ``False`` = free.
    origin : cp.ndarray (3,)
        World position of voxel (0, 0, 0).
    resolution : float
        Voxel edge length (metres).
    """

    grid: cp.ndarray
    origin: cp.ndarray
    resolution: float

    # ── Coordinate transforms ───────────────────────────────────────────────

    def world_to_voxel(self, world_xyz: cp.ndarray) -> cp.ndarray:
        """Convert world-frame ``(..., 3)`` coords -> integer voxel indices."""
        return cp.floor((world_xyz - self.origin) / self.resolution).astype(cp.int32)

    def voxel_to_world(self, voxel_ijk: cp.ndarray) -> cp.ndarray:
        """Convert integer voxel indices ``(..., 3)`` -> world-frame."""
        return voxel_ijk.astype(cp.float64) * self.resolution + self.origin + self.resolution * 0.5

    def is_valid_voxel(self, ijk: cp.ndarray) -> bool:
        """Return True if ``ijk`` is inside the grid bounds."""
        return bool(cp.all(ijk >= 0) and cp.all(ijk < cp.array(self.grid.shape)))

    def is_free_world(self, world_xyz: cp.ndarray) -> bool:
        """Return True if the world-frame point is in a free voxel."""
        ijk = self.world_to_voxel(world_xyz)
        if not self.is_valid_voxel(ijk):
            return False
        return not bool(self.grid[int(ijk[0]), int(ijk[1]), int(ijk[2])])

    def is_free_world_batch(self, world_xyz: cp.ndarray) -> cp.ndarray:
        """Check (N, 3) world positions -> (N,) bool. Out of bounds -> False."""
        ijk = self.world_to_voxel(world_xyz)
        shape = cp.array(self.grid.shape, dtype=cp.int32)
        in_bounds = cp.all(ijk >= 0, axis=1) & cp.all(ijk < shape, axis=1)
        result = cp.zeros(len(ijk), dtype=cp.bool_)
        valid_ijk = ijk[in_bounds]
        if len(valid_ijk) > 0:
            result[in_bounds] = ~self.grid[valid_ijk[:, 0], valid_ijk[:, 1], valid_ijk[:, 2]]
        return result

    def world_to_flat_index(self, world_xyz: cp.ndarray) -> int:
        """Return the flat grid index for a world-frame point."""
        ijk = self.world_to_voxel(world_xyz)
        Ny, Nz = self.grid.shape[1], self.grid.shape[2]
        return int(ijk[0] * (Ny * Nz) + ijk[1] * Nz + ijk[2])

    def flat_index_to_world(self, flat_idx: int) -> cp.ndarray:
        """Inverse of ``world_to_flat_index``."""
        Ny, Nz = self.grid.shape[1], self.grid.shape[2]
        i = flat_idx // (Ny * Nz)
        j = (flat_idx % (Ny * Nz)) // Nz
        k = flat_idx % Nz
        ijk = cp.array([i, j, k])
        return self.voxel_to_world(ijk)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.grid.shape)  # type: ignore[return-value]

    @property
    def num_free(self) -> int:
        return int(cp.sum(~self.grid))

    @property
    def num_occupied(self) -> int:
        return int(cp.sum(self.grid))

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save the grid to NPZ (arrays) + JSON (metadata)."""
        import json

        base = path.rsplit(".", 1)[0] if "." in path else path
        np.savez_compressed(base + ".npz", grid=cp.asnumpy(self.grid))
        meta = {
            "resolution": self.resolution,
            "origin": cp.asnumpy(self.origin).tolist(),
        }
        with open(base + ".json", "w") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, path: str) -> OccupancyGrid:
        import json
        import os

        base = path.rsplit(".", 1)[0] if "." in path else path
        npz_path = base + ".npz"
        json_path = base + ".json"
        if os.path.exists(npz_path) and os.path.exists(json_path):
            data = np.load(npz_path)
            with open(json_path) as f:
                meta = json.load(f)
            return cls(
                grid=cp.asarray(data["grid"]),
                origin=cp.asarray(meta["origin"], dtype=cp.float64),
                resolution=meta["resolution"],
            )
