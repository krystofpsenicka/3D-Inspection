"""
Occupancy Grid
==============

``OccupancyGrid`` dataclass and numpy utilities that are used by
the VRP planner and the visibility/sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class OccupancyGrid:
    """3D binary occupancy grid.

    Attributes
    ----------
    grid : np.ndarray, dtype=bool, shape (Nx, Ny, Nz)
        ``True`` = occupied / collision, ``False`` = free.
    origin : np.ndarray (3,)
        World position of voxel (0, 0, 0).
    resolution : float
        Voxel edge length (metres).
    """

    grid: np.ndarray
    origin: np.ndarray
    resolution: float

    # ── Coordinate transforms ───────────────────────────────────────────────

    def world_to_voxel(self, world_xyz: np.ndarray) -> np.ndarray:
        """Convert world-frame ``(..., 3)`` coords -> integer voxel indices.

        Returned indices are *not* clipped; callers should use
        ``is_valid_voxel`` before indexing into the grid.
        """
        return np.floor(
            (world_xyz - self.origin) / self.resolution
        ).astype(int)

    def voxel_to_world(self, voxel_ijk: np.ndarray) -> np.ndarray:
        """Convert integer voxel indices ``(..., 3)`` -> world-frame."""
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
        """Check (N, 3) world positions -> (N,) bool. Out of bounds -> False."""
        ijk = self.world_to_voxel(world_xyz)
        shape = np.array(self.grid.shape)
        in_bounds = np.all(ijk >= 0, axis=1) & np.all(ijk < shape, axis=1)
        result = np.zeros(len(ijk), dtype=bool)
        valid_ijk = ijk[in_bounds]
        result[in_bounds] = ~self.grid[valid_ijk[:, 0], valid_ijk[:, 1], valid_ijk[:, 2]]
        return result

    def world_to_flat_index(self, world_xyz: np.ndarray) -> int:
        """Return the flat grid index for a world-frame point."""
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

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save the grid to NPZ (arrays) + JSON (metadata)."""
        import json
        base = path.rsplit(".", 1)[0] if "." in path else path
        np.savez_compressed(base + ".npz", grid=self.grid)
        meta = {
            "resolution": self.resolution,
            "origin": self.origin.tolist(),
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
                grid=data["grid"],
                origin=np.array(meta["origin"]),
                resolution=meta["resolution"],
            )
