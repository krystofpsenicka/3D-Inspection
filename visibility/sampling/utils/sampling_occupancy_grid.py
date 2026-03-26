"""SamplingOccupancyGrid — child class of OccupancyGrid with extra fields + methods for sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from shared.occupancy_grid import OccupancyGrid


@dataclass
class SamplingOccupancyGrid(OccupancyGrid):
    """OccupancyGrid extended with raw and filled grids for SDF/sampling.

    Attributes
    ----------
    raw_grid : np.ndarray, dtype=bool
        Pre-inflation obstacle grid (surface-only or filled, no dilation).
    filled_raw_grid : np.ndarray, dtype=bool
        Flood-filled voxelization for two-EDT SDF computation.
    mesh_scale : float | None
        Scale factor applied to the mesh.
    """

    raw_grid: np.ndarray
    filled_raw_grid: np.ndarray
    mesh_scale: Optional[float] = None

    def sample_random_free_points(
        self, n: int, rng: Optional[np.random.RandomState] = None
    ) -> np.ndarray:
        """Return ``(n, 3)`` random world-frame points inside free voxels."""
        if rng is None:
            rng = np.random.RandomState()
        free_ijk = np.argwhere(~self.grid)
        chosen = free_ijk[rng.choice(len(free_ijk), n, replace=True)]
        # Add random offsets within the voxel
        offsets = rng.uniform(0.0, self.resolution, size=(n, 3))
        return chosen.astype(float) * self.resolution + self.origin + offsets

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save all fields to NPZ + JSON."""
        import json
        base = path.rsplit(".", 1)[0] if "." in path else path
        np.savez_compressed(
            base + ".npz",
            grid=self.grid,
            raw_grid=self.raw_grid,
            filled_raw_grid=self.filled_raw_grid,
        )
        meta = {
            "resolution": self.resolution,
            "origin": self.origin.tolist(),
            "mesh_scale": self.mesh_scale,
        }
        with open(base + ".json", "w") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, path: str) -> SamplingOccupancyGrid:
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
                raw_grid=data["raw_grid"],
                filled_raw_grid=data["filled_raw_grid"],
                mesh_scale=meta.get("mesh_scale"),
            )
