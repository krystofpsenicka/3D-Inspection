"""SamplingOccupancyGrid — child class of OccupancyGrid with extra fields + methods for sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cupy as cp
import numpy as np

from shared.occupancy_grid import OccupancyGrid


@dataclass
class SamplingOccupancyGrid(OccupancyGrid):
    """OccupancyGrid extended with raw and filled grids for SDF/sampling.

    Attributes
    ----------
    raw_grid : cp.ndarray, dtype=bool
        Pre-inflation obstacle grid (surface-only or filled, no dilation).
    filled_raw_grid : cp.ndarray, dtype=bool
        Flood-filled voxelization for two-EDT SDF computation.
    mesh_scale : float | None
        Scale factor applied to the mesh.
    """

    raw_grid: cp.ndarray
    filled_raw_grid: cp.ndarray
    mesh_scale: Optional[float] = None

    def sample_random_free_points(
        self, n: int, rng: Optional[cp.random.RandomState] = None
    ) -> cp.ndarray:
        """Return ``(n, 3)`` random world-frame points inside free voxels."""
        free_ijk = cp.argwhere(~self.grid)
        indices = cp.random.randint(0, len(free_ijk), size=n)
        chosen = free_ijk[indices]
        offsets = cp.random.uniform(0.0, self.resolution, size=(n, 3)).astype(cp.float64)
        return chosen.astype(cp.float64) * self.resolution + self.origin + offsets

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save all fields to NPZ + JSON."""
        import json
        base = path.rsplit(".", 1)[0] if "." in path else path
        np.savez_compressed(
            base + ".npz",
            grid=cp.asnumpy(self.grid),
            raw_grid=cp.asnumpy(self.raw_grid),
            filled_raw_grid=cp.asnumpy(self.filled_raw_grid),
        )
        meta = {
            "resolution": self.resolution,
            "origin": cp.asnumpy(self.origin).tolist(),
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
                grid=cp.asarray(data["grid"]),
                origin=cp.asarray(meta["origin"], dtype=cp.float64),
                resolution=meta["resolution"],
                raw_grid=cp.asarray(data["raw_grid"]),
                filled_raw_grid=cp.asarray(data["filled_raw_grid"]),
                mesh_scale=meta.get("mesh_scale"),
            )
