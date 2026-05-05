"""Visualize ESDF voxel grids in 3D (Isaac Sim variant)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import trimesh

from shared.grid_utils import esdf_to_rgb
from shared.occupancy_grid import OccupancyGrid

from .._usd_primitives import (
    create_coordinate_frame_prim,
    create_mesh_prim,
    create_points_prim,
)

logger = logging.getLogger(__name__)

MESH_COLOR = (0.6, 0.6, 0.6)
OCCUPIED_COLOR = (1.0, 0.7, 0.0)
INFLATED_COLOR = (0.0, 0.7, 0.3)
COORD_FRAME_SIZE = 2.0


@dataclass
class _PreparedPoints:
    positions: np.ndarray
    colors: np.ndarray | tuple
    n_total: int


@dataclass
class _PreparedScene:
    band: _PreparedPoints | None
    occupied: _PreparedPoints | None
    inflated: _PreparedPoints | None
    esdf_band: float


def _subsample(ijk: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(ijk) <= max_points:
        return ijk
    rng = np.random.RandomState(seed)
    return ijk[rng.choice(len(ijk), max_points, replace=False)]


def _to_numpy(arr):
    if isinstance(arr, np.ndarray):
        return arr
    get = getattr(arr, "get", None)
    if callable(get):
        return get()
    return np.asarray(arr)


class EsdfVisualizer:
    """Heavy numpy work belongs in :meth:`prepare`, called *before* the IsaacApp context manager
    opens — Kit auto-shuts the app down if the main thread doesn't pump ``app.update()`` for ~1-2s,
    so data must be ready when ``add_3d`` is invoked from a phase enter callback."""

    def __init__(
        self,
        occupancy_grid: OccupancyGrid,
        raw_grid: np.ndarray,
        esdf: np.ndarray,
        scaled_mesh: trimesh.Trimesh | None = None,
    ):
        self.og = occupancy_grid
        self.raw = raw_grid
        self.esdf = esdf
        self.scaled_mesh = scaled_mesh
        self._prepared: _PreparedScene | None = None

    def prepare(
        self,
        esdf_band: float = 2.0,
        show_occupied: bool = False,
        show_inflated: bool = False,
        max_points: int = 300_000,
    ) -> None:
        """Pre-compute all voxel arrays before the SimulationApp launches."""
        esdf_np = _to_numpy(self.esdf)
        raw_np = _to_numpy(self.raw)
        origin_np = np.asarray(_to_numpy(self.og.origin), dtype=np.float64)
        resolution = float(self.og.resolution)

        mask = np.abs(esdf_np) < esdf_band
        ijk = np.argwhere(mask)
        if len(ijk) > 0:
            n_total = len(ijk)
            ijk_s = _subsample(ijk, max_points, seed=42)
            centres = origin_np + (ijk_s.astype(np.float64) + 0.5) * resolution
            values = esdf_np[ijk_s[:, 0], ijk_s[:, 1], ijk_s[:, 2]]
            colors = esdf_to_rgb(values, vmin=-esdf_band, vmax=esdf_band * 0.5)
            band = _PreparedPoints(positions=centres, colors=colors, n_total=n_total)
        else:
            band = None

        occupied = None
        if show_occupied:
            occ_ijk = np.argwhere(raw_np)
            n_total = len(occ_ijk)
            occ_ijk = _subsample(occ_ijk, max_points, seed=42)
            pts = origin_np + (occ_ijk.astype(np.float64) + 0.5) * resolution
            occupied = _PreparedPoints(positions=pts, colors=OCCUPIED_COLOR, n_total=n_total)

        inflated = None
        if show_inflated:
            grid_np = _to_numpy(self.og.grid)
            shell = grid_np & ~raw_np
            shell_ijk = np.argwhere(shell)
            n_total = len(shell_ijk)
            shell_ijk = _subsample(shell_ijk, max_points, seed=99)
            pts = origin_np + (shell_ijk.astype(np.float64) + 0.5) * resolution
            inflated = _PreparedPoints(positions=pts, colors=INFLATED_COLOR, n_total=n_total)

        self._prepared = _PreparedScene(
            band=band, occupied=occupied, inflated=inflated, esdf_band=esdf_band
        )

    def add_3d(
        self,
        stage,
        base_path: str,
        show_mesh: bool = True,
    ) -> list[str]:
        if self._prepared is None:
            raise RuntimeError("EsdfVisualizer.prepare() must be called before add_3d()")
        prep = self._prepared
        paths: list[str] = []

        if show_mesh and self.scaled_mesh is not None:
            paths.append(
                create_mesh_prim(stage, f"{base_path}/mesh", self.scaled_mesh, color=MESH_COLOR)
            )
            logger.info(
                "Mesh: %d verts, %d faces",
                len(self.scaled_mesh.vertices),
                len(self.scaled_mesh.faces),
            )

        if prep.band is not None:
            paths.append(
                create_points_prim(
                    stage, f"{base_path}/esdf_band", prep.band.positions, colors=prep.band.colors
                )
            )
            logger.info(
                "ESDF band |d| < %.1fm: %d points (of %d)",
                prep.esdf_band, len(prep.band.positions), prep.band.n_total,
            )
        else:
            logger.warning("No voxels fall within ESDF band")

        if prep.occupied is not None:
            paths.append(
                create_points_prim(
                    stage, f"{base_path}/occupied",
                    prep.occupied.positions, colors=prep.occupied.colors,
                )
            )
            logger.info(
                "Occupied voxels: %d points (of %d)",
                len(prep.occupied.positions), prep.occupied.n_total,
            )

        if prep.inflated is not None:
            paths.append(
                create_points_prim(
                    stage, f"{base_path}/inflated",
                    prep.inflated.positions, colors=prep.inflated.colors,
                )
            )
            logger.info(
                "Inflation shell: %d points (of %d)",
                len(prep.inflated.positions), prep.inflated.n_total,
            )

        paths.append(
            create_coordinate_frame_prim(stage, f"{base_path}/coord_frame", size=COORD_FRAME_SIZE)
        )

        return paths
