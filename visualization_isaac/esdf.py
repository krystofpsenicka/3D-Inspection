"""Visualize ESDF voxel grids in 3D (Isaac Sim variant — no 2D slice)."""

from __future__ import annotations

import logging

import numpy as np
import trimesh

from shared.grid_utils import esdf_to_rgb
from shared.occupancy_grid import OccupancyGrid
from ._usd_primitives import (
    create_mesh_prim,
    create_points_prim,
    create_coordinate_frame_prim,
)

logger = logging.getLogger(__name__)

MESH_COLOR = (0.6, 0.6, 0.6)
OCCUPIED_COLOR = (1.0, 0.7, 0.0)
INFLATED_COLOR = (0.0, 0.7, 0.3)
COORD_FRAME_SIZE = 2.0


class EsdfVisualizer:
    """Stage-builder for ESDF voxel grid visualization.

    Parameters
    ----------
    occupancy_grid : OccupancyGrid object.
    raw_grid : Pre-inflation boolean occupancy grid.
    esdf : 3-D float array of signed distances.
    scaled_mesh : Optional trimesh (already scaled/posed) for overlays.
    """

    def __init__(self, occupancy_grid: OccupancyGrid, raw_grid: np.ndarray,
                 esdf: np.ndarray, scaled_mesh: trimesh.Trimesh | None = None):
        self.og = occupancy_grid
        self.raw = raw_grid
        self.esdf = esdf
        self.scaled_mesh = scaled_mesh

    def add_3d(self, stage, base_path: str,
               esdf_band: float = 2.0,
               show_occupied: bool = False,
               show_inflated: bool = False,
               show_mesh: bool = True,
               max_points: int = 300_000) -> list[str]:
        """Add ESDF 3D visualization prims to *stage*.

        Returns
        -------
        List of created prim paths.
        """
        paths: list[str] = []

        # 1. Mesh (grey)
        if show_mesh and self.scaled_mesh is not None:
            paths.append(create_mesh_prim(
                stage, f"{base_path}/mesh", self.scaled_mesh,
                color=MESH_COLOR))
            logger.info("Mesh: %d verts, %d faces",
                        len(self.scaled_mesh.vertices), len(self.scaled_mesh.faces))

        # 2. Near-surface ESDF voxels coloured by distance
        mask = np.abs(self.esdf) < esdf_band
        ijk = np.argwhere(mask)
        if len(ijk) > 0:
            if len(ijk) > max_points:
                rng = np.random.RandomState(42)
                ijk = ijk[rng.choice(len(ijk), max_points, replace=False)]
            centres = (self.og.origin
                       + (ijk.astype(np.float64) + 0.5) * self.og.resolution)
            values = self.esdf[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
            colors = esdf_to_rgb(values, vmin=-esdf_band, vmax=esdf_band * 0.5)

            paths.append(create_points_prim(
                stage, f"{base_path}/esdf_band", centres, colors=colors))
            logger.info("ESDF band |d| < %.1fm: %d points", esdf_band, len(ijk))
        else:
            logger.warning("No voxels fall within ESDF band")

        # 3. Raw occupied voxels
        if show_occupied:
            occ_ijk = np.argwhere(self.raw)
            if len(occ_ijk) > max_points:
                rng = np.random.RandomState(42)
                occ_ijk = occ_ijk[rng.choice(len(occ_ijk), max_points, replace=False)]
            pts = (self.og.origin
                   + (occ_ijk.astype(np.float64) + 0.5) * self.og.resolution)
            paths.append(create_points_prim(
                stage, f"{base_path}/occupied", pts, colors=OCCUPIED_COLOR))
            logger.info("Occupied voxels: %d points", len(occ_ijk))

        # 4. Inflation shell
        if show_inflated:
            shell = self.og.grid & ~self.raw
            shell_ijk = np.argwhere(shell)
            if len(shell_ijk) > max_points:
                rng = np.random.RandomState(99)
                shell_ijk = shell_ijk[rng.choice(len(shell_ijk), max_points,
                                                  replace=False)]
            pts = (self.og.origin
                   + (shell_ijk.astype(np.float64) + 0.5) * self.og.resolution)
            paths.append(create_points_prim(
                stage, f"{base_path}/inflated", pts, colors=INFLATED_COLOR))
            logger.info("Inflation shell: %d points", len(shell_ijk))

        # 5. Coordinate frame
        paths.append(create_coordinate_frame_prim(
            stage, f"{base_path}/coord_frame", size=COORD_FRAME_SIZE))

        return paths
