"""
Grid Builder Utilities
==============================

Voxelization helpers and a simple occupancy-grid builder
used by both the VRP planner and the visibility/sampling.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import numpy as np

from .occupancy_grid import OccupancyGrid
from .grid_utils import inflate_grid

logger = logging.getLogger(__name__)


def _map_voxels_to_grid(
    vg: trimesh.voxel.VoxelGrid,
    grid_shape: np.ndarray,
    origin: np.ndarray,
    resolution: float,
) -> np.ndarray:
    """Map a trimesh VoxelGrid into a padded boolean grid.

    Parameters
    ----------
    vg : trimesh.voxel.VoxelGrid
        Trimesh voxelization result.
    grid_shape : array-like (3,)
        Target grid dimensions ``(Nx, Ny, Nz)``.
    origin : np.ndarray (3,)
        World position of voxel ``(0, 0, 0)`` in the target grid.
    resolution : float
        Voxel edge length.

    Returns
    -------
    np.ndarray, dtype=bool, shape ``grid_shape``
    """
    result = np.zeros(grid_shape, dtype=bool)
    vox_matrix = vg.matrix
    vox_world_origin = np.asarray(vg.transform[:3, 3])
    vox_origin_ijk = np.floor(
        (vox_world_origin - origin) / resolution
    ).astype(int)
    dst_min = np.maximum(vox_origin_ijk, 0)
    src_min = np.maximum(-vox_origin_ijk, 0)
    dst_max = np.minimum(vox_origin_ijk + np.array(vox_matrix.shape), grid_shape)
    src_max = src_min + (dst_max - dst_min)
    result[
        dst_min[0]:dst_max[0],
        dst_min[1]:dst_max[1],
        dst_min[2]:dst_max[2],
    ] = vox_matrix[
        src_min[0]:src_max[0],
        src_min[1]:src_max[1],
        src_min[2]:src_max[2],
    ]
    return result


def voxelize_mesh(
    mesh: trimesh.Trimesh,
    grid_shape: np.ndarray,
    origin: np.ndarray,
    resolution: float,
    fill_interior: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Voxelize a trimesh mesh into raw occupancy grids.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh to voxelize.
    grid_shape : array-like (3,)
        Target grid dimensions.
    origin : np.ndarray (3,)
        World position of voxel ``(0, 0, 0)``.
    resolution : float
        Voxel edge length (metres).
    fill_interior : bool
        If ``True``, the main grid uses the flood-filled voxelization.
        ``filled_raw_grid`` is always the flood-filled version (for
        two-EDT SDF computation).

    Returns
    -------
    (raw_grid, filled_raw_grid) : tuple of np.ndarray, dtype=bool
        ``raw_grid`` uses surface-only or filled depending on
        ``fill_interior``.  ``filled_raw_grid`` is always flood-filled.
    """
    vg_surface = mesh.voxelized(pitch=resolution)
    vg_filled = vg_surface.fill()

    vg = vg_filled if fill_interior else vg_surface
    fill_label = "Filled" if fill_interior else "Surface-only"
    logger.info(
        "[voxelize_mesh] %s voxelization: %s voxels occupied (shape %s)",
        fill_label, int(vg.matrix.sum()), vg.matrix.shape,
    )

    raw_grid = _map_voxels_to_grid(vg, grid_shape, origin, resolution)
    filled_raw_grid = _map_voxels_to_grid(vg_filled, grid_shape, origin, resolution)

    logger.info(
        "[voxelize_mesh] Mesh voxels occupied: %d (raw), %d (filled)",
        int(raw_grid.sum()), int(filled_raw_grid.sum()),
    )
    return raw_grid, filled_raw_grid


def compute_grid_bounds(
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    padding: float,
    resolution: float,
    extra_free_points: Optional[np.ndarray] = None,
    extra_margin_voxels: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute grid origin and shape from mesh bounds.

    Parameters
    ----------
    bounds_min, bounds_max : np.ndarray (3,)
        AABB of the mesh.
    padding : float
        Extra space added around the bounding box.
    resolution : float
        Voxel edge length.
    extra_free_points : np.ndarray | None
        Optional ``(N, 3)`` array of world-frame positions that must
        lie inside the grid.
    extra_margin_voxels : int
        Margin (in voxels) around ``extra_free_points``.

    Returns
    -------
    (origin, grid_shape) : tuple of np.ndarray
        ``origin`` is (3,) float, ``grid_shape`` is (3,) int.
    """
    bmin = np.asarray(bounds_min, dtype=float) - padding
    bmax = np.asarray(bounds_max, dtype=float) + padding

    if extra_free_points is not None and len(extra_free_points) > 0:
        pts = np.asarray(extra_free_points, dtype=float)
        margin = extra_margin_voxels * resolution
        pts_min = pts.min(axis=0) - margin
        pts_max = pts.max(axis=0) + margin
        bmin = np.minimum(bmin, pts_min)
        bmax = np.maximum(bmax, pts_max)
        logger.info(
            "[compute_grid_bounds] Extended bounds to cover %d extra point(s).",
            len(pts),
        )

    origin = bmin.copy()
    grid_shape = np.ceil((bmax - bmin) / resolution).astype(int)
    grid_shape = np.maximum(grid_shape, 1)
    logger.info(
        "[compute_grid_bounds] Grid shape: %s  (%.1f M voxels)",
        tuple(grid_shape), np.prod(grid_shape) / 1e6,
    )
    return origin, grid_shape


def build_occupancy_grid(
    mesh,
    padding: float,
    inflation_voxels: int,
    resolution: float,
    fill_interior: bool = False,
    extra_free_points: Optional[np.ndarray] = None,
    extra_margin_voxels: int = 0,
) -> OccupancyGrid:
    """Build a base OccupancyGrid from a trimesh mesh.

    This a generic builder that produces a base ``OccupancyGrid``
    (no raw/filled grids).

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The environment mesh.
    padding : float
        Extra space (metres) around the bounding box.
    inflation_voxels : int
        Dilation radius in voxels.
    resolution : float
        Voxel edge length (metres).
    fill_interior : bool
        Fill the mesh interior (solid obstacle) or surface-only.
    extra_free_points : np.ndarray | None
        Positions that must lie inside the grid as free voxels.
    extra_margin_voxels : int
        Margin (in voxels) around ``extra_free_points``.

    Returns
    -------
    OccupancyGrid
    """
    origin, grid_shape = compute_grid_bounds(
        mesh.bounds[0], mesh.bounds[1],
        padding, resolution,
        extra_free_points=extra_free_points,
        extra_margin_voxels=extra_margin_voxels,
    )

    raw_grid, _filled = voxelize_mesh(
        mesh, grid_shape, origin, resolution, fill_interior=fill_interior,
    )

    if inflation_voxels > 0:
        inflated = inflate_grid(raw_grid, inflation_voxels)
    else:
        inflated = raw_grid

    logger.info(
        "[build_occupancy_grid] After inflation: %d occupied  (%d free)",
        int(inflated.sum()), int((~inflated).sum()),
    )

    return OccupancyGrid(
        grid=inflated,
        origin=origin,
        resolution=resolution,
    )
