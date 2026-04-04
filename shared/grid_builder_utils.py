"""
Grid Builder Utilities
==============================

Voxelization helpers and a simple occupancy-grid builder
used by both the VRP planner and the visibility/sampling.

Trimesh voxelization produces numpy arrays on CPU; the results are
converted to CuPy (GPU) at the boundary and stay on GPU from there.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import cupy as cp
import numpy as np
import trimesh

from .occupancy_grid import OccupancyGrid
from .grid_utils import inflate_grid

logger = logging.getLogger(__name__)


def _map_voxels_to_grid(
    vg: trimesh.voxel.VoxelGrid,
    grid_shape: np.ndarray,
    origin: cp.ndarray,
    resolution: float,
) -> cp.ndarray:
    """Map a trimesh VoxelGrid into a padded boolean grid (GPU).

    The voxel matrix from trimesh is uploaded to GPU; all index arithmetic
    is done with CuPy.

    Parameters
    ----------
    vg : trimesh.voxel.VoxelGrid
        Trimesh voxelization result.
    grid_shape : array-like (3,)
        Target grid dimensions ``(Nx, Ny, Nz)``.
    origin : cp.ndarray (3,)
        World position of voxel ``(0, 0, 0)``.
    resolution : float
        Voxel edge length.

    Returns
    -------
    cp.ndarray, dtype=bool, shape ``grid_shape``
    """
    result = cp.zeros(tuple(grid_shape), dtype=cp.bool_)
    vox_matrix = cp.asarray(vg.matrix)
    vox_world_origin = cp.asarray(vg.transform[:3, 3])
    vox_origin_ijk = cp.floor(
        (vox_world_origin - origin) / resolution
    ).astype(cp.int32)
    grid_shape_gpu = cp.asarray(grid_shape)
    vox_shape_gpu = cp.asarray(vox_matrix.shape)
    dst_min = cp.maximum(vox_origin_ijk, 0)
    src_min = cp.maximum(-vox_origin_ijk, 0)
    dst_max = cp.minimum(vox_origin_ijk + vox_shape_gpu, grid_shape_gpu)
    src_max = src_min + (dst_max - dst_min)
    # Extract as Python ints for slice indexing
    d0, d1, d2 = int(dst_min[0]), int(dst_min[1]), int(dst_min[2])
    D0, D1, D2 = int(dst_max[0]), int(dst_max[1]), int(dst_max[2])
    s0, s1, s2 = int(src_min[0]), int(src_min[1]), int(src_min[2])
    S0, S1, S2 = int(src_max[0]), int(src_max[1]), int(src_max[2])
    result[d0:D0, d1:D1, d2:D2] = vox_matrix[s0:S0, s1:S1, s2:S2]
    return result


def voxelize_mesh(
    mesh: trimesh.Trimesh,
    grid_shape: np.ndarray,
    origin: cp.ndarray,
    resolution: float,
    fill_interior: bool = False,
) -> Tuple[cp.ndarray, cp.ndarray]:
    """Voxelize a trimesh mesh into GPU-resident occupancy grids.

    Returns
    -------
    (raw_grid, filled_raw_grid) : tuple of cp.ndarray, dtype=bool
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
) -> Tuple[cp.ndarray, np.ndarray]:
    """Compute grid origin and shape from mesh bounds.

    Parameters
    ----------
    bounds_min, bounds_max : np.ndarray (3,)
        AABB of the mesh (numpy — from trimesh).
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
    (origin, grid_shape) : tuple
        ``origin`` is CuPy (3,) float64, ``grid_shape`` is numpy (3,) int.
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

    origin = cp.asarray(bmin.copy(), dtype=cp.float64)
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
    OccupancyGrid (GPU-resident)
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
