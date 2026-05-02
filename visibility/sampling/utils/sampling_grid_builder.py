"""Sampling occupancy grid builder and SDF precomputation."""

import logging
import time
from math import ceil

import cupy as cp
import numpy as np
import open3d as o3d
import trimesh

from shared.grid_builder_utils import compute_grid_bounds, voxelize_mesh
from shared.grid_utils import inflate_grid
from shared.occupancy_grid import OccupancyGrid

logger = logging.getLogger(__name__)


def build_sampling_occupancy_grid(
    mesh: o3d.geometry.TriangleMesh,
    frustum_far: float,
    min_clearance: float,
    resolution: float = 0.10,
) -> tuple[OccupancyGrid, cp.ndarray, cp.ndarray]:
    """Build a surface-only inflated OccupancyGrid for sampling/collision.

    Returns
    -------
    (og, raw_grid, filled_grid)
        *og* has the inflated grid for collision checks.
        *raw_grid* is the surface-only voxelization (pre-inflation).
        *filled_grid* is the interior-filled voxelization (for SDF signs).
    """
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    tm = trimesh.Trimesh(vertices=vertices, faces=faces)

    padding = frustum_far + min_clearance
    origin, grid_shape = compute_grid_bounds(
        vertices.min(axis=0),
        vertices.max(axis=0),
        padding,
        resolution,
    )

    raw_grid, filled_grid = voxelize_mesh(
        tm,
        grid_shape,
        origin,
        resolution,
        fill_interior=False,
    )

    inflation_voxels = max(1, ceil(min_clearance / resolution))
    inflated = inflate_grid(raw_grid, inflation_voxels)
    logger.info(
        "[build_sampling_occupancy_grid] Built OG: shape=%s, surface=%d voxels, "
        "occupied=%d, free=%d, res=%.3fm (inflation=%d voxels)",
        tuple(grid_shape),
        int(raw_grid.sum()),
        int(inflated.sum()),
        int((~inflated).sum()),
        resolution,
        inflation_voxels,
    )

    og = OccupancyGrid(grid=inflated, origin=origin, resolution=resolution)
    return og, raw_grid, filled_grid


def build_sdf_grid(
    mesh: o3d.geometry.TriangleMesh,
    og: OccupancyGrid,
    filled_grid: cp.ndarray | None = None,
) -> cp.ndarray:
    """Build a volumetric SDF grid for *og* via the two-EDT method.

    If *filled_grid* (interior-filled voxelization on *og*'s grid) is
    provided, it is used directly.  Otherwise the mesh is re-voxelized to
    obtain it.  Positive outside, negative inside.  scipy
    distance_transform_edt requires CPU -- transfer at the boundary.
    """
    from scipy.ndimage import distance_transform_edt

    if filled_grid is None:
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        tm = trimesh.Trimesh(vertices=vertices, faces=faces)
        grid_shape = np.array(og.grid.shape)
        _, filled_grid = voxelize_mesh(
            tm,
            grid_shape,
            og.origin,
            og.resolution,
            fill_interior=False,
        )

    t0 = time.perf_counter()
    filled_np = cp.asnumpy(filled_grid)
    outside_dist = distance_transform_edt(~filled_np).astype(np.float32) * og.resolution
    inside_dist = distance_transform_edt(filled_np).astype(np.float32) * og.resolution
    sdf_grid = cp.asarray(outside_dist - inside_dist)
    dt = time.perf_counter() - t0
    logger.info(
        "[build_sdf_grid] SDF grid computed (two-EDT): shape=%s, %.2fs, range=[%.2f, %.2f]m",
        sdf_grid.shape,
        dt,
        float(sdf_grid.min()),
        float(sdf_grid.max()),
    )
    return sdf_grid
