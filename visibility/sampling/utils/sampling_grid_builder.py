"""Sampling occupancy grid builder and SDF precomputation."""

import logging
import time
from math import ceil

import numpy as np
import open3d as o3d
import trimesh

from shared.grid_builder_utils import compute_grid_bounds, voxelize_mesh
from shared.grid_utils import inflate_grid
from .sampling_occupancy_grid import SamplingOccupancyGrid

logger = logging.getLogger(__name__)


def build_sampling_occupancy_grid(
    mesh: o3d.geometry.TriangleMesh,
    frustum_far: float,
    min_clearance: float,
    resolution: float = 0.10,
) -> SamplingOccupancyGrid:
    """Build a surface-only SamplingOccupancyGrid from an Open3D mesh.

    Args:
        mesh:          Open3D triangle mesh.
        frustum_far:   camera frustum far plane distance.
        min_clearance:  minimum clearance from mesh surface.
        resolution:    voxel resolution in meters.

    Returns:
        SamplingOccupancyGrid with inflated, raw, and filled grids.
    """
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    tm = trimesh.Trimesh(vertices=vertices, faces=faces)

    padding = frustum_far + min_clearance
    origin, grid_shape = compute_grid_bounds(
        vertices.min(axis=0), vertices.max(axis=0),
        padding, resolution,
    )

    raw_grid, filled_raw_grid = voxelize_mesh(
        tm, grid_shape, origin, resolution, fill_interior=False,
    )

    logger.info(
        "[build_sampling_occupancy_grid] Built OG: shape=%s, surface=%d voxels, "
        "filled=%d voxels, res=%.3fm",
        tuple(grid_shape), int(raw_grid.sum()), int(filled_raw_grid.sum()),
        resolution,
    )

    inflation_voxels = max(1, ceil(min_clearance / resolution))
    inflated = inflate_grid(raw_grid, inflation_voxels)
    logger.info(
        "[build_sampling_occupancy_grid] Inflated OG: %d occupied, %d free "
        "(inflation=%d voxels)",
        int(inflated.sum()), int((~inflated).sum()), inflation_voxels,
    )

    return SamplingOccupancyGrid(
        grid=inflated,
        origin=origin,
        resolution=resolution,
        raw_grid=raw_grid,
        filled_raw_grid=filled_raw_grid,
    )


def build_sdf_grid(og: SamplingOccupancyGrid) -> np.ndarray:
    """Build a volumetric SDF grid using the two-EDT method.

    Uses ``og.filled_raw_grid`` for robust interior detection.
    Positive outside, negative inside.

    Returns:
        float32 numpy array with signed distance values.
    """
    from scipy.ndimage import distance_transform_edt

    t0 = time.perf_counter()
    outside_dist = distance_transform_edt(~og.filled_raw_grid).astype(np.float32) * og.resolution
    inside_dist = distance_transform_edt(og.filled_raw_grid).astype(np.float32) * og.resolution
    sdf_grid = outside_dist - inside_dist
    dt = time.perf_counter() - t0
    logger.info(
        "[build_sdf_grid] SDF grid computed (two-EDT): shape=%s, "
        "%.2fs, range=[%.2f, %.2f]m",
        sdf_grid.shape, dt,
        float(sdf_grid.min()), float(sdf_grid.max()),
    )
    return sdf_grid
