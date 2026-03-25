"""Occupancy grid construction and SDF precomputation for viewpoint sampling."""

import logging
import time
from math import ceil

import numpy as np
import open3d as o3d
import trimesh

logger = logging.getLogger(__name__)


def build_occupancy_grid(mesh: o3d.geometry.TriangleMesh,
                         frustum_far: float, min_clearance: float,
                         resolution: float = 0.10):
    """Build a surface-only OccupancyGrid from an Open3D mesh.

    Args:
        mesh:          Open3D triangle mesh.
        frustum_far:   camera frustum far plane distance.
        min_clearance:  minimum clearance from mesh surface.
        resolution:    voxel resolution in meters.

    Returns:
        OccupancyGrid with inflated, raw, and filled grids.
    """
    from shared.occupancy_grid import OccupancyGrid, inflate_grid

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    tm = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Grid bounds: extend by frustum_far + min_clearance so the SDF
    # covers the full sampling shell around the mesh.
    padding = frustum_far + min_clearance
    bounds_min = vertices.min(axis=0) - padding
    bounds_max = vertices.max(axis=0) + padding
    origin = bounds_min.copy()
    grid_shape = np.ceil((bounds_max - bounds_min) / resolution).astype(int)
    grid_shape = np.maximum(grid_shape, 1)

    # Surface-only voxelization
    vg_surface = tm.voxelized(pitch=resolution)
    vg_filled = vg_surface.fill()

    def _map_voxels_to_grid(vg):
        """Map a trimesh VoxelGrid into the padded grid."""
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

    raw_grid = _map_voxels_to_grid(vg_surface)
    filled_raw_grid = _map_voxels_to_grid(vg_filled)

    logger.info("[ViewpointSampler] Auto-built OG: shape=%s, surface=%d voxels, "
                "filled=%d voxels, res=%.3fm",
                grid_shape, int(raw_grid.sum()), int(filled_raw_grid.sum()),
                resolution)

    # Inflate by min_clearance so free-space voxels are guaranteed
    # to have at least min_clearance distance from the mesh surface.
    inflation_voxels = max(1, ceil(min_clearance / resolution))
    inflated = inflate_grid(raw_grid, inflation_voxels)
    logger.info("[ViewpointSampler] Inflated OG: %d occupied, %d free (inflation=%d voxels)",
                int(inflated.sum()), int((~inflated).sum()), inflation_voxels)

    return OccupancyGrid(
        grid=inflated,
        origin=origin,
        resolution=resolution,
        raw_grid=raw_grid,
        filled_raw_grid=filled_raw_grid,
    )


def precompute_sdf_grid(occupancy_grid) -> np.ndarray:
    """Build a volumetric SDF grid using the two-EDT method.

    Uses ``og.filled_raw_grid`` (trimesh ray-based flood fill) for robust
    interior detection.  Convention: positive outside, negative inside.

    Returns:
        float32 numpy array with signed distance values.
    """
    from scipy.ndimage import distance_transform_edt

    og = occupancy_grid

    # Prefer filled_raw_grid (from trimesh vg.fill()); fall back to raw_grid
    if og.filled_raw_grid is not None:
        filled = og.filled_raw_grid
    elif og.raw_grid is not None:
        logger.warning("[ViewpointSampler] filled_raw_grid not available — "
                       "falling back to raw_grid with binary_fill_holes "
                       "(less robust for non-watertight meshes).")
        from scipy.ndimage import binary_fill_holes
        filled = binary_fill_holes(og.raw_grid)
    else:
        logger.warning("[ViewpointSampler] No raw grid available — "
                       "falling back to inflated grid for SDF.")
        filled = og.grid

    t0 = time.perf_counter()
    outside_dist = distance_transform_edt(~filled).astype(np.float32) * og.resolution
    inside_dist = distance_transform_edt(filled).astype(np.float32) * og.resolution
    sdf_grid = outside_dist - inside_dist
    dt = time.perf_counter() - t0
    logger.info("[ViewpointSampler] SDF grid computed (two-EDT): shape=%s, "
                "%.2fs, range=[%.2f, %.2f]m",
                sdf_grid.shape, dt,
                float(sdf_grid.min()), float(sdf_grid.max()))

    return sdf_grid
