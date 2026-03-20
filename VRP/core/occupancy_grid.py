"""
VRP Planner – 3D Occupancy Grid

Builds an inflated voxel occupancy grid from the environment mesh
(Duke of Lancaster shipwreck + static cuboid obstacles) using trimesh.

The grid is the central data structure shared by:
  • gpu_distance_matrix.py  (cuGraph / A* cost-matrix input)
  • traffic_light.py        (path corridor overlap checks)
All coordinates below use the **world** frame (metres).
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ..config import (
    INFLATION_VOXELS,
    MESH_PATH,
    MESH_POSE,
    MESH_TARGET_LENGTH,
    ROBOT_RADIUS,
    STATIC_OBSTACLES,
    VOXEL_RESOLUTION,
)

import logging

logger = logging.getLogger(__name__)

# ── Re-export core types from shared ──────────────────────────────────────────
from shared.occupancy_grid import (
    OccupancyGrid,
    inflate_grid,
    downsample_occupancy_grid,
)


# ── Grid builder helpers ─────────────────────────────────────────────────────

def _add_cuboid_obstacles(
    grid: np.ndarray,
    origin: np.ndarray,
    resolution: float,
    obstacles: dict,
) -> None:
    """Mark cuboid obstacle voxels in-place."""
    shape = np.array(grid.shape)
    for name, obs in obstacles.items():
        dims = np.asarray(obs["dims"], dtype=float)
        pose = obs["pose"]
        centre = np.array(pose[:3], dtype=float)
        half   = dims / 2.0
        min_w  = centre - half
        max_w  = centre + half
        ijk_min = np.floor((min_w - origin) / resolution).astype(int)
        ijk_max = np.ceil((max_w - origin) / resolution).astype(int)
        ijk_min = np.clip(ijk_min, 0, shape - 1)
        ijk_max = np.clip(ijk_max, 0, shape - 1)
        grid[
            ijk_min[0]: ijk_max[0] + 1,
            ijk_min[1]: ijk_max[1] + 1,
            ijk_min[2]: ijk_max[2] + 1,
        ] = True


def get_mesh_world_bounds(
    mesh_path: str = MESH_PATH,
    mesh_target_length: float = MESH_TARGET_LENGTH,
    mesh_pose: list = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(bounds_min, bounds_max)`` of the environment mesh in world frame.

    Applies the same uniform scale and pose that ``build_occupancy_grid``
    uses, but skips voxelisation.  Use this to determine depot / start
    positions *before* the grid is built.

    Returns
    -------
    bounds_min, bounds_max : np.ndarray (3,)
        World-frame axis-aligned bounding box corners after scale + pose.
    """
    from shared.mesh_loader import load_and_transform_mesh

    if mesh_pose is None:
        mesh_pose = MESH_POSE
    mesh = load_and_transform_mesh(mesh_path, mesh_target_length, mesh_pose)
    return np.asarray(mesh.bounds[0], dtype=float), np.asarray(mesh.bounds[1], dtype=float)


def _voxelize_mesh(
    mesh,
    grid_shape: np.ndarray,
    origin: np.ndarray,
    resolution: float,
    fill_interior: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Voxelize a trimesh mesh into a raw occupancy grid.

    Returns ``(raw_grid, filled_raw_grid)`` where ``filled_raw_grid`` is
    always the flood-filled version (for two-EDT SDF), regardless of
    ``fill_interior``.
    """
    vg_surface = mesh.voxelized(pitch=resolution)
    vg_filled = vg_surface.fill()

    vg = vg_filled if fill_interior else vg_surface

    vox_matrix = vg.matrix
    fill_label = "Filled" if fill_interior else "Surface-only"
    logger.info(
        "[OccupancyGrid] %s voxelization: %s voxels occupied (shape %s)",
        fill_label, int(vox_matrix.sum()), vox_matrix.shape,
    )

    raw_grid = np.zeros(grid_shape, dtype=bool)

    vox_world_origin = np.asarray(vg.transform[:3, 3])
    vox_origin_ijk = np.floor(
        (vox_world_origin - origin) / resolution
    ).astype(int)
    dst_min = np.maximum(vox_origin_ijk, 0)
    src_min = np.maximum(-vox_origin_ijk, 0)
    dst_max = np.minimum(vox_origin_ijk + np.array(vox_matrix.shape), grid_shape)
    src_max = src_min + (dst_max - dst_min)
    raw_grid[
        dst_min[0]: dst_max[0],
        dst_min[1]: dst_max[1],
        dst_min[2]: dst_max[2],
    ] = vox_matrix[
        src_min[0]: src_max[0],
        src_min[1]: src_max[1],
        src_min[2]: src_max[2],
    ]
    logger.info("[OccupancyGrid] Mesh voxels occupied: %s", int(raw_grid.sum()))

    # Store the filled voxelization for two-EDT SDF computation.
    filled_raw_grid = np.zeros(grid_shape, dtype=bool)
    vox_world_origin_f = np.asarray(vg_filled.transform[:3, 3])
    vox_origin_ijk_f = np.floor(
        (vox_world_origin_f - origin) / resolution
    ).astype(int)
    vox_matrix_f = vg_filled.matrix
    dst_min_f = np.maximum(vox_origin_ijk_f, 0)
    src_min_f = np.maximum(-vox_origin_ijk_f, 0)
    dst_max_f = np.minimum(vox_origin_ijk_f + np.array(vox_matrix_f.shape), grid_shape)
    src_max_f = src_min_f + (dst_max_f - dst_min_f)
    filled_raw_grid[
        dst_min_f[0]: dst_max_f[0],
        dst_min_f[1]: dst_max_f[1],
        dst_min_f[2]: dst_max_f[2],
    ] = vox_matrix_f[
        src_min_f[0]: src_max_f[0],
        src_min_f[1]: src_max_f[1],
        src_min_f[2]: src_max_f[2],
    ]
    logger.info(
        "[OccupancyGrid] Filled voxelization stored: %s voxels",
        int(filled_raw_grid.sum()),
    )

    return raw_grid, filled_raw_grid


def build_occupancy_grid(
    mesh_path: str = MESH_PATH,
    resolution: float = VOXEL_RESOLUTION,
    inflation_voxels: int = INFLATION_VOXELS,
    padding: float = 1.0,
    extra_free_points: Optional[np.ndarray] = None,
    cache_path: Optional[str] = None,
    force_rebuild: bool = False,
    fill_interior: bool = True,
    mesh_target_length: float = MESH_TARGET_LENGTH,
    mesh_pose: list = None,
) -> OccupancyGrid:
    """Build a 3D occupancy grid from the environment mesh + static obstacles.

    Pipeline (follows PDF spec):
    1. Load mesh with trimesh.
    2. Voxelize at ``resolution`` using trimesh's ``VoxelGrid``.
    3. Mark static cuboid obstacles (rocks, coral).
    4. Inflate the raw grid by ``inflation_voxels`` (robot radius / resolution).
    5. Wrap in an ``OccupancyGrid`` object.

    Parameters
    ----------
    mesh_path
        Path to the ``.glb`` / ``.obj`` / ``.stl`` environment mesh.
    resolution
        Voxel edge length in metres.
    inflation_voxels
        Dilation radius in voxels (should equal ceil(robot_radius / resolution)).
    padding
        Extra space (metres) added around the mesh bounding box.
    extra_free_points
        Optional ``(N, 3)`` array of world-frame positions that must lie
        inside the grid as **free** voxels (e.g. robot depot positions
        above the mesh).  The grid bounds are extended to cover them with
        one voxel of margin before voxelisation.
    cache_path
        If given, load from disk when available; save after building.
    force_rebuild
        Ignore any cached grid and rebuild from scratch.
    fill_interior : bool
        If ``True`` (default), fill the interior of the mesh so the hull is
        a solid obstacle.  Set to ``False`` for a surface-only occupancy grid
        (used by viewpoint sampling to allow free space on both sides of the
        mesh shell).

    Returns
    -------
    OccupancyGrid
    """
    # ── Cache hit ─────────────────────────────────────────────────────────────
    if cache_path and not force_rebuild and os.path.exists(cache_path):
        logger.info("[OccupancyGrid] Loading cached grid from %s", cache_path)
        return OccupancyGrid.load(cache_path)

    if mesh_pose is None:
        mesh_pose = MESH_POSE

    fill_msg = "filled" if fill_interior else "surface-only"
    logger.info(
        "[OccupancyGrid] Building grid from %s (res=%sm, inflation=%svox, %s) …",
        mesh_path, resolution, inflation_voxels, fill_msg,
    )

    # ── 1. Load mesh ──────────────────────────────────────────────────────────
    mesh_scale_factor: Optional[float] = None
    if os.path.exists(mesh_path):
        from shared.mesh_loader import load_and_transform_mesh
        mesh = load_and_transform_mesh(mesh_path, mesh_target_length, mesh_pose)
        mesh_scale_factor = mesh.metadata.get("scale_factor")
        logger.info(
            "[OccupancyGrid] Mesh loaded: %s verts, %s faces, scale=%s",
            len(mesh.vertices), len(mesh.faces), mesh_scale_factor,
        )
        logger.info("[OccupancyGrid] Mesh bounds: %s", mesh.bounds.tolist())
        mesh_available = True
    else:
        logger.warning(
            "[OccupancyGrid] WARNING: mesh not found at %s. Using obstacle-only grid.",
            mesh_path,
        )
        mesh_available = False

    # ── 2. Determine grid bounds ───────────────────────────────────────────────
    if mesh_available:
        bounds_min = mesh.bounds[0] - padding
        bounds_max = mesh.bounds[1] + padding
    else:
        # Fall back to a sensible default workspace
        bounds_min = np.array([-5.0, -5.0,  0.0])
        bounds_max = np.array([15.0, 15.0, 10.0])

    # Extend bounds to cover any extra free points (e.g. depots above the mesh)
    if extra_free_points is not None and len(extra_free_points) > 0:
        pts = np.asarray(extra_free_points, dtype=float)
        # Margin: at least 3 voxels or ceil(robot_radius / resolution) voxels,
        # ensuring waypoints remain free after inflation + coarse downsampling.
        margin_voxels = max(3, int(np.ceil(ROBOT_RADIUS / resolution)))
        margin = margin_voxels * resolution
        pts_min = pts.min(axis=0) - margin
        pts_max = pts.max(axis=0) + margin
        orig_max_z = bounds_max[2]
        bounds_min = np.minimum(bounds_min, pts_min)
        bounds_max = np.maximum(bounds_max, pts_max)
        logger.info(
            "[OccupancyGrid] Extended bounds to cover %s depot point(s). Z: %.2f → %.2f m",
            len(pts), orig_max_z, bounds_max[2],
        )

    origin = bounds_min.copy()
    grid_shape = np.ceil((bounds_max - bounds_min) / resolution).astype(int)
    grid_shape = np.maximum(grid_shape, 1)      # guard against zero-size
    logger.info(
        "[OccupancyGrid] Grid shape: %s  (%.1f M voxels)",
        tuple(grid_shape), np.prod(grid_shape) / 1e6,
    )

    # ── 3. Voxelise mesh ──────────────────────────────────────────────────────
    filled_raw_grid = None
    if mesh_available:
        raw_grid, filled_raw_grid = _voxelize_mesh(
            mesh, grid_shape, origin, resolution, fill_interior,
        )
    else:
        raw_grid = np.zeros(grid_shape, dtype=bool)

    # ── 4. Mark static cuboid obstacles ───────────────────────────────────────
    _add_cuboid_obstacles(raw_grid, origin, resolution, STATIC_OBSTACLES)
    logger.info("[OccupancyGrid] After cuboids: %s occupied voxels", int(raw_grid.sum()))

    # ── 5. Preserve raw grid before inflation (for ESDF) ────────────────────
    raw_grid_copy = raw_grid.copy()

    # ── 6. Inflate by robot radius ───────────────────────────────────────────
    if inflation_voxels > 0:
        inflated_grid = inflate_grid(raw_grid, inflation_voxels)
    else:
        inflated_grid = raw_grid
    logger.info(
        "[OccupancyGrid] After inflation: %s occupied  (%s free)",
        int(inflated_grid.sum()), int((~inflated_grid).sum()),
    )

    # ── 5b. Preserve filled raw grid (for two-EDT SDF) ───────────────────
    # Add cuboid obstacles to filled_raw_grid too (they are solid obstacles)
    if filled_raw_grid is not None:
        _add_cuboid_obstacles(filled_raw_grid, origin, resolution, STATIC_OBSTACLES)
        filled_raw_copy = filled_raw_grid.copy()
    else:
        # No mesh — filled is same as raw
        filled_raw_copy = raw_grid_copy.copy()

    og = OccupancyGrid(
        grid=inflated_grid,
        origin=origin,
        resolution=resolution,
        raw_grid=raw_grid_copy,
        filled_raw_grid=filled_raw_copy,
        mesh_scale=mesh_scale_factor,
    )

    # ── Cache ─────────────────────────────────────────────────────────────────
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        og.save(cache_path)
        logger.info("[OccupancyGrid] Saved to %s", cache_path)

    return og
