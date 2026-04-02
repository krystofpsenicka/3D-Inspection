"""
VRP Planner – 3D Occupancy Grid
================================

Re-exports the shared ``OccupancyGrid`` and grid utilities,
plus VRP-specific helpers (mesh world bounds, VRP grid builder).
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import numpy as np

from .constants import (
    INFLATION_VOXELS,
    MESH_PATH,
    MESH_POSE,
    MESH_TARGET_LENGTH,
    ROBOT_RADIUS,
    VOXEL_RESOLUTION,
)

logger = logging.getLogger(__name__)

# ── Re-export core types from shared ──────────────────────────────────────────
from shared.occupancy_grid import OccupancyGrid
from shared.grid_utils import inflate_grid, downsample_occupancy_grid
from shared.grid_builder_utils import (
    build_occupancy_grid as _shared_build_occupancy_grid,
    compute_grid_bounds,
    voxelize_mesh,
)


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


def build_occupancy_grid(
    mesh_path: str = MESH_PATH,
    resolution: float = VOXEL_RESOLUTION,
    inflation_voxels: int = INFLATION_VOXELS,
    padding: float = 1.0,
    extra_free_points: Optional[np.ndarray] = None,
    fill_interior: bool = True,
    mesh_target_length: float = MESH_TARGET_LENGTH,
    mesh_pose: list = None,
) -> OccupancyGrid:
    """Build a 3D occupancy grid from the environment mesh.

    VRP-specific wrapper: loads the mesh from disk, then delegates to the
    shared builder in ``shared.grid_builder_utils``.

    Parameters
    ----------
    mesh_path
        Path to the ``.glb`` / ``.obj`` / ``.stl`` environment mesh.
    resolution
        Voxel edge length in metres.
    inflation_voxels
        Dilation radius in voxels.
    padding
        Extra space (metres) added around the mesh bounding box.
    extra_free_points
        Optional ``(N, 3)`` array of world-frame positions that must lie
        inside the grid as free voxels.
    fill_interior : bool
        If ``True`` (default), fill the interior of the mesh.
    mesh_target_length
        Target length (m) of the mesh along its longest axis.
    mesh_pose
        ``[x, y, z, qw, qx, qy, qz]`` pose.

    Returns
    -------
    OccupancyGrid
    """
    if mesh_pose is None:
        mesh_pose = MESH_POSE

    logger.info(
        "[build_occupancy_grid] Building grid from %s (res=%sm, inflation=%svox) …",
        mesh_path, resolution, inflation_voxels,
    )

    from shared.mesh_loader import load_and_transform_mesh
    mesh = load_and_transform_mesh(mesh_path, mesh_target_length, mesh_pose)

    # Margin for extra free points: at least 3 voxels or ceil(robot_radius / resolution)
    extra_margin = max(3, int(np.ceil(ROBOT_RADIUS / resolution))) if extra_free_points is not None else 0

    return _shared_build_occupancy_grid(
        mesh=mesh,
        padding=padding,
        inflation_voxels=inflation_voxels,
        resolution=resolution,
        fill_interior=fill_interior,
        extra_free_points=extra_free_points,
        extra_margin_voxels=extra_margin,
    )
