"""
VRP Planner – 3D Occupancy Grid

Builds an inflated voxel occupancy grid from the environment mesh
(Duke of Lancaster shipwreck + static cuboid obstacles) using trimesh.

The grid is the central data structure shared by:
  • gpu_distance_matrix.py  (cuGraph / A* cost-matrix input)
  • traffic_light.py        (path corridor overlap checks)
  • trajectory_planner.py   (OMPL RRT* obstacle input)

All coordinates below use the **world** frame (metres).
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .config import (
    INFLATION_VOXELS,
    MESH_PATH,
    MESH_POSE,
    MESH_TARGET_LENGTH,
    ROBOT_RADIUS,
    STATIC_OBSTACLES,
    VOXEL_RESOLUTION,
)

# ── Re-export core types from shared ──────────────────────────────────────────
from shared.occupancy_grid import (
    OccupancyGrid,
    inflate_grid,
    downsample_occupancy_grid,
)

# Keep the old private name as an alias for backward compatibility
_inflate_grid = inflate_grid


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
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(bounds_min, bounds_max)`` of the environment mesh in world frame.

    Applies the same uniform scale (``MESH_TARGET_LENGTH``) and pose
    (``MESH_POSE``) that ``build_occupancy_grid`` uses, but skips
    voxelisation.  Use this to determine depot / start positions *before*
    the grid is built.

    Returns
    -------
    bounds_min, bounds_max : np.ndarray (3,)
        World-frame axis-aligned bounding box corners after scale + pose.

    Raises
    ------
    FileNotFoundError
        If ``mesh_path`` does not exist.
    """
    import trimesh
    from scipy.spatial.transform import Rotation as R

    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"[get_mesh_world_bounds] Mesh not found: {mesh_path}")

    scene_or_mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(scene_or_mesh.geometry.values()))
    else:
        mesh = scene_or_mesh

    longest = float(mesh.extents.max())
    if longest > 0:
        mesh.apply_scale(MESH_TARGET_LENGTH / longest)

    T_pose = np.eye(4)
    T_pose[:3, 3] = MESH_POSE[:3]
    quat_wxyz = MESH_POSE[3:7]
    rot = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    T_pose[:3, :3] = rot.as_matrix()
    mesh.apply_transform(T_pose)

    return np.asarray(mesh.bounds[0], dtype=float), np.asarray(mesh.bounds[1], dtype=float)


def build_occupancy_grid(
    mesh_path: str = MESH_PATH,
    resolution: float = VOXEL_RESOLUTION,
    inflation_voxels: int = INFLATION_VOXELS,
    padding: float = 1.0,
    extra_free_points: Optional[np.ndarray] = None,
    cache_path: Optional[str] = None,
    force_rebuild: bool = False,
    fill_interior: bool = True,
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
    import trimesh

    # ── Cache hit ─────────────────────────────────────────────────────────────
    if cache_path and not force_rebuild and os.path.exists(cache_path):
        print(f"[OccupancyGrid] Loading cached grid from {cache_path}")
        return OccupancyGrid.load(cache_path)

    fill_msg = "filled" if fill_interior else "surface-only"
    print(f"[OccupancyGrid] Building grid from {mesh_path} "
          f"(res={resolution}m, inflation={inflation_voxels}vox, {fill_msg}) …")

    # ── 1. Load mesh ──────────────────────────────────────────────────────────
    mesh_scale_factor: Optional[float] = None
    if os.path.exists(mesh_path):
        scene_or_mesh = trimesh.load(mesh_path, force="mesh")
        if isinstance(scene_or_mesh, trimesh.Scene):
            # Merge all scene geometries into a single mesh
            mesh = trimesh.util.concatenate(
                [g for g in scene_or_mesh.geometry.values()]
            )
        else:
            mesh = scene_or_mesh
        print(f"[OccupancyGrid] Mesh loaded: {len(mesh.vertices)} verts, "
              f"{len(mesh.faces)} faces")

        # ── 1a. Uniform scale so longest axis == MESH_TARGET_LENGTH ───────
        longest = float(mesh.extents.max())
        if longest > 0:
            mesh_scale_factor = MESH_TARGET_LENGTH / longest
            mesh.apply_scale(mesh_scale_factor)
            print(f"[OccupancyGrid] Mesh scaled ×{mesh_scale_factor:.4f}  "
                  f"extents now: {mesh.extents.tolist()}")

        # ── 1b. Apply MESH_POSE so occupancy matches the visual scene ─────
        T_pose = np.eye(4)
        T_pose[:3, 3] = MESH_POSE[:3]               # translation
        # Convert quaternion [qw, qx, qy, qz] → rotation matrix via trimesh
        quat_wxyz = MESH_POSE[3:7]
        from scipy.spatial.transform import Rotation as R
        rot = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])  # scipy expects xyzw
        T_pose[:3, :3] = rot.as_matrix()
        mesh.apply_transform(T_pose)
        print(f"[OccupancyGrid] Mesh pose applied: {MESH_POSE}  "
              f"bounds: {mesh.bounds.tolist()}")

        mesh_available = True
    else:
        print(f"[OccupancyGrid] WARNING: mesh not found at {mesh_path}. "
              f"Using obstacle-only grid.")
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
        margin = resolution  # one voxel clearance above/below each point
        pts_min = pts.min(axis=0) - margin
        pts_max = pts.max(axis=0) + margin
        orig_max_z = bounds_max[2]
        bounds_min = np.minimum(bounds_min, pts_min)
        bounds_max = np.maximum(bounds_max, pts_max)
        print(f"[OccupancyGrid] Extended bounds to cover {len(pts)} depot point(s).  "
              f"Z: {orig_max_z:.2f} → {bounds_max[2]:.2f} m")

    origin = bounds_min.copy()
    grid_shape = np.ceil((bounds_max - bounds_min) / resolution).astype(int)
    grid_shape = np.maximum(grid_shape, 1)      # guard against zero-size
    print(f"[OccupancyGrid] Grid shape: {tuple(grid_shape)}  "
          f"({np.prod(grid_shape)/1e6:.1f} M voxels)")

    raw_grid = np.zeros(grid_shape, dtype=bool)

    # ── 3. Voxelise mesh ──────────────────────────────────────────────────────
    filled_raw_grid = None
    if mesh_available:
        # trimesh voxel grid pitch = resolution
        vg_surface = mesh.voxelized(pitch=resolution)
        vg_filled = vg_surface.fill()

        if fill_interior:
            # Fill the interior so the hull is a solid obstacle, not just a
            # thin surface shell.  This is critical for both A* pathfinding
            # and the ESDF (otherwise the SDF is near-zero everywhere and
            # cuRobo plans trajectories through the hull).
            vg = vg_filled
        else:
            vg = vg_surface

        # vg.matrix is a dense bool array; indices are relative to vg.origin
        vox_matrix = vg.matrix
        fill_label = "Filled" if fill_interior else "Surface-only"
        print(f"[OccupancyGrid] {fill_label} voxelization: "
              f"{int(vox_matrix.sum())} voxels occupied "
              f"(shape {vox_matrix.shape})")
        # trimesh stores the voxel grid origin in the transform matrix
        vox_world_origin = np.asarray(vg.transform[:3, 3])
        vox_origin_ijk = np.floor(
            (vox_world_origin - origin) / resolution
        ).astype(int)
        # Copy into raw_grid at the correct offset
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
        print(f"[OccupancyGrid] Mesh voxels occupied: {int(raw_grid.sum())}")

        # Also store the filled voxelization for two-EDT SDF computation.
        # When fill_interior=True, this is the same as raw_grid.
        # When fill_interior=False, this gives us the filled interior needed
        # for robust SDF sign determination.
        filled_raw_grid = np.zeros(grid_shape, dtype=bool)
        vg_f = vg_filled
        vox_world_origin_f = np.asarray(vg_f.transform[:3, 3])
        vox_origin_ijk_f = np.floor(
            (vox_world_origin_f - origin) / resolution
        ).astype(int)
        vox_matrix_f = vg_f.matrix
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
        print(f"[OccupancyGrid] Filled voxelization stored: "
              f"{int(filled_raw_grid.sum())} voxels")

    # ── 4. Mark static cuboid obstacles ───────────────────────────────────────
    _add_cuboid_obstacles(raw_grid, origin, resolution, STATIC_OBSTACLES)
    print(f"[OccupancyGrid] After cuboids: {int(raw_grid.sum())} occupied voxels")

    # ── 5. Preserve raw grid before inflation (for ESDF) ────────────────────
    raw_grid_copy = raw_grid.copy()

    # ── 6. Inflate by robot radius ───────────────────────────────────────────
    if inflation_voxels > 0:
        inflated_grid = inflate_grid(raw_grid, inflation_voxels)
    else:
        inflated_grid = raw_grid
    print(f"[OccupancyGrid] After inflation: {int(inflated_grid.sum())} occupied  "
          f"({int((~inflated_grid).sum())} free)")

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
        print(f"[OccupancyGrid] Saved to {cache_path}")

    return og
