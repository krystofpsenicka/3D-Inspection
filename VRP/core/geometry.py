"""Geometric transforms between viewpoint and robot-body coordinate frames."""

from __future__ import annotations

import math

import cupy as cp
import numpy as np

from .constants import CAMERA_OFFSET_FORWARD, CAMERA_OFFSET_UP


def compute_start_grid(
    num_robots: int,
    mesh_bounds_min: np.ndarray,
    mesh_bounds_max: np.ndarray,
    z_above: float = 1.0,
    spacing: float = 1.5,
) -> list[np.ndarray]:
    """Return per-robot start positions above the mesh centre.

    The ``num_robots`` start positions are arranged in a rectangular grid
    centred on the XY centre of the mesh bounding box, ``z_above`` metres
    above the mesh top, with ``spacing`` metres between adjacent robots.

    Parameters
    ----------
    num_robots:
        Number of robots to place.
    mesh_bounds_min:
        World-frame lower bound of the scaled+posed mesh ``(3,)``.
    mesh_bounds_max:
        World-frame upper bound of the scaled+posed mesh ``(3,)``.
    z_above:
        Clearance above the mesh top (metres).
    spacing:
        Distance between adjacent robot grid cells (metres).

    Returns
    -------
    start_positions : List[np.ndarray]
        Per-robot ``(3,)`` start XYZ, indexed 0 ... num_robots-1.
    """
    cx = float((mesh_bounds_min[0] + mesh_bounds_max[0]) / 2.0)
    cy = float((mesh_bounds_min[1] + mesh_bounds_max[1]) / 2.0)
    z = float(mesh_bounds_max[2] + z_above)

    cols = math.ceil(math.sqrt(num_robots))
    rows = math.ceil(num_robots / cols)

    start_positions: list[np.ndarray] = []
    for idx in range(num_robots):
        r = idx // cols
        c = idx % cols
        x = cx + (c - (cols - 1) / 2.0) * spacing
        y = cy + (r - (rows - 1) / 2.0) * spacing
        start_positions.append(np.array([x, y, z], dtype=np.float32))

    return start_positions


def viewpoints_to_robot_waypoints(
    positions: cp.ndarray,
    rotmats: cp.ndarray,
    home_indices: set,
) -> cp.ndarray:
    """Convert viewpoint (camera) positions to robot body-centre positions.

    Inspection viewpoints encode the desired camera position and viewing
    rotation. The robot body centre is offset backward along the viewing
    direction so the camera arrives at the viewpoint position.

    Home nodes are returned unchanged (no camera offset).

    Args:
        positions:    (N, 3) CuPy  --  viewpoint/camera positions.
        rotmats:      (N, 3, 3) CuPy  --  rotation matrices (column 0 = forward).
        home_indices: set of int  --  indices of home/depot nodes.

    Returns:
        (N, 3) CuPy  --  robot body-centre positions.
    """
    xyz = positions.copy()

    # Forward direction = column 0 of rotation matrix
    forwards = rotmats[:, :, 0]

    # Camera offset: subtract forward and up offsets
    xy_norm = cp.linalg.norm(forwards[:, :2], axis=1, keepdims=True)
    xy_norm = cp.maximum(xy_norm, 1e-9)
    xyz[:, 0] -= CAMERA_OFFSET_FORWARD * (forwards[:, 0] / xy_norm[:, 0])
    xyz[:, 1] -= CAMERA_OFFSET_FORWARD * (forwards[:, 1] / xy_norm[:, 0])
    xyz[:, 2] -= CAMERA_OFFSET_UP

    # Restore home positions (no camera offset)
    for idx in home_indices:
        xyz[idx] = positions[idx]

    return xyz
