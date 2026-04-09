"""Geometric transforms between viewpoint and robot-body coordinate frames."""

from __future__ import annotations

import cupy as cp

from .constants import CAMERA_OFFSET_FORWARD, CAMERA_OFFSET_UP


def viewpoints_to_robot_waypoints(
    positions: cp.ndarray,
    rotmats: cp.ndarray,
    home_indices: set,
) -> cp.ndarray:
    """Convert viewpoint (camera) positions to robot body-centre positions.

    Inspection waypoints encode the desired camera position and viewing
    direction. The robot body centre is offset backward along the viewing
    direction so the camera arrives at the waypoint position.

    Home nodes are returned unchanged (no camera offset).

    Args:
        positions:    (N, 3) CuPy — viewpoint/camera positions.
        rotmats:      (N, 3, 3) CuPy — rotation matrices (column 0 = forward).
        home_indices: set of int — indices of home/depot nodes.

    Returns:
        (N, 3) CuPy — robot body-centre positions.
    """
    xyz = positions.copy()

    # Forward direction = column 0 of rotation matrix
    forwards = rotmats[:, :, 0]  # (N, 3)

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
