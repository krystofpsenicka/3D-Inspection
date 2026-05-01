"""
Geometry utils.
"""

from __future__ import annotations

import cupy as cp
import numpy as np


def orient_normals_outward(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Flip normals if they predominantly point inward (toward centroid)."""
    centroid = np.mean(points, axis=0)
    radial = points - centroid  # outward radial vectors
    mean_dot = np.mean(np.sum(normals * radial, axis=1))
    if mean_dot < 0:  # majority pointing inward -> flip
        return -normals
    return normals


def direction_to_rotmat(direction: np.ndarray) -> np.ndarray:
    """Rotation matrix that maps the local +X axis to *direction*.

    Convention: the camera looks along local +X.
    Uses the Rodrigues formula (axis-angle -> rotation matrix) directly.

    Returns:
        (3, 3) numpy rotation matrix.
    """
    d = direction / (np.linalg.norm(direction) + 1e-12)
    forward = np.array([1.0, 0.0, 0.0])

    cross = np.cross(forward, d)
    cross_norm = np.linalg.norm(cross)

    if cross_norm < 1e-8:
        if np.dot(forward, d) > 0:
            return np.eye(3)
        else:
            # 180 deg about Y: Ry(π) = diag(-1, 1, -1)
            return np.diag([-1.0, 1.0, -1.0])

    axis = cross / cross_norm
    angle = np.arccos(np.clip(np.dot(forward, d), -1.0, 1.0))

    # Rodrigues formula: R = I + sin(theta)K + (1 - cos(theta))K²
    K = np.array(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ]
    )
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


def direction_roll_to_rotmat(direction: np.ndarray, roll: float = 0.0) -> np.ndarray:
    """Direction + roll (radians) -> (3,3) rotation matrix."""
    base = direction_to_rotmat(direction)
    if abs(roll) < 1e-8:
        return base
    # Roll rotation about the local X axis
    c, s = np.cos(roll), np.sin(roll)
    Rx = np.array(
        [
            [1, 0, 0],
            [0, c, -s],
            [0, s, c],
        ]
    )
    return base @ Rx


def directions_rolls_to_rotmats(directions_gpu, rolls_gpu):
    """(N,3) unit directions + (N,) roll angles -> (N,3,3) rotation matrices (GPU).

    Convention: camera looks along local +X.
    """
    NORM_EPS = 1e-12
    x = directions_gpu  # (N, 3) — forward axis

    # Reference up vector; fall back to +Y when direction ~ +/-Z
    up = cp.tile(cp.array([0.0, 0.0, 1.0], dtype=cp.float32), (len(x), 1))
    up[cp.abs(x[:, 2]) > 0.99] = cp.array([0.0, 1.0, 0.0], dtype=cp.float32)

    # y = normalize(up x x),  z = x x y
    y = cp.cross(up, x)
    y /= cp.maximum(cp.linalg.norm(y, axis=1, keepdims=True), NORM_EPS)
    z = cp.cross(x, y)

    # Apply roll around viewing axis
    cos_r = cp.cos(rolls_gpu)[:, None]
    sin_r = cp.sin(rolls_gpu)[:, None]
    y_r = cos_r * y + sin_r * z
    z_r = -sin_r * y + cos_r * z

    return cp.stack([x, y_r, z_r], axis=-1)  # (N, 3, 3)


def rotmats_to_directions_rolls(rotmats_gpu):
    """(N,3,3) rotation matrices -> (N,3) directions + (N,) roll angles (GPU).

    Inverse of :func:`directions_rolls_to_rotmats`.
    Convention: camera looks along local +X.
    """
    NORM_EPS = 1e-12

    # Forward direction
    directions = rotmats_gpu[:, :, 0]  # (N, 3)

    # Reconstruct zero-roll y and z axes
    x = directions
    up = cp.tile(cp.array([0.0, 0.0, 1.0], dtype=cp.float32), (len(x), 1))
    up[cp.abs(x[:, 2]) > 0.99] = cp.array([0.0, 1.0, 0.0], dtype=cp.float32)

    y0 = cp.cross(up, x)
    y0 /= cp.maximum(cp.linalg.norm(y0, axis=1, keepdims=True), NORM_EPS)
    z0 = cp.cross(x, y0)

    # Actual y-axis from the rotation matrix
    y_actual = rotmats_gpu[:, :, 1]  # (N, 3)

    # y_r = cos(roll)*y0 + sin(roll)*z0  ->  roll = atan2(y_r.z0, y_r.y0)
    cos_roll = cp.sum(y_actual * y0, axis=1)
    sin_roll = cp.sum(y_actual * z0, axis=1)
    rolls = cp.arctan2(sin_roll, cos_roll)

    return directions, rolls
