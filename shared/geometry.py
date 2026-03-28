"""
Geometry utils.
"""

from __future__ import annotations

import cupy as cp
import numpy as np


def direction_to_rotmat(direction: np.ndarray) -> np.ndarray:
    """Rotation matrix that maps the local +X axis to *direction*.

    Convention: the camera looks along local +X.
    Uses the Rodrigues formula (axis-angle → rotation matrix) directly.

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
            # 180° about Y: Ry(π) = diag(-1, 1, -1)
            return np.diag([-1.0, 1.0, -1.0])

    axis = cross / cross_norm
    angle = np.arccos(np.clip(np.dot(forward, d), -1.0, 1.0))

    # Rodrigues formula: R = I + sin(θ)K + (1 - cos(θ))K²
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


def direction_roll_to_rotmat(direction: np.ndarray,
                              roll: float = 0.0) -> np.ndarray:
    """Direction + roll (radians) → (3,3) rotation matrix."""
    base = direction_to_rotmat(direction)
    if abs(roll) < 1e-8:
        return base
    # Roll rotation about the local X axis
    c, s = np.cos(roll), np.sin(roll)
    Rx = np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ])
    return base @ Rx


def directions_rolls_to_rotmats(directions_gpu, rolls_gpu):
    """(N,3) unit directions + (N,) roll angles -> (N,3,3) rotation matrices (GPU).

    Convention: camera looks along local +X.
    """
    NORM_EPS = 1e-12
    x = directions_gpu  # (N, 3) — forward axis

    # Reference up vector; fall back to +Y when direction ≈ ±Z
    up = cp.tile(cp.array([0.0, 0.0, 1.0], dtype=cp.float32), (len(x), 1))
    up[cp.abs(x[:, 2]) > 0.99] = cp.array([0.0, 1.0, 0.0], dtype=cp.float32)

    # y = normalize(up × x),  z = x × y
    y = cp.cross(up, x)
    y /= cp.maximum(cp.linalg.norm(y, axis=1, keepdims=True), NORM_EPS)
    z = cp.cross(x, y)

    # Apply roll around viewing axis
    cos_r = cp.cos(rolls_gpu)[:, None]
    sin_r = cp.sin(rolls_gpu)[:, None]
    y_r = cos_r * y + sin_r * z
    z_r = -sin_r * y + cos_r * z

    return cp.stack([x, y_r, z_r], axis=-1)   # (N, 3, 3)
