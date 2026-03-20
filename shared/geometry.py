"""
Shared geometry utilities.
"""

from __future__ import annotations

import numpy as np


def direction_to_quaternion(direction: np.ndarray) -> np.ndarray:
    """Convert a unit view-direction vector to quaternion [qw, qx, qy, qz].

    Convention: the camera looks along +X in its local frame, so this
    computes the rotation that maps +X -> *direction*.
    """
    from scipy.spatial.transform import Rotation as R

    d = direction / (np.linalg.norm(direction) + 1e-12)
    forward = np.array([1.0, 0.0, 0.0])

    cross = np.cross(forward, d)
    cross_norm = np.linalg.norm(cross)

    if cross_norm < 1e-8:
        if np.dot(forward, d) > 0:
            return np.array([1.0, 0.0, 0.0, 0.0])  # identity
        else:
            return np.array([0.0, 0.0, 1.0, 0.0])  # 180 deg about Y

    axis = cross / cross_norm
    angle = np.arccos(np.clip(np.dot(forward, d), -1.0, 1.0))
    rot = R.from_rotvec(axis * angle)
    qx, qy, qz, qw = rot.as_quat()  # scipy returns [x,y,z,w]
    return np.array([qw, qx, qy, qz])


def direction_roll_to_quaternion(direction: np.ndarray, roll: float = 0.0) -> np.ndarray:
    """Convert direction + roll angle (radians) to quaternion [qw,qx,qy,qz]."""
    from scipy.spatial.transform import Rotation as R

    base_q = direction_to_quaternion(direction)  # [qw,qx,qy,qz]
    if abs(roll) < 1e-8:
        return base_q
    # Roll about the local forward axis (+X)
    roll_rot = R.from_rotvec(np.array([roll, 0.0, 0.0]))
    base_rot = R.from_quat([base_q[1], base_q[2], base_q[3], base_q[0]])  # scipy xyzw
    combined = base_rot * roll_rot  # apply roll in local frame
    qx, qy, qz, qw = combined.as_quat()
    return np.array([qw, qx, qy, qz])


def quaternion_to_forward(q_wxyz: np.ndarray) -> np.ndarray:
    """Extract the forward direction (+X) from a [qw,qx,qy,qz] quaternion."""
    from scipy.spatial.transform import Rotation as R

    rot = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    return rot.as_matrix()[:, 0]
