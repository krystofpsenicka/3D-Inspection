"""Shared frustum geometry utility."""

import numpy as np
import open3d as o3d

from visibility.core.types import FrustumParams


def create_frustum_lineset(
    viewpoint: np.ndarray,
    orientation: np.ndarray,
    params: FrustumParams,
) -> o3d.geometry.LineSet:
    """Create a LineSet representing the frustum volume.

    Parameters
    ----------
    viewpoint : (3,) array — camera position.
    orientation : (3, 3) rotation matrix — columns are [forward, right, up].
    params : FrustumParams defining FOV, near/far planes.
    """
    half_angle_rad = params.fov_y / 2.0
    far_half_size = params.far * np.tan(half_angle_rad)

    forward, right, up = orientation[:, 0], orientation[:, 1], orientation[:, 2]

    far_center = viewpoint + forward * params.far

    r = right * far_half_size
    u = up * far_half_size

    corners = [
        viewpoint,
        far_center + r + u,
        far_center - r + u,
        far_center - r - u,
        far_center + r - u,
    ]

    lines = [
        [1, 2], [2, 3], [3, 4], [4, 1],
        [0, 1], [0, 2], [0, 3], [0, 4],
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(corners))
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
    return line_set
