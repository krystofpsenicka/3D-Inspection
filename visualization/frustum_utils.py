"""Shared frustum geometry utility."""

import numpy as np
import open3d as o3d

from visibility.core.types import FrustumParams


def create_frustum_lineset(
    viewpoint: np.ndarray,
    rotation: np.ndarray,
    params: FrustumParams,
) -> o3d.geometry.LineSet:
    """Create a LineSet representing the frustum.

    Parameters
    ----------
    viewpoint : (3,) array  --  camera position.
    rotation : (3, 3) rotation matrix  --  columns are [forward, right, up].
    params : FrustumParams defining FOV, near/far planes.
    """
    half_angle_rad = params.fov_y / 2.0
    far_half_size = params.far * np.tan(half_angle_rad)

    forward, right, up = rotation[:, 0], rotation[:, 1], rotation[:, 2]

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
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 1],
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(corners))
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
    return line_set


def create_viewpoint_geometry(
    position: np.ndarray,
    rotation: np.ndarray,
    frustum_params: FrustumParams,
    color: tuple | list,
    sphere_radius: float = 0.15,
    arrow_length: float = 0.5,
) -> list[o3d.geometry.Geometry]:
    """Create the standard viewpoint geometry: sphere + frustum + direction arrow.

    Parameters
    ----------
    position : (3,) camera position.
    rotation : (3, 3) rotation matrix  --  columns are [forward, right, up].
    frustum_params : Camera frustum geometry.
    color : RGB colour for all three primitives.
    sphere_radius : Radius of the viewpoint marker sphere.
    arrow_length : Length of the direction arrow line.

    Returns
    -------
    List of [sphere, frustum_lineset, arrow_lineset].
    """
    color = list(color)

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    sphere.translate(position)
    sphere.paint_uniform_color(color)
    sphere.compute_vertex_normals()

    frustum = create_frustum_lineset(position, rotation, frustum_params)
    frustum.paint_uniform_color(color)

    forward = rotation[:, 0]
    arrow_end = position + forward * arrow_length
    arrow = o3d.geometry.LineSet()
    arrow.points = o3d.utility.Vector3dVector(np.array([position, arrow_end]))
    arrow.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
    arrow.paint_uniform_color(color)

    return [sphere, frustum, arrow]
