"""Shared frustum geometry utility (Isaac Sim / USD variant)."""

import numpy as np

from visibility.core.types import FrustumParams

from .._usd_primitives import create_lineset_prim, create_sphere_prim


def add_frustum_lineset(
    stage,
    path: str,
    viewpoint: np.ndarray,
    rotation: np.ndarray,
    params: FrustumParams,
    color: tuple = (1.0, 1.0, 0.0),
    width: float = 2.0,
) -> str:
    """Create a frustum wireframe as a ``BasisCurves`` prim.

    Parameters
    ----------
    stage : Usd.Stage
    path : USD prim path.
    viewpoint : (3,) array  --  camera position.
    rotation : (3, 3) rotation matrix  --  columns are [forward, right, up].
    params : FrustumParams defining FOV, near/far planes.
    color : RGB colour.
    width : Line width.

    Returns
    -------
    The prim path string.
    """
    half_angle_rad = params.fov_y / 2.0
    far_half_size = params.far * np.tan(half_angle_rad)

    forward, right, up = rotation[:, 0], rotation[:, 1], rotation[:, 2]

    far_center = viewpoint + forward * params.far

    r = right * far_half_size
    u = up * far_half_size

    corners = np.array(
        [
            viewpoint,
            far_center + r + u,
            far_center - r + u,
            far_center - r - u,
            far_center + r - u,
        ]
    )

    lines = np.array(
        [
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1],
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
        ]
    )

    return create_lineset_prim(stage, path, corners, lines, color=color, width=width)


def add_viewpoint_geometry(
    stage,
    base_path: str,
    position: np.ndarray,
    rotation: np.ndarray,
    frustum_params: FrustumParams,
    color: tuple | list,
    sphere_radius: float = 0.15,
    arrow_length: float = 0.5,
) -> list[str]:
    """Create the standard viewpoint geometry: sphere + frustum + direction arrow.

    Parameters
    ----------
    stage : Usd.Stage
    base_path : Parent prim path  --  children are created beneath it.
    position : (3,) camera position.
    rotation : (3, 3) rotation matrix  --  columns are [forward, right, up].
    frustum_params : Camera frustum geometry.
    color : RGB colour for all three primitives.
    sphere_radius : Radius of the viewpoint marker sphere.
    arrow_length : Length of the direction arrow line.

    Returns
    -------
    List of created prim paths [sphere, frustum, arrow].
    """
    color = tuple(color)
    paths = []

    paths.append(
        create_sphere_prim(
            stage, f"{base_path}/sphere", position=position, radius=sphere_radius, color=color
        )
    )

    paths.append(
        add_frustum_lineset(
            stage,
            f"{base_path}/frustum",
            viewpoint=position,
            rotation=rotation,
            params=frustum_params,
            color=color,
        )
    )

    forward = rotation[:, 0]
    arrow_end = position + forward * arrow_length
    arrow_pts = np.array([position, arrow_end])
    arrow_lines = np.array([[0, 1]])
    paths.append(
        create_lineset_prim(
            stage, f"{base_path}/arrow", points=arrow_pts, lines=arrow_lines, color=color, width=2.0
        )
    )

    return paths
