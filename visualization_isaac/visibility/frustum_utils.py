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
    color: tuple = (0.2, 0.55, 1.0),
    width: float = 0.02,
) -> str:
    """Frustum wireframe as a ``BasisCurves`` prim. ``rotation`` columns: [forward, right, up]."""
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
    """Sphere + frustum + direction arrow under ``base_path``. Returns [sphere, frustum, arrow] paths."""
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
            stage, f"{base_path}/arrow", points=arrow_pts, lines=arrow_lines, color=color, width=0.02
        )
    )

    return paths
