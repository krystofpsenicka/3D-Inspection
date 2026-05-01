"""Visualize mesh model, target point cloud, and surface normals (Isaac Sim variant)."""

import logging

import numpy as np
import trimesh

from .._usd_primitives import (
    create_lineset_prim,
    create_mesh_prim,
    create_points_prim,
    create_wireframe_from_trimesh,
)

logger = logging.getLogger(__name__)


class ModelVisualizer:
    """Stage-builder for mesh, target points, and/or normals.

    Parameters
    ----------
    mesh : trimesh.Trimesh  --  the model mesh.
    target_points : (N, 3) array of surface sample points (optional).
    normals : (N, 3) outward surface normals (optional).
    """

    def __init__(
        self,
        mesh: trimesh.Trimesh,
        target_points: np.ndarray | None = None,
        normals: np.ndarray | None = None,
    ):
        self.mesh = mesh
        self.target_points = target_points
        self.normals = normals

    def add_mesh(
        self, stage, path: str, color: tuple = (0.8, 0.8, 0.8), opacity: float = 1.0
    ) -> str:
        """Add a grey mesh prim to *stage* at *path*."""
        return create_mesh_prim(stage, path, self.mesh, color=color, opacity=opacity)

    def add_wireframe(self, stage, path: str, color: tuple = (0.7, 0.7, 0.7)) -> str:
        """Add a wireframe prim to *stage* at *path*."""
        return create_wireframe_from_trimesh(stage, path, self.mesh, color=color)

    def add_points(
        self, stage, path: str, color: tuple = (1.0, 0.0, 0.0), point_size: float = 4.0
    ) -> str | None:
        """Add a coloured point cloud of target points, or ``None`` if unavailable."""
        if self.target_points is None:
            return None
        return create_points_prim(
            stage, path, self.target_points, colors=color, point_size=point_size
        )

    def add_normals(self, stage, path: str, normal_scale: float = 0.05) -> str | None:
        """Add normal-vector line segments, or ``None`` if unavailable."""
        if self.target_points is None or self.normals is None:
            return None

        points = self.target_points
        endpoints = points + (self.normals * normal_scale)
        vertices = np.concatenate((points, endpoints), axis=0)

        n = len(points)
        lines = np.column_stack((np.arange(n), np.arange(n, 2 * n)))

        return create_lineset_prim(stage, path, vertices, lines, color=(0, 0, 0), width=1.0)
