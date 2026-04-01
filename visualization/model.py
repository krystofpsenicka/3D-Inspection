"""Visualize mesh model, target point cloud, and surface normals."""

import logging

import numpy as np
import open3d as o3d

logger = logging.getLogger(__name__)


class ModelVisualizer:
    """Visualizer for mesh, target points, and/or normals.

    Parameters
    ----------
    mesh : Open3D TriangleMesh.
    target_points : (N, 3) array of surface sample points (optional).
    normals : (N, 3) outward surface normals (optional).
    scale : If provided, the mesh (and target points) is scaled by this
        factor.
    """

    def __init__(self, mesh: o3d.geometry.TriangleMesh,
                 target_points: np.ndarray | None = None,
                 normals: np.ndarray | None = None,
                 scale: float | None = None):
        self.mesh = mesh
        self.target_points = target_points
        self.normals = normals
        self.scale = scale

    # ------------------------------------------------------------------
    # Geometry builders
    # ------------------------------------------------------------------

    def create_mesh_geometry(self) -> o3d.geometry.TriangleMesh:
        """Return a grey copy of the mesh (optionally scaled)."""
        mesh_vis = o3d.geometry.TriangleMesh(self.mesh)
        if self.scale is not None:
            mesh_vis.scale(self.scale, center=mesh_vis.get_center())
        mesh_vis.paint_uniform_color([0.8, 0.8, 0.8])
        mesh_vis.compute_vertex_normals()
        return mesh_vis

    def create_wireframe_geometry(
            self, color: tuple = (0.7, 0.7, 0.7)) -> o3d.geometry.LineSet:
        """Return a wireframe LineSet from the mesh."""
        wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(self.mesh)
        wireframe.paint_uniform_color(list(color))
        return wireframe

    def create_points_geometry(
            self, color: tuple = (1.0, 0.0, 0.0)) -> o3d.geometry.PointCloud | None:
        """Return a coloured PointCloud of target points, or None."""
        if self.target_points is None:
            return None
        points = self.target_points
        if self.scale is not None:
            points = points * self.scale
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color(list(color))
        return pcd

    def create_normals_geometry(
            self, normal_scale: float = 0.05) -> o3d.geometry.LineSet | None:
        """Return a LineSet of normal vectors, or None."""
        if self.target_points is None or self.normals is None:
            return None
        points = self.target_points
        if self.scale is not None:
            points = points * self.scale
        normal_endpoints = points + (self.normals * normal_scale)
        normal_vertices = np.concatenate((points, normal_endpoints), axis=0)

        indices = np.arange(len(points))
        normal_lines_indices = np.vstack((indices, indices + len(points))).T

        normal_lines = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(normal_vertices),
            lines=o3d.utility.Vector2iVector(normal_lines_indices),
        )
        normal_lines.colors = o3d.utility.Vector3dVector(
            [[0, 0, 0] for _ in range(len(normal_lines_indices))]
        )
        return normal_lines

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def visualize(self, show_mesh: bool = True, show_points: bool = True,
                  show_normals: bool = True, normal_scale: float = 0.05):
        """Show any combination of mesh, point cloud, and normals."""
        geometries = []

        if show_mesh:
            geometries.append(self.create_mesh_geometry())

        if show_points:
            pcd = self.create_points_geometry()
            if pcd is not None:
                geometries.append(pcd)

        if show_normals:
            normals_geom = self.create_normals_geometry(normal_scale)
            if normals_geom is not None:
                geometries.append(normals_geom)

        if not geometries:
            logger.warning("[ModelVisualizer] Nothing to show.")
            return

        o3d.visualization.draw_geometries(
            geometries, window_name="Target Points and Normals")
