"""Visualize mesh model, target point cloud, and surface normals."""

import logging

import numpy as np
import open3d as o3d

logger = logging.getLogger(__name__)


class ModelVisualizer:
    """Flexible visualizer for mesh, target points, and/or normals.

    Parameters
    ----------
    mesh : Open3D TriangleMesh.
    target_points : (N, 3) array of surface sample points (optional).
    normals : (N, 3) outward surface normals (optional).
    scale : If provided, the mesh (and target points) are scaled by this
        factor for display — useful for showing raw unscaled meshes at the
        pipeline's working scale.
    """

    def __init__(self, mesh: o3d.geometry.TriangleMesh,
                 target_points: np.ndarray | None = None,
                 normals: np.ndarray | None = None,
                 scale: float | None = None):
        self.mesh = mesh
        self.target_points = target_points
        self.normals = normals
        self.scale = scale

    def visualize(self, show_mesh: bool = True, show_points: bool = True,
                  show_normals: bool = True, normal_scale: float = 0.05):
        """Show any combination of mesh, point cloud, and normals."""
        geometries = []

        if show_mesh:
            mesh_vis = o3d.geometry.TriangleMesh(self.mesh)
            if self.scale is not None:
                mesh_vis.scale(self.scale, center=mesh_vis.get_center())
            mesh_vis.paint_uniform_color([0.8, 0.8, 0.8])
            mesh_vis.compute_vertex_normals()
            geometries.append(mesh_vis)

        points = self.target_points
        if points is not None and self.scale is not None:
            points = points * self.scale

        if show_points and points is not None:
            pcd_vis = o3d.geometry.PointCloud()
            pcd_vis.points = o3d.utility.Vector3dVector(points)
            pcd_vis.paint_uniform_color([1.0, 0.0, 0.0])
            geometries.append(pcd_vis)

        if show_normals and points is not None and self.normals is not None:
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
            geometries.append(normal_lines)

        if not geometries:
            logger.warning("[ModelVisualizer] Nothing to show.")
            return

        o3d.visualization.draw_geometries(
            geometries, window_name="Target Points and Normals")
