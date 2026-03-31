"""Visualize per-viewpoint and aggregate visibility query results."""

import logging

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from visibility.core.types import FrustumParams
from .frustum_utils import create_frustum_lineset

logger = logging.getLogger(__name__)


class VisibilityVisualizer:
    """Render visibility results over a mesh and target point cloud.

    Parameters
    ----------
    mesh : Open3D TriangleMesh.
    target_points : (N, 3) surface sample points.
    frustum_params : Camera frustum geometry.
    """

    def __init__(self, mesh: o3d.geometry.TriangleMesh,
                 target_points: np.ndarray,
                 frustum_params: FrustumParams):
        self.mesh = mesh
        self.target_points = target_points
        self.num_points = len(target_points)
        self.frustum_params = frustum_params

    # ------------------------------------------------------------------
    # Single viewpoint
    # ------------------------------------------------------------------

    def visualize_single(self, visible_indices: np.ndarray,
                         position: np.ndarray, rotation: np.ndarray):
        """Visualize one viewpoint's visibility.

        Parameters
        ----------
        visible_indices : 1-D int array of visible point indices.
        position : (3,) viewpoint position.
        rotation : (3, 3) rotation matrix.
        """
        geometries = []

        mesh_vis = o3d.geometry.TriangleMesh(self.mesh)
        mesh_vis.paint_uniform_color([0.8, 0.8, 0.8])
        mesh_vis.compute_vertex_normals()
        geometries.append(mesh_vis)

        # Point cloud coloured by visibility
        pcd_vis = o3d.geometry.PointCloud()
        pcd_vis.points = o3d.utility.Vector3dVector(self.target_points)
        colors = np.full((self.num_points, 3), [0.3, 0.3, 0.3], dtype=np.float64)
        colors[visible_indices] = [0.0, 1.0, 0.0]
        pcd_vis.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(pcd_vis)

        # Viewpoint sphere
        vp_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
        vp_sphere.translate(position)
        vp_sphere.paint_uniform_color([1.0, 0.0, 0.0])
        geometries.append(vp_sphere)

        # Direction arrow
        arrow_length = self.frustum_params.far * 0.2
        forward = rotation[:, 0]
        arrow_end = position + forward * arrow_length
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.array([position, arrow_end]))
        line_set.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
        line_set.colors = o3d.utility.Vector3dVector([[1.0, 1.0, 0.0]])
        geometries.append(line_set)

        # Frustum
        frustum = create_frustum_lineset(position, rotation, self.frustum_params)
        frustum.paint_uniform_color([1.0, 1.0, 0.0])
        geometries.append(frustum)

        logger.info("[VisibilityVisualizer] Viewpoint: %d visible points.",
                    len(visible_indices))
        o3d.visualization.draw_geometries(
            geometries, window_name="Visibility Visualization")

    # ------------------------------------------------------------------
    # All viewpoints
    # ------------------------------------------------------------------

    def visualize_all(self, visibility_map, candidates):
        """Visualize all viewpoints together with per-viewpoint colours.

        Parameters
        ----------
        visibility_map : dict  {int: np.ndarray} mapping VP index -> visible indices.
        candidates : list of (position, rotation) tuples.
        """
        if not visibility_map:
            logger.warning("[VisibilityVisualizer] Visibility map is empty.")
            return

        geometries = []

        base_mesh = o3d.geometry.TriangleMesh(self.mesh)
        base_mesh.paint_uniform_color([0.8, 0.8, 0.8])
        base_mesh.compute_vertex_normals()
        geometries.append(base_mesh)

        num_vps = len(visibility_map)

        # Generate colours, skipping those too close to red (reserved for uncovered)
        raw_colors = plt.cm.tab20(np.linspace(0, 1, max(20, num_vps)))
        vp_colors = []
        for c in raw_colors:
            r, g, b = c[:3]
            if r > 0.7 and g < 0.3 and b < 0.3:
                continue
            vp_colors.append((r, g, b))

        # Uncovered points
        all_covered = set()
        for visible_indices in visibility_map.values():
            all_covered.update(visible_indices.tolist())
        uncovered_indices = set(range(self.num_points)) - all_covered

        if uncovered_indices:
            uncovered_pcd = o3d.geometry.PointCloud()
            uncovered_pcd.points = o3d.utility.Vector3dVector(
                self.target_points[list(uncovered_indices)])
            uncovered_pcd.paint_uniform_color([1.0, 0.0, 0.0])
            geometries.append(uncovered_pcd)

        for i in range(num_vps):
            pos = np.asarray(candidates[i][0])
            orientation = candidates[i][1]
            visible_indices = visibility_map[i]
            color = list(vp_colors[i % len(vp_colors)])

            # Sphere
            vp_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
            vp_sphere.translate(pos)
            vp_sphere.paint_uniform_color(color)
            vp_sphere.compute_vertex_normals()
            geometries.append(vp_sphere)

            # Frustum
            frustum = create_frustum_lineset(pos, orientation, self.frustum_params)
            frustum.paint_uniform_color(color)
            geometries.append(frustum)

            # Direction arrow
            forward = orientation[:, 0]
            arrow_end = pos + forward * 0.5
            arrow = o3d.geometry.LineSet()
            arrow.points = o3d.utility.Vector3dVector(np.array([pos, arrow_end]))
            arrow.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
            arrow.paint_uniform_color(color)
            geometries.append(arrow)

            # Visible points
            if len(visible_indices) > 0:
                visible_pcd = o3d.geometry.PointCloud()
                visible_pcd.points = o3d.utility.Vector3dVector(
                    self.target_points[visible_indices])
                visible_pcd.paint_uniform_color(color)
                geometries.append(visible_pcd)

        logger.info("[VisibilityVisualizer] Showing %d viewpoints (%d/%d covered).",
                    num_vps, self.num_points - len(uncovered_indices), self.num_points)

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="All Viewpoints Visibility",
                          width=1920, height=1080)

        render_option = vis.get_render_option()
        render_option.point_size = 4.0
        render_option.line_width = 2.0
        render_option.mesh_show_back_face = True

        for geom in geometries:
            vis.add_geometry(geom)

        logger.info("Press Q to close visualization")
        vis.run()
        vis.destroy_window()
