"""Visualize per-viewpoint and batch visibility query results."""

import logging

import numpy as np
import open3d as o3d

from visibility.core.types import FrustumParams

from ._helpers import generate_tab20_colors, show_geometries
from .frustum_utils import create_viewpoint_geometry
from .model import ModelVisualizer

logger = logging.getLogger(__name__)


class VisibilityVisualizer:
    """Render visibility results over a mesh and target point cloud.

    Parameters
    ----------
    mesh : Open3D TriangleMesh.
    target_points : (N, 3) surface sample points.
    frustum_params : Camera frustum geometry.
    """

    def __init__(
        self,
        mesh: o3d.geometry.TriangleMesh,
        target_points: np.ndarray,
        frustum_params: FrustumParams,
    ):
        self.mesh = mesh
        self.target_points = target_points
        self.num_points = len(target_points)
        self.frustum_params = frustum_params

    # ------------------------------------------------------------------
    # Single viewpoint
    # ------------------------------------------------------------------

    def _create_single_geometries(
        self, visible_indices: np.ndarray, position: np.ndarray, rotation: np.ndarray
    ) -> list:
        """Build geometries for a single viewpoint's visibility."""
        geometries = [ModelVisualizer(self.mesh).create_mesh_geometry()]

        # Point cloud coloured by visibility
        pcd_vis = o3d.geometry.PointCloud()
        pcd_vis.points = o3d.utility.Vector3dVector(self.target_points)
        colors = np.full((self.num_points, 3), [0.3, 0.3, 0.3], dtype=np.float64)
        colors[visible_indices] = [0.0, 1.0, 0.0]
        pcd_vis.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(pcd_vis)

        # Viewpoint sphere + frustum + arrow (small sphere, short arrow for single-VP view)
        geometries += create_viewpoint_geometry(
            position,
            rotation,
            self.frustum_params,
            color=[1.0, 1.0, 0.0],
            sphere_radius=0.015,
            arrow_length=self.frustum_params.far * 0.2,
        )

        return geometries

    def visualize_single(
        self, visible_indices: np.ndarray, position: np.ndarray, rotation: np.ndarray
    ):
        """Visualize one viewpoint's visibility.

        Parameters
        ----------
        visible_indices : 1-D int array of visible point indices.
        position : (3,) viewpoint position.
        rotation : (3, 3) rotation matrix.
        """
        geometries = self._create_single_geometries(visible_indices, position, rotation)
        logger.info("[VisibilityVisualizer] Viewpoint: %d visible points.", len(visible_indices))
        o3d.visualization.draw_geometries(geometries, window_name="Visibility Visualization")

    # ------------------------------------------------------------------
    # All viewpoints
    # ------------------------------------------------------------------

    def _create_all_geometries(self, visibility_map, candidates) -> list:
        """Build geometries for all viewpoints together."""
        geometries = [ModelVisualizer(self.mesh).create_mesh_geometry()]

        num_vps = len(visibility_map)
        vp_colors = generate_tab20_colors(num_vps)

        # Uncovered points
        all_covered = set()
        for visible_indices in visibility_map.values():
            all_covered.update(visible_indices.tolist())
        uncovered_indices = set(range(self.num_points)) - all_covered

        if uncovered_indices:
            uncovered_pcd = o3d.geometry.PointCloud()
            uncovered_pcd.points = o3d.utility.Vector3dVector(
                self.target_points[list(uncovered_indices)]
            )
            uncovered_pcd.paint_uniform_color([1.0, 0.0, 0.0])
            geometries.append(uncovered_pcd)

        for i in range(num_vps):
            pos = np.asarray(candidates[i][0])
            rotation = candidates[i][1]
            visible_indices = visibility_map[i]
            color = list(vp_colors[i % len(vp_colors)])

            geometries += create_viewpoint_geometry(pos, rotation, self.frustum_params, color)

            # Visible points
            if len(visible_indices) > 0:
                visible_pcd = o3d.geometry.PointCloud()
                visible_pcd.points = o3d.utility.Vector3dVector(self.target_points[visible_indices])
                visible_pcd.paint_uniform_color(color)
                geometries.append(visible_pcd)

        return geometries

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

        num_vps = len(visibility_map)
        uncovered_count = self.num_points - len(
            set().union(*(idx.tolist() for idx in visibility_map.values()))
        )

        geometries = self._create_all_geometries(visibility_map, candidates)

        logger.info(
            "[VisibilityVisualizer] Showing %d viewpoints (%d/%d covered).",
            num_vps,
            self.num_points - uncovered_count,
            self.num_points,
        )

        show_geometries(geometries, window_name="All Viewpoints Visibility")
