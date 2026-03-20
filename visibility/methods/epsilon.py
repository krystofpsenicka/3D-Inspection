import logging
import numpy as np
import open3d as o3d
from numpy.linalg import norm
from time import time as get_time
from typing import Tuple, Optional

from ..core.types import FrustumParams, EpsilonHyperparams
from ..core.base import VisibilityQuery
from ..core.constants import NORM_EPS, DELTA_AGG_FUNCS, GAMMA_AGG_FUNCS

logger = logging.getLogger(__name__)


class EpsilonVisibilityQuery(VisibilityQuery):
    """
    Epsilon-visibility based on Lien's paper.

    Implements:
    1. Epsilon estimation (Lemma 6.1).
    2. Epsilon-based occlusion via radial partitioning (Algorithm 5.1).
    """

    def __init__(self, mesh: o3d.geometry.TriangleMesh, target_points: np.ndarray,
                 normals: np.ndarray, frustum_params: FrustumParams,
                 epsilon_deg: Optional[float] = None,
                 hyperparams: Optional[EpsilonHyperparams] = None):
        super().__init__(mesh, target_points, normals, frustum_params)
        self.hp = hyperparams or EpsilonHyperparams()

        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(target_points)
        self.pcd.normals = o3d.utility.Vector3dVector(normals)

        if epsilon_deg is not None:
            logger.info("Using provided epsilon: %s degrees", epsilon_deg)
            self.fixed_epsilon = np.deg2rad(epsilon_deg)
            self.delta = None
            logger.info("Using Epsilon (radians): %.6f (%.3f degrees)",
                        self.fixed_epsilon, np.rad2deg(self.fixed_epsilon))
        else:
            logger.info("Epsilon not provided, estimating δ from point set...")
            self.fixed_epsilon = None
            self.delta = self._estimate_delta()
            logger.info("Estimated δ (sampling density): %.6f", self.delta)

    def visualize_normals(self, normal_scale=20.0):
        """Visualizes the mesh, the target points, and their computed normals."""
        logger.info("Visualizing Normals")

        mesh_vis = o3d.geometry.TriangleMesh(self.mesh)
        mesh_vis.paint_uniform_color([0.8, 0.8, 0.8])
        mesh_vis.compute_vertex_normals()

        pcd_vis = o3d.geometry.PointCloud(self.pcd)
        pcd_vis.paint_uniform_color([1.0, 0.0, 0.0])

        points = np.asarray(pcd_vis.points)
        normals = np.asarray(pcd_vis.normals)

        normal_endpoints = points + (normals * normal_scale)
        normal_vertices = np.concatenate((points, normal_endpoints), axis=0)

        indices = np.arange(len(points))
        normal_lines_indices = np.vstack((indices, indices + len(points))).T

        normal_lines = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(normal_vertices),
            lines=o3d.utility.Vector2iVector(normal_lines_indices)
        )
        normal_lines.colors = o3d.utility.Vector3dVector(
            [[0, 0, 0] for _ in range(len(normal_lines_indices))]
        )

        geometries = [mesh_vis, pcd_vis, normal_lines]

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Approximate Star-Shaped Initialization",
                          width=1600, height=900)

        render_option = vis.get_render_option()
        render_option.point_size = 4.0
        render_option.mesh_show_back_face = True

        for geom in geometries:
            vis.add_geometry(geom)

        vis.run()
        vis.destroy_window()

    def _estimate_delta(self):
        """Estimate sampling density δ (Lemma 6.1): aggregation of k-neighbor distances."""
        sample_size = min(1000, self.num_points)
        if sample_size == 0:
            return 0.1

        sample_indices = np.random.choice(self.num_points, sample_size, replace=False)
        agg_func = DELTA_AGG_FUNCS[self.hp.delta_agg]

        distances = []
        for idx in sample_indices:
            dists, _ = self.kdtree.query(self.target_points[idx], k=self.hp.delta_k)
            if len(dists) > 1:
                distances.append(agg_func(dists[1:]))

        return agg_func(distances) if distances else 0.1

    def _compute_epsilon(self, distances, front_facing):
        """Compute per-viewpoint ε = 2·arctan(δ/(4γ)) where γ = aggregated viewing distance."""
        front_distances = distances[front_facing]
        gamma_func = GAMMA_AGG_FUNCS[self.hp.gamma_method]
        if len(front_distances) == 0:
            gamma = self.frustum_params.far / 4.0
        else:
            gamma = float(gamma_func(front_distances))
        gamma = max(gamma, 1e-6)
        return 2.0 * np.arctan(self.delta / (4.0 * gamma)) * self.hp.epsilon_scale

    def compute_visibility(self, viewpoint, orientation):
        """Compute epsilon-visible region using back-face and occlusion checks."""
        start = get_time()

        frustum_indices = self.points_in_frustum_with_kdtree(viewpoint, orientation)

        if len(frustum_indices) == 0:
            return np.array([]), get_time() - start

        frustum_points = self.target_points[frustum_indices]
        frustum_normals = self.normals[frustum_indices]

        # Check back-face visibility
        view_dirs = frustum_points - viewpoint
        view_dirs_norm = norm(view_dirs, axis=1)
        view_dirs = view_dirs / (view_dirs_norm[:, np.newaxis] + NORM_EPS)

        dot_products = np.sum(view_dirs * frustum_normals, axis=1)
        front_facing = dot_products < self.hp.back_face_threshold

        # Compute per-viewpoint epsilon
        if self.fixed_epsilon is not None:
            epsilon = self.fixed_epsilon
        else:
            epsilon = self._compute_epsilon(view_dirs_norm, front_facing)

        visible_mask = self._check_occlusion(
            viewpoint, frustum_points, frustum_normals, front_facing, epsilon
        )

        visible_indices = frustum_indices[visible_mask]

        comp_time = get_time() - start
        return visible_indices, comp_time

    def _check_occlusion(self, viewpoint: np.ndarray, points: np.ndarray,
                         normals: np.ndarray, front_facing: np.ndarray,
                         epsilon: float) -> np.ndarray:
        """Check occlusion using radial partitioning (Algorithm 5.1)."""
        num_points = len(points)
        if num_points == 0:
            return np.array([], dtype=bool)

        visible = front_facing.copy()

        if not np.any(front_facing):
            return visible

        if epsilon < 1e-6:
            epsilon = 1e-6

        relative = points - viewpoint
        distances = norm(relative, axis=1)

        # Spherical binning
        num_bins_theta = max(1, int(np.ceil(2 * np.pi / epsilon)))
        num_bins_phi = max(1, int(np.ceil(np.pi / epsilon)))

        theta = np.arctan2(relative[:, 1], relative[:, 0])
        phi = np.arcsin(np.clip(relative[:, 2] / (distances + NORM_EPS), -1, 1))

        theta_bins = ((theta + np.pi) / (2 * np.pi) * num_bins_theta).astype(int) % num_bins_theta
        phi_bins = ((phi + np.pi / 2) / np.pi * num_bins_phi).astype(int) % num_bins_phi

        occluder_bins = {}

        for i in range(num_points):
            if not front_facing[i]:
                bin_key = (theta_bins[i], phi_bins[i])
                dist = distances[i]

                if bin_key not in occluder_bins or dist < occluder_bins[bin_key]:
                    occluder_bins[bin_key] = dist

        for i in range(num_points):
            if front_facing[i]:
                bin_key = (theta_bins[i], phi_bins[i])

                if bin_key in occluder_bins:
                    occluder_dist = occluder_bins[bin_key]
                    if distances[i] > occluder_dist:
                        visible[i] = False

        return visible
