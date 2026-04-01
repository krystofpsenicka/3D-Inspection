"""Visualize free-space sampling and resampling progression."""

import logging

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from visibility.core.types import FrustumParams
from .frustum_utils import create_viewpoint_geometry
from .model import ModelVisualizer
from ._helpers import generate_tab20_colors, show_geometries

logger = logging.getLogger(__name__)


class SamplingVisualizer:
    """Render sampling heatmaps and resampling progression.

    Parameters
    ----------
    mesh : Open3D TriangleMesh.
    target_points : (N, 3) surface sample points.
    normals : (N, 3) outward surface normals.
    frustum_params : Camera frustum geometry.
    """

    def __init__(self, mesh: o3d.geometry.TriangleMesh,
                 target_points: np.ndarray,
                 normals: np.ndarray,
                 frustum_params: FrustumParams):
        self.mesh = mesh
        self.target_points = target_points
        self.normals = normals
        self.num_points = len(target_points)
        self.frustum_params = frustum_params

    # ------------------------------------------------------------------
    # Free-space heatmap
    # ------------------------------------------------------------------

    def _create_free_space_geometries(self,
                                      outside_positions, outside_weights,
                                      inside_positions, inside_weights,
                                      outside_colormap, inside_colormap) -> list:
        """Build geometries for the free-space sampling heatmap."""
        geometries = [ModelVisualizer(self.mesh).create_wireframe_geometry()]

        for positions, weights, cmap_name in [
            (outside_positions, outside_weights, outside_colormap),
            (inside_positions, inside_weights, inside_colormap),
        ]:
            if len(positions) == 0:
                continue
            w_min, w_max = weights.min(), weights.max()
            if w_max > w_min:
                norm_w = (weights - w_min) / (w_max - w_min)
            else:
                norm_w = np.ones_like(weights)
            cmap = plt.cm.get_cmap(cmap_name)
            rgba = cmap(norm_w)
            colors = rgba[:, :3]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(positions)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            geometries.append(pcd)

        return geometries

    def visualize_free_space(self,
                             outside_positions: np.ndarray,
                             outside_weights: np.ndarray,
                             inside_positions: np.ndarray,
                             inside_weights: np.ndarray,
                             outside_colormap: str = "Reds",
                             inside_colormap: str = "Blues",
                             point_size: float = 3.0,
                             window_name: str = "Free-Space Sampling Heatmap"):
        """Visualize feasible sampling regions as coloured point clouds."""
        geometries = self._create_free_space_geometries(
            outside_positions, outside_weights,
            inside_positions, inside_weights,
            outside_colormap, inside_colormap)

        logger.info("[SamplingVisualizer] Free-space heatmap: %d outside, %d inside points",
                    len(outside_positions), len(inside_positions))
        logger.info("Press Q to close visualization")

        show_geometries(geometries, window_name=window_name,
                        point_size=point_size, line_width=1.0,
                        mesh_show_back_face=False)

    # ------------------------------------------------------------------
    # Resampling progression
    # ------------------------------------------------------------------

    def visualize_resampling_progression(self, normal_vis_map, normal_candidates,
                                         targeted_vis_map, targeted_candidates,
                                         point_size=4.0):
        """Visualize resampling in three phases: normal VPs, targeted one-by-one, combined."""
        BLUE = [0.0, 0.4, 0.8]
        ORANGE = [1.0, 0.5, 0.0]
        RED = [1.0, 0.0, 0.0]
        GREEN = [0.0, 1.0, 0.0]
        GRAY = [0.5, 0.5, 0.5]

        n_normal = len(normal_candidates)
        n_targeted = len(targeted_candidates)
        model_vis = ModelVisualizer(self.mesh)

        # Cumulative coverage from normal candidates
        normal_covered = set()
        for visible_indices in normal_vis_map.values():
            normal_covered.update(visible_indices.tolist())
        normal_coverage_pct = len(normal_covered) / self.num_points * 100

        # --- Phase 1: Normal candidates ---
        geometries = [model_vis.create_wireframe_geometry()]

        raw_blues = plt.cm.Blues(np.linspace(0.4, 0.9, max(1, n_normal)))

        for i in range(n_normal):
            pos = np.asarray(normal_candidates[i][0])
            orientation = normal_candidates[i][1]
            color = list(raw_blues[i][:3])

            geometries += create_viewpoint_geometry(
                pos, orientation, self.frustum_params, color)

            visible_indices = normal_vis_map.get(i, np.array([], dtype=int))
            if len(visible_indices) > 0:
                vis_pcd = o3d.geometry.PointCloud()
                vis_pcd.points = o3d.utility.Vector3dVector(
                    self.target_points[visible_indices])
                vis_pcd.paint_uniform_color(color)
                geometries.append(vis_pcd)

        uncovered_indices = set(range(self.num_points)) - normal_covered
        if uncovered_indices:
            uncov_pcd = o3d.geometry.PointCloud()
            uncov_pcd.points = o3d.utility.Vector3dVector(
                self.target_points[list(uncovered_indices)])
            uncov_pcd.paint_uniform_color(RED)
            geometries.append(uncov_pcd)

        logger.info("Phase 1: %d normal VPs, coverage=%.1f%%. Press Q to continue.",
                    n_normal, normal_coverage_pct)
        show_geometries(
            geometries,
            window_name=f"Phase 1: Normal VPs ({n_normal} VPs, "
                        f"coverage={normal_coverage_pct:.1f}%)",
            point_size=point_size)

        # --- Phase 2: Targeted candidates one-by-one ---
        cumulative_covered = set(normal_covered)

        for t_idx in range(n_targeted):
            geometries = [model_vis.create_wireframe_geometry()]

            # Normal candidates as small blue spheres (marker-only, no frustum/arrow)
            for i in range(n_normal):
                pos = np.asarray(normal_candidates[i][0])
                vp_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
                vp_sphere.translate(pos)
                vp_sphere.paint_uniform_color(BLUE)
                vp_sphere.compute_vertex_normals()
                geometries.append(vp_sphere)

            # Previously-added targeted candidates as small orange spheres
            for j in range(t_idx):
                pos = np.asarray(targeted_candidates[j][0])
                vp_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
                vp_sphere.translate(pos)
                vp_sphere.paint_uniform_color(ORANGE)
                vp_sphere.compute_vertex_normals()
                geometries.append(vp_sphere)

            # Current targeted candidate — sphere + frustum + arrow
            cur_pos = np.asarray(targeted_candidates[t_idx][0])
            cur_orient = targeted_candidates[t_idx][1]

            geometries += create_viewpoint_geometry(
                cur_pos, cur_orient, self.frustum_params, ORANGE,
                sphere_radius=0.2)

            # Point colouring
            cur_visible = targeted_vis_map.get(t_idx, np.array([], dtype=int))
            newly_covered = set(cur_visible.tolist()) - cumulative_covered
            still_uncovered = set(range(self.num_points)) - cumulative_covered - newly_covered

            if cumulative_covered:
                gray_pcd = o3d.geometry.PointCloud()
                gray_pcd.points = o3d.utility.Vector3dVector(
                    self.target_points[list(cumulative_covered)])
                gray_pcd.paint_uniform_color(GRAY)
                geometries.append(gray_pcd)

            if newly_covered:
                green_pcd = o3d.geometry.PointCloud()
                green_pcd.points = o3d.utility.Vector3dVector(
                    self.target_points[list(newly_covered)])
                green_pcd.paint_uniform_color(GREEN)
                geometries.append(green_pcd)

            if still_uncovered:
                red_pcd = o3d.geometry.PointCloud()
                red_pcd.points = o3d.utility.Vector3dVector(
                    self.target_points[list(still_uncovered)])
                red_pcd.paint_uniform_color(RED)
                geometries.append(red_pcd)

            cumulative_covered.update(newly_covered)
            cur_coverage_pct = len(cumulative_covered) / self.num_points * 100

            logger.info("Targeted VP %d/%d: +%d pts, coverage=%.1f%%. Press Q to continue.",
                        t_idx + 1, n_targeted, len(newly_covered), cur_coverage_pct)
            show_geometries(
                geometries,
                window_name=f"Targeted VP {t_idx+1}/{n_targeted} "
                            f"(+{len(newly_covered)} pts, "
                            f"coverage={cur_coverage_pct:.1f}%)",
                point_size=point_size)

        # --- Phase 3: Final combined view ---
        all_candidates = list(normal_candidates) + list(targeted_candidates)
        all_vis_map = {}
        for i, vis_indices in normal_vis_map.items():
            all_vis_map[i] = vis_indices
        for i, vis_indices in targeted_vis_map.items():
            all_vis_map[n_normal + i] = vis_indices

        total_covered = set()
        for vis_indices in all_vis_map.values():
            total_covered.update(vis_indices.tolist())
        final_coverage_pct = len(total_covered) / self.num_points * 100

        geometries = [model_vis.create_wireframe_geometry()]

        vp_colors = generate_tab20_colors(len(all_candidates))

        uncovered_final = set(range(self.num_points)) - total_covered
        if uncovered_final:
            uncov_pcd = o3d.geometry.PointCloud()
            uncov_pcd.points = o3d.utility.Vector3dVector(
                self.target_points[list(uncovered_final)])
            uncov_pcd.paint_uniform_color(RED)
            geometries.append(uncov_pcd)

        for i, candidate in enumerate(all_candidates):
            pos = np.asarray(candidate[0])
            orientation = candidate[1]
            is_targeted = i >= n_normal
            base_color = ORANGE if is_targeted else BLUE
            vis_color = list(vp_colors[i % len(vp_colors)])

            geometries += create_viewpoint_geometry(
                pos, orientation, self.frustum_params, base_color)

            visible_indices = all_vis_map.get(i, np.array([], dtype=int))
            if len(visible_indices) > 0:
                vis_pcd = o3d.geometry.PointCloud()
                vis_pcd.points = o3d.utility.Vector3dVector(
                    self.target_points[visible_indices])
                vis_pcd.paint_uniform_color(vis_color)
                geometries.append(vis_pcd)

        logger.info("Final: %d normal + %d targeted VPs, coverage=%.1f%%. Press Q to close.",
                    n_normal, n_targeted, final_coverage_pct)
        show_geometries(
            geometries,
            window_name=f"Final: {n_normal} normal + {n_targeted} targeted "
                        f"(coverage={final_coverage_pct:.1f}%)",
            point_size=point_size)
