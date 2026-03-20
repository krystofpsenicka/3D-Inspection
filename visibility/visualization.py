import logging
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import PIL.Image
from typing import Dict, Tuple

from .core.types import FrustumParams, OptimizationResult
from .core.base import get_frustum_basis, get_frustum_basis_from_quaternion
from shared.geometry import quaternion_to_forward

logger = logging.getLogger(__name__)


class Visualizer:
    """
    Handles all Open3D visualization tasks for the visibility pipeline,
    including normals, frustums, and final coverage results.
    """

    def __init__(self, mesh: o3d.geometry.TriangleMesh, target_points: np.ndarray,
                 normals: np.ndarray, frustum_params: FrustumParams):
        self.mesh = mesh
        self.target_points = target_points
        self.normals = normals
        self.num_points = len(target_points)
        self.frustum_params = frustum_params
        logger.info("[Visualizer] Initialized visualization module.")

    def visualize_normals(self, normal_scale: float = 0.05):
        """Visualizes the mesh, the target points, and their computed normals."""
        logger.info("[Visualizer] Visualizing target points and normals...")

        pcd_vis = o3d.geometry.PointCloud()
        pcd_vis.points = o3d.utility.Vector3dVector(self.target_points)
        pcd_vis.paint_uniform_color([1.0, 0.0, 0.0])

        points = np.asarray(pcd_vis.points)

        normal_endpoints = points + (self.normals * normal_scale)
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

        mesh_vis = o3d.geometry.TriangleMesh(self.mesh)
        mesh_vis.paint_uniform_color([0.8, 0.8, 0.8])
        mesh_vis.compute_vertex_normals()

        geometries = [mesh_vis, pcd_vis, normal_lines]
        o3d.visualization.draw_geometries(geometries, window_name="Target Points and Normals")

    def visualize_visibility_results(self,
                                     visibility_map,
                                     candidate_index: int = 0,
                                     candidates=None):
        """Visualizes a specific candidate's visibility against the mesh and target points."""
        if not visibility_map:
            logger.warning("[Visualizer] Visibility map is empty.")
            return

        index_to_visualize = candidate_index % len(visibility_map)
        selected_vp_pos = np.asarray(candidates[index_to_visualize][0])
        selected_vp_orient = np.asarray(candidates[index_to_visualize][1])
        visible_indices = visibility_map[index_to_visualize]

        geometries = []

        mesh_vis = o3d.geometry.TriangleMesh(self.mesh)
        mesh_vis.paint_uniform_color([0.8, 0.8, 0.8])
        mesh_vis.compute_vertex_normals()
        geometries.append(mesh_vis)

        pcd_vis = o3d.geometry.PointCloud()
        pcd_vis.points = o3d.utility.Vector3dVector(self.target_points)

        colors = np.full((self.num_points, 3), [0.3, 0.3, 0.3], dtype=np.float64)
        colors[visible_indices] = [0.0, 1.0, 0.0]
        pcd_vis.colors = o3d.utility.Vector3dVector(colors)

        for i, key in enumerate(candidate_list):
            pos = np.array(key[0])
            color = [0.0, 0.0, 1.0]
            radius = 0.01

            if i == index_to_visualize:
                color = [1.0, 0.0, 0.0]
                radius = 0.015

            vp_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            vp_sphere.translate(pos)
            vp_sphere.paint_uniform_color(color)
            geometries.append(vp_sphere)

        arrow_length = self.frustum_params.far * 0.2
        forward = quaternion_to_forward(selected_vp_orient)
        arrow_end = selected_vp_pos + forward * arrow_length

        arrow_points = np.array([selected_vp_pos, arrow_end])
        arrow_lines = np.array([[0, 1]])

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(arrow_points)
        line_set.lines = o3d.utility.Vector2iVector(arrow_lines)
        line_set.colors = o3d.utility.Vector3dVector([[1.0, 1.0, 0.0]])
        geometries.append(line_set)

        frustum = self.create_frustrum_lineset(selected_vp_pos, selected_vp_orient, self.frustum_params)
        frustum.paint_uniform_color([1.0, 1.0, 0.0])
        geometries.append(frustum)

        geometries.append(pcd_vis)

        logger.info("[Visualizer] Visualizing Candidate %d: %d visible points.",
                    index_to_visualize, len(visible_indices))
        o3d.visualization.draw_geometries(geometries,
                                          window_name=f"Visibility Visualization (Candidate {index_to_visualize})")

    def visualize_all_visibility_results(self,
                                         visibility_map,
                                         candidates=None):
        """Visualizes all viewpoints together in a single window with per-viewpoint colors."""
        if not visibility_map:
            logger.warning("[Visualizer] Visibility map is empty.")
            return

        geometries = []

        base_mesh = o3d.geometry.TriangleMesh(self.mesh)
        base_mesh.paint_uniform_color([0.8, 0.8, 0.8])
        base_mesh.compute_vertex_normals()
        geometries.append(base_mesh)

        num_vps = len(visibility_map)

        # Generate colors, skipping any too close to red (reserved for uncovered)
        raw_colors = plt.cm.tab20(np.linspace(0, 1, max(20, num_vps)))
        vp_colors = []
        for c in raw_colors:
            r, g, b = c[:3]
            if r > 0.7 and g < 0.3 and b < 0.3:
                continue
            vp_colors.append((r, g, b))

        # Compute uncovered points
        all_covered = set()
        for visible_indices in visibility_map.values():
            all_covered.update(visible_indices.tolist())
        uncovered_indices = set(range(self.num_points)) - all_covered

        if uncovered_indices:
            uncovered_pcd = o3d.geometry.PointCloud()
            uncovered_pcd.points = o3d.utility.Vector3dVector(self.target_points[list(uncovered_indices)])
            uncovered_pcd.paint_uniform_color([1.0, 0.0, 0.0])
            geometries.append(uncovered_pcd)

        for i in range(num_vps):
            pos = np.asarray(candidates[i][0])
            orientation = np.asarray(candidates[i][1])
            visible_indices = visibility_map[i]
            color = list(vp_colors[i % len(vp_colors)])

            # Sphere
            vp_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
            vp_sphere.translate(pos)
            vp_sphere.paint_uniform_color(color)
            vp_sphere.compute_vertex_normals()
            geometries.append(vp_sphere)

            # Frustum
            frustum = self.create_frustrum_lineset(pos, orientation, self.frustum_params)
            frustum.paint_uniform_color(color)
            geometries.append(frustum)

            # Direction arrow
            arrow_length = 0.5
            forward = quaternion_to_forward(orientation)
            arrow_end = pos + forward * arrow_length
            arrow = o3d.geometry.LineSet()
            arrow.points = o3d.utility.Vector3dVector(np.array([pos, arrow_end]))
            arrow.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
            arrow.paint_uniform_color(color)
            geometries.append(arrow)

            # Visible points
            if len(visible_indices) > 0:
                visible_pcd = o3d.geometry.PointCloud()
                visible_pcd.points = o3d.utility.Vector3dVector(self.target_points[visible_indices])
                visible_pcd.paint_uniform_color(color)
                geometries.append(visible_pcd)

        logger.info("[Visualizer] Showing all %d viewpoints together (%d/%d covered).",
                    num_vps, self.num_points - len(uncovered_indices), self.num_points)

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="All Viewpoints Visibility", width=1920, height=1080)

        render_option = vis.get_render_option()
        render_option.point_size = 4.0
        render_option.line_width = 2.0
        render_option.mesh_show_back_face = True

        for geom in geometries:
            vis.add_geometry(geom)

        logger.info("Press Q to close visualization")
        vis.run()
        vis.destroy_window()

    def _create_solution_geometries(self, result: OptimizationResult):
        """Helper to create the list of geometries for solution visualization."""
        geometries = []

        base_mesh = o3d.geometry.TriangleMesh(self.mesh)
        base_mesh.paint_uniform_color([0.8, 0.8, 0.8])
        base_mesh.compute_vertex_normals()
        geometries.append(base_mesh)

        all_covered = set()
        for vp in result.viewpoints:
            all_covered.update(vp.visible_indices)

        uncovered_indices = set(range(len(self.target_points))) - all_covered

        if uncovered_indices:
            uncovered_pcd = o3d.geometry.PointCloud()
            uncovered_pcd.points = o3d.utility.Vector3dVector(self.target_points[list(uncovered_indices)])
            uncovered_pcd.paint_uniform_color([1.0, 0.0, 0.0])
            geometries.append(uncovered_pcd)

        colors = plt.cm.tab20(np.linspace(0, 1, max(20, result.num_viewpoints)))

        for i, vp in enumerate(result.viewpoints):
            color = colors[i % len(colors)][:3]

            viewpoint_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
            viewpoint_sphere.translate(vp.position)
            viewpoint_sphere.paint_uniform_color(color)
            viewpoint_sphere.compute_vertex_normals()
            geometries.append(viewpoint_sphere)

            frustum = self.create_frustrum_lineset(vp.position, vp.orientation, self.frustum_params)
            frustum.paint_uniform_color(color)
            geometries.append(frustum)

            if len(vp.visible_indices) > 0:
                visible_pcd = o3d.geometry.PointCloud()
                visible_pcd.points = o3d.utility.Vector3dVector(self.target_points[vp.visible_indices])
                visible_pcd.paint_uniform_color(color)
                geometries.append(visible_pcd)

            arrow_length = 0.5
            forward = quaternion_to_forward(vp.orientation)
            arrow_end = vp.position + forward * arrow_length
            arrow_points = np.array([vp.position, arrow_end])
            arrow_lines = np.array([[0, 1]])
            arrow = o3d.geometry.LineSet()
            arrow.points = o3d.utility.Vector3dVector(arrow_points)
            arrow.lines = o3d.utility.Vector2iVector(arrow_lines)
            arrow.paint_uniform_color(color)
            geometries.append(arrow)

        return geometries

    def visualize_solution_pcd(self, result: OptimizationResult,
                               title: str = "Viewpoint Solution"):
        """Visualize the complete solution interactively."""
        logger.info("Visualizing solution: %s", title)
        logger.info("Total viewpoints: %d", result.num_viewpoints)
        logger.info("Coverage: %.2f%%", result.total_coverage * 100)

        geometries = self._create_solution_geometries(result)

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"{title} - {result.method_name}", width=1920, height=1080)

        render_option = vis.get_render_option()
        render_option.point_size = 4.0
        render_option.line_width = 2.0
        render_option.mesh_show_back_face = True

        for geom in geometries:
            vis.add_geometry(geom)

        logger.info("Press Q to close visualization")
        vis.run()
        vis.destroy_window()

    def save_solution_animation(self, result: OptimizationResult, filename: str, frames: int = 200):
        """Saves a GIF animation of the solution by orbiting the camera."""
        logger.info("Generating animation: %s", filename)

        geometries = self._create_solution_geometries(result)

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1600, height=1200, visible=True)

        for geom in geometries:
            vis.add_geometry(geom)

        render_opt = vis.get_render_option()
        render_opt.point_size = 4.0
        render_opt.line_width = 2.0
        render_opt.mesh_show_back_face = True

        ctr = vis.get_view_control()

        image_frames = []

        step_size = 10.0
        ctr.rotate(0.0, -500.0)

        logger.info("  - Rendering %d frames...", frames)
        for i in range(frames):
            ctr.rotate(step_size, 0.0)
            vis.poll_events()
            vis.update_renderer()

            img_array = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            img_uint8 = (img_array * 255).astype(np.uint8)
            image_frames.append(PIL.Image.fromarray(img_uint8))

        vis.destroy_window()

        if image_frames:
            image_frames[0].save(filename, save_all=True, append_images=image_frames[1:],
                                 duration=50, loop=0)
            logger.info("  - Saved GIF to %s", filename)

    def visualize_solution_triangles(self, result: OptimizationResult,
                                     mesh: o3d.geometry.TriangleMesh,
                                     title: str = "Triangle Visibility Solution"):
        """Visualize the complete solution with triangle-based visibility."""
        logger.info("Visualizing triangle-based solution: %s", title)
        logger.info("Total viewpoints: %d", result.num_viewpoints)

        num_triangles = len(np.asarray(mesh.triangles))
        geometries = []

        all_visible_triangles = set()
        for vp in result.viewpoints:
            all_visible_triangles.update(vp.visible_indices)

        triangle_colors = np.full((num_triangles, 3), [0.5, 0.5, 0.5], dtype=np.float64)

        viewpoint_colors = plt.cm.tab20(np.linspace(0, 1, max(20, result.num_viewpoints)))

        triangle_to_viewpoint = {}
        for i, vp in enumerate(result.viewpoints):
            for tri_idx in vp.visible_indices:
                if tri_idx not in triangle_to_viewpoint:
                    triangle_to_viewpoint[tri_idx] = i
                    triangle_colors[tri_idx] = viewpoint_colors[i % len(viewpoint_colors)][:3]

        colored_mesh = o3d.geometry.TriangleMesh(mesh)
        colored_mesh.vertex_colors = o3d.utility.Vector3dVector([])
        colored_mesh.triangle_colors = o3d.utility.Vector3dVector(triangle_colors)
        colored_mesh.compute_vertex_normals()
        geometries.append(colored_mesh)

        for i, vp in enumerate(result.viewpoints):
            color = viewpoint_colors[i % len(viewpoint_colors)][:3]

            viewpoint_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
            viewpoint_sphere.translate(vp.position)
            viewpoint_sphere.paint_uniform_color(color)
            viewpoint_sphere.compute_vertex_normals()
            geometries.append(viewpoint_sphere)

            frustum = self.create_frustrum_lineset(vp.position, vp.orientation, self.frustum_params)
            frustum.paint_uniform_color(color)
            geometries.append(frustum)

            arrow_length = 0.5
            forward = quaternion_to_forward(vp.orientation)
            arrow_end = vp.position + forward * arrow_length
            arrow_points = np.array([vp.position, arrow_end])
            arrow_lines = np.array([[0, 1]])
            arrow = o3d.geometry.LineSet()
            arrow.points = o3d.utility.Vector3dVector(arrow_points)
            arrow.lines = o3d.utility.Vector2iVector(arrow_lines)
            arrow.paint_uniform_color(color)
            geometries.append(arrow)

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"{title} - {result.method_name}", width=1920, height=1080)

        render_option = vis.get_render_option()
        render_option.mesh_show_back_face = True
        render_option.line_width = 2.0

        for geom in geometries:
            vis.add_geometry(geom)

        vis.run()
        vis.destroy_window()

    def visualize_visibility_results_triangles(self,
                                               visibility_map,
                                               mesh: o3d.geometry.TriangleMesh,
                                               candidate_index: int = 0,
                                               candidates=None):
        """Visualizes a specific candidate's triangle visibility."""
        if not visibility_map:
            logger.warning("[Visualizer] Visibility map is empty.")
            return

        index_to_visualize = candidate_index % len(visibility_map)
        selected_vp_pos = np.asarray(candidates[index_to_visualize][0])
        selected_vp_orient = np.asarray(candidates[index_to_visualize][1])
        visible_triangle_indices = visibility_map[index_to_visualize]

        geometries = []

        num_triangles = len(np.asarray(mesh.triangles))
        logger.info("[Visualizer] Visualizing Candidate %d: %d/%d visible triangles.",
                    index_to_visualize, len(visible_triangle_indices), num_triangles)

        triangle_colors = np.full((num_triangles, 3), [0.5, 0.5, 0.5], dtype=np.float64)
        triangle_colors[visible_triangle_indices] = [0.0, 1.0, 0.0]

        mesh_vis = o3d.geometry.TriangleMesh(mesh)
        mesh_vis.vertex_colors = o3d.utility.Vector3dVector([])
        mesh_vis.triangle_colors = o3d.utility.Vector3dVector(triangle_colors)
        mesh_vis.compute_vertex_normals()
        geometries.append(mesh_vis)

        for i, key in enumerate(candidate_list):
            pos = np.array(key[0])
            color = [0.0, 0.0, 1.0]
            radius = 0.01

            if i == index_to_visualize:
                color = [1.0, 0.0, 0.0]
                radius = 0.015

            vp_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            vp_sphere.translate(pos)
            vp_sphere.paint_uniform_color(color)
            geometries.append(vp_sphere)

        arrow_length = self.frustum_params.far * 0.2
        forward = quaternion_to_forward(selected_vp_orient)
        arrow_end = selected_vp_pos + forward * arrow_length

        arrow_points = np.array([selected_vp_pos, arrow_end])
        arrow_lines = np.array([[0, 1]])

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(arrow_points)
        line_set.lines = o3d.utility.Vector2iVector(arrow_lines)
        line_set.colors = o3d.utility.Vector3dVector([[1.0, 1.0, 0.0]])
        geometries.append(line_set)

        o3d.visualization.draw_geometries(geometries,
                                          window_name=f"Triangle Visibility (Candidate {index_to_visualize})")

    def visualize_free_space(self,
                             outside_positions: np.ndarray, outside_weights: np.ndarray,
                             inside_positions: np.ndarray, inside_weights: np.ndarray,
                             outside_colormap: str = "Reds",
                             inside_colormap: str = "Blues",
                             point_size: float = 3.0,
                             window_name: str = "Free-Space Sampling Heatmap"):
        """Visualize feasible sampling regions as colored point clouds over a wireframe mesh."""
        geometries = []

        # Wireframe mesh
        wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(self.mesh)
        wireframe.paint_uniform_color([0.7, 0.7, 0.7])
        geometries.append(wireframe)

        for positions, weights, cmap_name in [
            (outside_positions, outside_weights, outside_colormap),
            (inside_positions, inside_weights, inside_colormap),
        ]:
            if len(positions) == 0:
                continue
            # Normalize weights to [0, 1]
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

        logger.info("[Visualizer] Free-space heatmap: %d outside, %d inside points",
                    len(outside_positions), len(inside_positions))

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, width=1920, height=1080)

        render_option = vis.get_render_option()
        render_option.point_size = point_size
        render_option.line_width = 1.0

        for geom in geometries:
            vis.add_geometry(geom)

        logger.info("Press Q to close visualization")
        vis.run()
        vis.destroy_window()

    def visualize_resampling_progression(self, normal_vis_map, normal_candidates,
                                         targeted_vis_map, targeted_candidates,
                                         point_size=4.0):
        """Visualize resampling in three phases: normal VPs, targeted VPs one-by-one, combined."""
        BLUE = [0.0, 0.4, 0.8]
        ORANGE = [1.0, 0.5, 0.0]
        RED = [1.0, 0.0, 0.0]
        GREEN = [0.0, 1.0, 0.0]
        GRAY = [0.5, 0.5, 0.5]

        n_normal = len(normal_candidates)
        n_targeted = len(targeted_candidates)

        # Compute cumulative coverage from normal candidates
        normal_covered = set()
        for visible_indices in normal_vis_map.values():
            normal_covered.update(visible_indices.tolist())

        normal_coverage_pct = len(normal_covered) / self.num_points * 100

        # --- Phase 1: Normal candidates ---
        geometries = []

        wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(self.mesh)
        wireframe.paint_uniform_color([0.7, 0.7, 0.7])
        geometries.append(wireframe)

        # Generate per-VP blue shades
        raw_blues = plt.cm.Blues(np.linspace(0.4, 0.9, max(1, n_normal)))

        for i in range(n_normal):
            pos = np.asarray(normal_candidates[i][0])
            orientation = np.asarray(normal_candidates[i][1])
            color = list(raw_blues[i][:3])

            vp_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
            vp_sphere.translate(pos)
            vp_sphere.paint_uniform_color(color)
            vp_sphere.compute_vertex_normals()
            geometries.append(vp_sphere)

            frustum = self.create_frustrum_lineset(pos, orientation, self.frustum_params)
            frustum.paint_uniform_color(color)
            geometries.append(frustum)

            forward = quaternion_to_forward(orientation)
            arrow_end = pos + forward * 0.5
            arrow = o3d.geometry.LineSet()
            arrow.points = o3d.utility.Vector3dVector(np.array([pos, arrow_end]))
            arrow.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
            arrow.paint_uniform_color(color)
            geometries.append(arrow)

            visible_indices = normal_vis_map.get(i, np.array([], dtype=int))
            if len(visible_indices) > 0:
                vis_pcd = o3d.geometry.PointCloud()
                vis_pcd.points = o3d.utility.Vector3dVector(self.target_points[visible_indices])
                vis_pcd.paint_uniform_color(color)
                geometries.append(vis_pcd)

        uncovered_indices = set(range(self.num_points)) - normal_covered
        if uncovered_indices:
            uncov_pcd = o3d.geometry.PointCloud()
            uncov_pcd.points = o3d.utility.Vector3dVector(self.target_points[list(uncovered_indices)])
            uncov_pcd.paint_uniform_color(RED)
            geometries.append(uncov_pcd)

        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=f"Phase 1: Normal VPs ({n_normal} VPs, coverage={normal_coverage_pct:.1f}%)",
            width=1920, height=1080)
        render_option = vis.get_render_option()
        render_option.point_size = point_size
        render_option.line_width = 2.0
        render_option.mesh_show_back_face = True
        for geom in geometries:
            vis.add_geometry(geom)
        logger.info("Phase 1: %d normal VPs, coverage=%.1f%%. Press Q to continue.",
                     n_normal, normal_coverage_pct)
        vis.run()
        vis.destroy_window()

        # --- Phase 2: Targeted candidates one-by-one ---
        cumulative_covered = set(normal_covered)

        for t_idx in range(n_targeted):
            geometries = []

            wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(self.mesh)
            wireframe.paint_uniform_color([0.7, 0.7, 0.7])
            geometries.append(wireframe)

            # Normal candidates as small blue spheres (no frustums)
            for i in range(n_normal):
                pos = np.asarray(normal_candidates[i][0])
                vp_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
                vp_sphere.translate(pos)
                vp_sphere.paint_uniform_color(BLUE)
                vp_sphere.compute_vertex_normals()
                geometries.append(vp_sphere)

            # Previously-added targeted candidates as small orange spheres
            for j in range(t_idx):
                pos = np.asarray(targeted_candidates[j][0])
                vp_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
                vp_sphere.translate(pos)
                vp_sphere.paint_uniform_color(ORANGE)
                vp_sphere.compute_vertex_normals()
                geometries.append(vp_sphere)

            # Current targeted candidate as large orange sphere with frustum + arrow
            cur_pos = np.asarray(targeted_candidates[t_idx][0])
            cur_orient = np.asarray(targeted_candidates[t_idx][1])

            cur_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
            cur_sphere.translate(cur_pos)
            cur_sphere.paint_uniform_color(ORANGE)
            cur_sphere.compute_vertex_normals()
            geometries.append(cur_sphere)

            frustum = self.create_frustrum_lineset(cur_pos, cur_orient, self.frustum_params)
            frustum.paint_uniform_color(ORANGE)
            geometries.append(frustum)

            forward = quaternion_to_forward(cur_orient)
            arrow_end = cur_pos + forward * 0.5
            arrow = o3d.geometry.LineSet()
            arrow.points = o3d.utility.Vector3dVector(np.array([cur_pos, arrow_end]))
            arrow.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
            arrow.paint_uniform_color(ORANGE)
            geometries.append(arrow)

            # Determine newly covered points
            cur_visible = targeted_vis_map.get(t_idx, np.array([], dtype=int))
            newly_covered = set(cur_visible.tolist()) - cumulative_covered
            still_uncovered = set(range(self.num_points)) - cumulative_covered - newly_covered

            # Gray = already covered
            if cumulative_covered:
                gray_pcd = o3d.geometry.PointCloud()
                gray_pcd.points = o3d.utility.Vector3dVector(self.target_points[list(cumulative_covered)])
                gray_pcd.paint_uniform_color(GRAY)
                geometries.append(gray_pcd)

            # Green = newly covered by this VP
            if newly_covered:
                green_pcd = o3d.geometry.PointCloud()
                green_pcd.points = o3d.utility.Vector3dVector(self.target_points[list(newly_covered)])
                green_pcd.paint_uniform_color(GREEN)
                geometries.append(green_pcd)

            # Red = still uncovered
            if still_uncovered:
                red_pcd = o3d.geometry.PointCloud()
                red_pcd.points = o3d.utility.Vector3dVector(self.target_points[list(still_uncovered)])
                red_pcd.paint_uniform_color(RED)
                geometries.append(red_pcd)

            cumulative_covered.update(newly_covered)
            cur_coverage_pct = len(cumulative_covered) / self.num_points * 100

            vis = o3d.visualization.Visualizer()
            vis.create_window(
                window_name=f"Targeted VP {t_idx+1}/{n_targeted} "
                            f"(+{len(newly_covered)} pts, coverage={cur_coverage_pct:.1f}%)",
                width=1920, height=1080)
            render_option = vis.get_render_option()
            render_option.point_size = point_size
            render_option.line_width = 2.0
            render_option.mesh_show_back_face = True
            for geom in geometries:
                vis.add_geometry(geom)
            logger.info("Targeted VP %d/%d: +%d pts, coverage=%.1f%%. Press Q to continue.",
                         t_idx + 1, n_targeted, len(newly_covered), cur_coverage_pct)
            vis.run()
            vis.destroy_window()

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

        geometries = []

        wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(self.mesh)
        wireframe.paint_uniform_color([0.7, 0.7, 0.7])
        geometries.append(wireframe)

        # Generate per-VP colors
        raw_colors = plt.cm.tab20(np.linspace(0, 1, max(20, len(all_candidates))))
        vp_colors = []
        for c in raw_colors:
            r, g, b = c[:3]
            if r > 0.7 and g < 0.3 and b < 0.3:
                continue
            vp_colors.append((r, g, b))

        uncovered_final = set(range(self.num_points)) - total_covered
        if uncovered_final:
            uncov_pcd = o3d.geometry.PointCloud()
            uncov_pcd.points = o3d.utility.Vector3dVector(self.target_points[list(uncovered_final)])
            uncov_pcd.paint_uniform_color(RED)
            geometries.append(uncov_pcd)

        for i, candidate in enumerate(all_candidates):
            pos = np.asarray(candidate[0])
            orientation = np.asarray(candidate[1])
            is_targeted = i >= n_normal
            base_color = ORANGE if is_targeted else BLUE
            vis_color = list(vp_colors[i % len(vp_colors)])

            vp_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
            vp_sphere.translate(pos)
            vp_sphere.paint_uniform_color(base_color)
            vp_sphere.compute_vertex_normals()
            geometries.append(vp_sphere)

            frustum = self.create_frustrum_lineset(pos, orientation, self.frustum_params)
            frustum.paint_uniform_color(base_color)
            geometries.append(frustum)

            forward = quaternion_to_forward(orientation)
            arrow_end = pos + forward * 0.5
            arrow = o3d.geometry.LineSet()
            arrow.points = o3d.utility.Vector3dVector(np.array([pos, arrow_end]))
            arrow.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
            arrow.paint_uniform_color(base_color)
            geometries.append(arrow)

            visible_indices = all_vis_map.get(i, np.array([], dtype=int))
            if len(visible_indices) > 0:
                vis_pcd = o3d.geometry.PointCloud()
                vis_pcd.points = o3d.utility.Vector3dVector(self.target_points[visible_indices])
                vis_pcd.paint_uniform_color(vis_color)
                geometries.append(vis_pcd)

        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=f"Final: {n_normal} normal + {n_targeted} targeted "
                        f"(coverage={final_coverage_pct:.1f}%)",
            width=1920, height=1080)
        render_option = vis.get_render_option()
        render_option.point_size = point_size
        render_option.line_width = 2.0
        render_option.mesh_show_back_face = True
        for geom in geometries:
            vis.add_geometry(geom)
        logger.info("Final: %d normal + %d targeted VPs, coverage=%.1f%%. Press Q to close.",
                     n_normal, n_targeted, final_coverage_pct)
        vis.run()
        vis.destroy_window()

    def create_frustrum_lineset(self, viewpoint, orientation, params):
        """Create a LineSet representing the frustum volume."""
        half_angle_rad = (params.fov_y / 2.0)
        far_half_size = params.far * np.tan(half_angle_rad)

        forward, right, up = get_frustum_basis_from_quaternion(orientation)

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
            [1, 2], [2, 3], [3, 4], [4, 1],
            [0, 1], [0, 2], [0, 3], [0, 4]
        ]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.array(corners))
        line_set.lines = o3d.utility.Vector2iVector(np.array(lines))

        return line_set
