import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import PIL.Image
from typing import Dict, Tuple

from .core.types import FrustumParams, OptimizationResult
from .core.base import get_frustum_basis


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
        print("[Visualizer] Initialized visualization module.")

    def visualize_normals(self, normal_scale: float = 0.05):
        """Visualizes the mesh, the target points, and their computed normals."""
        print("\n[Visualizer] Visualizing target points and normals...")

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
                                     visibility_map: Dict[Tuple[Tuple[float, ...], Tuple[float, ...]], np.ndarray],
                                     candidate_index: int = 0):
        """Visualizes a specific candidate's visibility against the mesh and target points."""
        candidate_list = list(visibility_map.keys())
        if not candidate_list:
            print("[Visualizer] Error: Visibility map is empty.")
            return

        index_to_visualize = candidate_index % len(candidate_list)
        selected_key = candidate_list[index_to_visualize]
        selected_vp_pos = np.array(selected_key[0])
        selected_vp_dir = np.array(selected_key[1])
        visible_indices = visibility_map[selected_key]

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
        arrow_end = selected_vp_pos + selected_vp_dir * arrow_length

        arrow_points = np.array([selected_vp_pos, arrow_end])
        arrow_lines = np.array([[0, 1]])

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(arrow_points)
        line_set.lines = o3d.utility.Vector2iVector(arrow_lines)
        line_set.colors = o3d.utility.Vector3dVector([[1.0, 1.0, 0.0]])
        geometries.append(line_set)

        frustum = self.create_frustrum_lineset(selected_vp_pos, selected_vp_dir, self.frustum_params)
        frustum.paint_uniform_color([1.0, 1.0, 0.0])
        geometries.append(frustum)

        geometries.append(pcd_vis)

        print(f"\n[Visualizer] Visualizing Candidate {index_to_visualize}: {len(visible_indices)} visible points.")
        o3d.visualization.draw_geometries(geometries,
                                          window_name=f"Visibility Visualization (Candidate {index_to_visualize})")

    def visualize_all_visibility_results(self,
                                         visibility_map: Dict[Tuple[Tuple[float, ...], Tuple[float, ...]], np.ndarray]):
        """Visualizes all viewpoints together in a single window with per-viewpoint colors."""
        if not visibility_map:
            print("[Visualizer] Error: Visibility map is empty.")
            return

        geometries = []

        base_mesh = o3d.geometry.TriangleMesh(self.mesh)
        base_mesh.paint_uniform_color([0.8, 0.8, 0.8])
        base_mesh.compute_vertex_normals()
        geometries.append(base_mesh)

        candidate_list = list(visibility_map.keys())
        num_vps = len(candidate_list)

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

        for i, key in enumerate(candidate_list):
            pos = np.array(key[0])
            direction = np.array(key[1])
            visible_indices = visibility_map[key]
            color = list(vp_colors[i % len(vp_colors)])

            # Sphere
            vp_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
            vp_sphere.translate(pos)
            vp_sphere.paint_uniform_color(color)
            vp_sphere.compute_vertex_normals()
            geometries.append(vp_sphere)

            # Frustum
            frustum = self.create_frustrum_lineset(pos, direction, self.frustum_params)
            frustum.paint_uniform_color(color)
            geometries.append(frustum)

            # Direction arrow
            arrow_length = 0.5
            arrow_end = pos + direction * arrow_length
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

        print(f"\n[Visualizer] Showing all {num_vps} viewpoints together "
              f"({self.num_points - len(uncovered_indices)}/{self.num_points} covered).")

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="All Viewpoints Visibility", width=1920, height=1080)

        render_option = vis.get_render_option()
        render_option.point_size = 4.0
        render_option.line_width = 2.0
        render_option.mesh_show_back_face = True

        for geom in geometries:
            vis.add_geometry(geom)

        print("Press Q to close visualization")
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

            frustum = self.create_frustrum_lineset(vp.position, vp.direction, self.frustum_params)
            frustum.paint_uniform_color(color)
            geometries.append(frustum)

            if len(vp.visible_indices) > 0:
                visible_pcd = o3d.geometry.PointCloud()
                visible_pcd.points = o3d.utility.Vector3dVector(self.target_points[vp.visible_indices])
                visible_pcd.paint_uniform_color(color)
                geometries.append(visible_pcd)

            arrow_length = 0.5
            arrow_end = vp.position + vp.direction * arrow_length
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
        print(f"\nVisualizing solution: {title}")
        print(f"Total viewpoints: {result.num_viewpoints}")
        print(f"Coverage: {result.total_coverage * 100:.2f}%")

        geometries = self._create_solution_geometries(result)

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"{title} - {result.method_name}", width=1920, height=1080)

        render_option = vis.get_render_option()
        render_option.point_size = 4.0
        render_option.line_width = 2.0
        render_option.mesh_show_back_face = True

        for geom in geometries:
            vis.add_geometry(geom)

        print(f"\nPress Q to close visualization")
        vis.run()
        vis.destroy_window()

    def save_solution_animation(self, result: OptimizationResult, filename: str, frames: int = 200):
        """Saves a GIF animation of the solution by orbiting the camera."""
        print(f"Generating animation: {filename}")

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

        print(f"  - Rendering {frames} frames...")
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
            print(f"  - Saved GIF to {filename}")

    def visualize_solution_triangles(self, result: OptimizationResult,
                                     mesh: o3d.geometry.TriangleMesh,
                                     title: str = "Triangle Visibility Solution"):
        """Visualize the complete solution with triangle-based visibility."""
        print(f"\nVisualizing triangle-based solution: {title}")
        print(f"Total viewpoints: {result.num_viewpoints}")

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

            frustum = self.create_frustrum_lineset(vp.position, vp.direction, self.frustum_params)
            frustum.paint_uniform_color(color)
            geometries.append(frustum)

            arrow_length = 0.5
            arrow_end = vp.position + vp.direction * arrow_length
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
                                               visibility_map: Dict[Tuple[Tuple[float, ...], Tuple[float, ...]], np.ndarray],
                                               mesh: o3d.geometry.TriangleMesh,
                                               candidate_index: int = 0):
        """Visualizes a specific candidate's triangle visibility."""
        candidate_list = list(visibility_map.keys())
        if not candidate_list:
            print("[Visualizer] Error: Visibility map is empty.")
            return

        index_to_visualize = candidate_index % len(candidate_list)
        selected_key = candidate_list[index_to_visualize]
        selected_vp_pos = np.array(selected_key[0])
        selected_vp_dir = np.array(selected_key[1])
        visible_triangle_indices = visibility_map[selected_key]

        geometries = []

        num_triangles = len(np.asarray(mesh.triangles))
        print(f"\n[Visualizer] Visualizing Candidate {index_to_visualize}: "
              f"{len(visible_triangle_indices)}/{num_triangles} visible triangles.")

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
        arrow_end = selected_vp_pos + selected_vp_dir * arrow_length

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
                             point_size: float = 3.0):
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

        print(f"[Visualizer] Free-space heatmap: "
              f"{len(outside_positions)} outside, {len(inside_positions)} inside points")

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Free-Space Sampling Heatmap", width=1920, height=1080)

        render_option = vis.get_render_option()
        render_option.point_size = point_size
        render_option.line_width = 1.0

        for geom in geometries:
            vis.add_geometry(geom)

        print("Press Q to close visualization")
        vis.run()
        vis.destroy_window()

    def create_frustrum_lineset(self, viewpoint, direction, params):
        """Create a LineSet representing the frustum volume."""
        half_angle_rad = (params.fov_y / 2.0)
        far_half_size = params.far * np.tan(half_angle_rad)

        right, up = get_frustum_basis(direction)

        far_center = viewpoint + direction * params.far

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
