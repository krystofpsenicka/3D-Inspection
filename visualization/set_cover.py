"""Visualize set-cover optimization solutions."""

import logging

import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import open3d as o3d

from visibility.core.types import FrustumParams, OptimizationResult
from .frustum_utils import create_frustum_lineset

logger = logging.getLogger(__name__)


class SetCoverVisualizer:
    """Render set-cover / greedy optimization results.

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
        self.frustum_params = frustum_params

    # ------------------------------------------------------------------

    def _create_solution_geometries(self, result: OptimizationResult):
        """Build the list of Open3D geometries for a solution."""
        geometries = []

        base_mesh = o3d.geometry.TriangleMesh(self.mesh)
        base_mesh.paint_uniform_color([0.8, 0.8, 0.8])
        base_mesh.compute_vertex_normals()
        geometries.append(base_mesh)

        # Transfer GPU arrays to CPU
        positions_np = result.positions.get()
        orientations_np = result.orientations.get()
        vis_map_np = result.visibility_map.get()

        covered_mask = vis_map_np.any(axis=0)
        uncovered_indices = np.where(~covered_mask)[0]

        if len(uncovered_indices) > 0:
            uncovered_pcd = o3d.geometry.PointCloud()
            uncovered_pcd.points = o3d.utility.Vector3dVector(
                self.target_points[uncovered_indices])
            uncovered_pcd.paint_uniform_color([1.0, 0.0, 0.0])
            geometries.append(uncovered_pcd)

        colors = plt.cm.tab20(np.linspace(0, 1, max(20, result.num_viewpoints)))

        for i in range(result.num_viewpoints):
            color = colors[i % len(colors)][:3]
            pos = positions_np[i]
            rot = orientations_np[i]
            vis = np.where(vis_map_np[i])[0]

            viewpoint_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
            viewpoint_sphere.translate(pos)
            viewpoint_sphere.paint_uniform_color(color)
            viewpoint_sphere.compute_vertex_normals()
            geometries.append(viewpoint_sphere)

            frustum = create_frustum_lineset(pos, rot, self.frustum_params)
            frustum.paint_uniform_color(color)
            geometries.append(frustum)

            if len(vis) > 0:
                visible_pcd = o3d.geometry.PointCloud()
                visible_pcd.points = o3d.utility.Vector3dVector(
                    self.target_points[vis])
                visible_pcd.paint_uniform_color(color)
                geometries.append(visible_pcd)

            arrow_length = 0.5
            forward = rot[:, 0]
            arrow_end = pos + forward * arrow_length
            arrow = o3d.geometry.LineSet()
            arrow.points = o3d.utility.Vector3dVector(np.array([pos, arrow_end]))
            arrow.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
            arrow.paint_uniform_color(color)
            geometries.append(arrow)

        return geometries

    # ------------------------------------------------------------------

    def visualize_solution(self, result: OptimizationResult,
                           title: str = "Viewpoint Solution"):
        """Visualize the complete solution interactively."""
        logger.info("Visualizing solution: %s", title)
        logger.info("Total viewpoints: %d", result.num_viewpoints)
        logger.info("Coverage: %.2f%%", result.total_coverage * 100)

        geometries = self._create_solution_geometries(result)

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title, width=1920, height=1080)

        render_option = vis.get_render_option()
        render_option.point_size = 4.0
        render_option.line_width = 2.0
        render_option.mesh_show_back_face = True

        for geom in geometries:
            vis.add_geometry(geom)

        logger.info("Press Q to close visualization")
        vis.run()
        vis.destroy_window()

    # ------------------------------------------------------------------

    def save_animation(self, result: OptimizationResult,
                       filename: str, frames: int = 200):
        """Save a GIF animation of the solution by orbiting the camera."""
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
        for _ in range(frames):
            ctr.rotate(step_size, 0.0)
            vis.poll_events()
            vis.update_renderer()

            img_array = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            img_uint8 = (img_array * 255).astype(np.uint8)
            image_frames.append(PIL.Image.fromarray(img_uint8))

        vis.destroy_window()

        if image_frames:
            image_frames[0].save(filename, save_all=True,
                                 append_images=image_frames[1:],
                                 duration=50, loop=0)
            logger.info("  - Saved GIF to %s", filename)
