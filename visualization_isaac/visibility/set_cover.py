"""Visualize set-cover optimization solutions (Isaac Sim variant)."""

import logging

import numpy as np
import trimesh

from visibility.core.types import FrustumParams, OptimizationResult
from .frustum_utils import add_viewpoint_geometry
from .model import ModelVisualizer
from .._usd_primitives import create_points_prim
from .._helpers import generate_tab20_colors

logger = logging.getLogger(__name__)


class SetCoverVisualizer:
    """Stage-builder for set-cover / greedy optimization results.

    Parameters
    ----------
    mesh : trimesh.Trimesh.
    target_points : (N, 3) surface sample points.
    frustum_params : Camera frustum geometry.
    """

    def __init__(self, mesh: trimesh.Trimesh,
                 target_points: np.ndarray,
                 frustum_params: FrustumParams):
        self.mesh = mesh
        self.target_points = target_points
        self.frustum_params = frustum_params

    def add_solution(self, stage, base_path: str,
                     result: OptimizationResult) -> list[str]:
        """Add the complete solution geometry to *stage*.

        Parameters
        ----------
        stage : Usd.Stage
        base_path : Parent prim path.
        result : OptimizationResult (GPU arrays are transferred to CPU).

        Returns
        -------
        List of created prim paths.
        """
        paths: list[str] = []

        model_vis = ModelVisualizer(self.mesh)
        paths.append(model_vis.add_mesh(stage, f"{base_path}/mesh"))

        # Transfer GPU arrays to CPU
        positions_np = result.positions.get()
        rotations_np = result.rotations.get()
        vis_map_np = result.visibility_map.get()

        covered_mask = vis_map_np.any(axis=0)
        uncovered_indices = np.where(~covered_mask)[0]

        if len(uncovered_indices) > 0:
            paths.append(create_points_prim(
                stage, f"{base_path}/uncovered",
                self.target_points[uncovered_indices],
                colors=(1.0, 0.0, 0.0)))

        colors = generate_tab20_colors(result.num_viewpoints)

        for i in range(result.num_viewpoints):
            color = tuple(colors[i % len(colors)])
            pos = positions_np[i]
            rot = rotations_np[i]
            vis = np.where(vis_map_np[i])[0]

            paths += add_viewpoint_geometry(
                stage, f"{base_path}/vp_{i}",
                pos, rot, self.frustum_params, color)

            if len(vis) > 0:
                paths.append(create_points_prim(
                    stage, f"{base_path}/vp_{i}/visible",
                    self.target_points[vis], colors=color))

        logger.info("[SetCoverVisualizer] Added solution: %d VPs, coverage=%.2f%%",
                    result.num_viewpoints, result.total_coverage * 100)

        return paths
