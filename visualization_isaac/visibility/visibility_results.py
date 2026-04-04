"""Visualize per-viewpoint and batch visibility query results (Isaac Sim variant)."""

import logging

import numpy as np
import trimesh

from visibility.core.types import FrustumParams
from .frustum_utils import add_viewpoint_geometry
from .model import ModelVisualizer
from .._usd_primitives import create_points_prim
from .._helpers import generate_tab20_colors

logger = logging.getLogger(__name__)


class VisibilityVisualizer:
    """Stage-builder for visibility results over a mesh and target point cloud.

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
        self.num_points = len(target_points)
        self.frustum_params = frustum_params

    # ------------------------------------------------------------------
    # Single viewpoint
    # ------------------------------------------------------------------

    def add_single(self, stage, base_path: str,
                   visible_indices: np.ndarray,
                   position: np.ndarray,
                   rotation: np.ndarray) -> list[str]:
        """Add geometry for a single viewpoint's visibility.

        Parameters
        ----------
        stage : Usd.Stage
        base_path : Parent prim path.
        visible_indices : 1-D int array of visible point indices.
        position : (3,) viewpoint position.
        rotation : (3, 3) rotation matrix.

        Returns
        -------
        List of created prim paths.
        """
        paths: list[str] = []

        model_vis = ModelVisualizer(self.mesh)
        paths.append(model_vis.add_mesh(stage, f"{base_path}/mesh"))

        # Point cloud coloured by visibility
        colors = np.full((self.num_points, 3), [0.3, 0.3, 0.3], dtype=np.float64)
        colors[visible_indices] = [0.0, 1.0, 0.0]
        paths.append(create_points_prim(
            stage, f"{base_path}/points", self.target_points, colors=colors))

        # Viewpoint sphere + frustum + arrow
        paths += add_viewpoint_geometry(
            stage, f"{base_path}/viewpoint",
            position, rotation, self.frustum_params,
            color=(1.0, 1.0, 0.0),
            sphere_radius=0.015,
            arrow_length=self.frustum_params.far * 0.2,
        )

        return paths

    # ------------------------------------------------------------------
    # All viewpoints
    # ------------------------------------------------------------------

    def add_all(self, stage, base_path: str,
                visibility_map, candidates) -> list[str]:
        """Add geometry for all viewpoints with per-viewpoint colours.

        Parameters
        ----------
        stage : Usd.Stage
        base_path : Parent prim path.
        visibility_map : dict {int: np.ndarray} mapping VP index -> visible indices.
        candidates : list of (position, rotation) tuples.

        Returns
        -------
        List of created prim paths.
        """
        if not visibility_map:
            logger.warning("[VisibilityVisualizer] Visibility map is empty.")
            return []

        paths: list[str] = []

        model_vis = ModelVisualizer(self.mesh)
        paths.append(model_vis.add_mesh(stage, f"{base_path}/mesh"))

        num_vps = len(visibility_map)
        vp_colors = generate_tab20_colors(num_vps)

        # Uncovered points
        all_covered: set[int] = set()
        for visible_indices in visibility_map.values():
            all_covered.update(visible_indices.tolist())
        uncovered_indices = set(range(self.num_points)) - all_covered

        if uncovered_indices:
            paths.append(create_points_prim(
                stage, f"{base_path}/uncovered",
                self.target_points[list(uncovered_indices)],
                colors=(1.0, 0.0, 0.0)))

        for i in range(num_vps):
            pos = np.asarray(candidates[i][0])
            rotation = candidates[i][1]
            visible_indices = visibility_map[i]
            color = tuple(vp_colors[i % len(vp_colors)])

            paths += add_viewpoint_geometry(
                stage, f"{base_path}/vp_{i}",
                pos, rotation, self.frustum_params, color)

            if len(visible_indices) > 0:
                paths.append(create_points_prim(
                    stage, f"{base_path}/vp_{i}/visible",
                    self.target_points[visible_indices], colors=color))

        logger.info("[VisibilityVisualizer] Added %d viewpoints (%d/%d covered).",
                    num_vps, len(all_covered), self.num_points)

        return paths
