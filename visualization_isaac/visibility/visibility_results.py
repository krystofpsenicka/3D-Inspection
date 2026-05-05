"""Visualize per-viewpoint and batch visibility query results (Isaac Sim variant)."""

import logging

import numpy as np
import trimesh

from visibility.core.types import FrustumParams

from .._helpers import generate_tab20_colors
from .._usd_primitives import create_points_prim
from .frustum_utils import add_viewpoint_geometry
from .model import ModelVisualizer

logger = logging.getLogger(__name__)


class VisibilityVisualizer:
    def __init__(
        self, mesh: trimesh.Trimesh, target_points: np.ndarray, frustum_params: FrustumParams
    ):
        self.mesh = mesh
        self.target_points = target_points
        self.num_points = len(target_points)
        self.frustum_params = frustum_params

    def add_single(
        self,
        stage,
        base_path: str,
        visible_indices: np.ndarray,
        position: np.ndarray,
        rotation: np.ndarray,
    ) -> list[str]:
        paths: list[str] = []

        model_vis = ModelVisualizer(self.mesh)
        paths.append(model_vis.add_mesh(stage, f"{base_path}/mesh"))

        colors = np.full((self.num_points, 3), [0.3, 0.3, 0.3], dtype=np.float64)
        colors[visible_indices] = [0.0, 1.0, 0.0]
        paths.append(
            create_points_prim(stage, f"{base_path}/points", self.target_points, colors=colors)
        )

        paths += add_viewpoint_geometry(
            stage,
            f"{base_path}/viewpoint",
            position,
            rotation,
            self.frustum_params,
            color=(1.0, 1.0, 0.0),
            sphere_radius=0.015,
            arrow_length=self.frustum_params.far * 0.2,
        )

        return paths

    def add_all(self, stage, base_path: str, visibility_map, candidates) -> list[str]:
        """``visibility_map``: dict {vp_idx: visible_indices}. ``candidates``: list of (position, rotation)."""
        if not visibility_map:
            logger.warning("[VisibilityVisualizer] Visibility map is empty.")
            return []

        paths: list[str] = []

        model_vis = ModelVisualizer(self.mesh)
        paths.append(model_vis.add_mesh(stage, f"{base_path}/mesh"))

        num_vps = len(visibility_map)
        vp_colors = generate_tab20_colors(num_vps)

        all_covered: set[int] = set()
        for visible_indices in visibility_map.values():
            all_covered.update(visible_indices.tolist())
        uncovered_indices = set(range(self.num_points)) - all_covered

        if uncovered_indices:
            paths.append(
                create_points_prim(
                    stage,
                    f"{base_path}/uncovered",
                    self.target_points[list(uncovered_indices)],
                    colors=(1.0, 0.0, 0.0),
                )
            )

        for i in range(num_vps):
            pos = np.asarray(candidates[i][0])
            rotation = candidates[i][1]
            visible_indices = visibility_map[i]
            color = tuple(vp_colors[i % len(vp_colors)])

            paths += add_viewpoint_geometry(
                stage, f"{base_path}/vp_{i}", pos, rotation, self.frustum_params, color
            )

            if len(visible_indices) > 0:
                paths.append(
                    create_points_prim(
                        stage,
                        f"{base_path}/vp_{i}/visible",
                        self.target_points[visible_indices],
                        colors=color,
                    )
                )

        logger.info(
            "[VisibilityVisualizer] Added %d viewpoints (%d/%d covered).",
            num_vps,
            len(all_covered),
            self.num_points,
        )

        return paths
