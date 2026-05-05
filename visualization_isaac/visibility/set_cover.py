"""Visualize set-cover optimization solutions (Isaac Sim variant).

Three-phase API: candidates -> visibility -> selected. ``add_solution`` is an alias for
``add_phase_selected`` (legacy single-call form).
"""

from __future__ import annotations

import logging

import numpy as np
import trimesh

from visibility.core.types import FrustumParams, OptimizationResult

from .._helpers import generate_tab20_colors
from .._usd_primitives import create_points_prim
from .frustum_utils import add_viewpoint_geometry
from .model import ModelVisualizer

logger = logging.getLogger(__name__)


def _to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "get"):
        return x.get()
    return np.asarray(x)


class SetCoverVisualizer:
    def __init__(
        self, mesh: trimesh.Trimesh, target_points: np.ndarray, frustum_params: FrustumParams
    ):
        self.mesh = mesh
        self.target_points = target_points
        self.num_points = len(target_points)
        self.frustum_params = frustum_params

    def add_phase_candidates(
        self,
        stage,
        base_path: str,
        all_positions: np.ndarray,
        all_rotmats: np.ndarray,
        sphere_radius: float = 0.10,
        candidate_color: tuple = (0.85, 0.85, 0.2),
    ) -> list[str]:
        """Mesh + sphere/frustum per candidate VP. Surface points rendered grey for context."""
        paths: list[str] = []
        model_vis = ModelVisualizer(self.mesh, self.target_points)
        paths.append(model_vis.add_mesh(stage, f"{base_path}/mesh"))
        paths.append(model_vis.add_points(stage, f"{base_path}/points", color=(0.5, 0.5, 0.5)))

        positions = _to_numpy(all_positions)
        rotmats = _to_numpy(all_rotmats)
        for i, (pos, R) in enumerate(zip(positions, rotmats, strict=False)):
            paths += add_viewpoint_geometry(
                stage,
                f"{base_path}/cand_{i}",
                pos,
                R,
                self.frustum_params,
                color=candidate_color,
                sphere_radius=sphere_radius,
            )
        logger.info("[SetCoverVisualizer] phase candidates: %d frustums", len(positions))
        return paths

    def add_phase_visibility(
        self,
        stage,
        base_path: str,
        all_positions: np.ndarray,
        all_rotmats: np.ndarray,
        full_visibility_map: np.ndarray,
        point_size: float = 0.02,
    ) -> list[str]:
        """Mesh + every candidate frustum + visibility-coloured points (first-covering VP, tab20)."""
        paths: list[str] = []
        model_vis = ModelVisualizer(self.mesh)
        paths.append(model_vis.add_mesh(stage, f"{base_path}/mesh"))

        positions = _to_numpy(all_positions)
        rotmats = _to_numpy(all_rotmats)
        V = _to_numpy(full_visibility_map).astype(bool)

        n_cand = len(positions)
        colors = generate_tab20_colors(max(n_cand, 1))

        for i, (pos, R) in enumerate(zip(positions, rotmats, strict=False)):
            color = tuple(colors[i % len(colors)])
            paths += add_viewpoint_geometry(
                stage,
                f"{base_path}/cand_{i}",
                pos,
                R,
                self.frustum_params,
                color=color,
                sphere_radius=0.08,
            )

        first_cover = np.full(self.num_points, -1, dtype=np.int64)
        if V.size:
            covers = np.argmax(V, axis=0)
            any_cover = V.any(axis=0)
            first_cover[any_cover] = covers[any_cover]

        pt_colors = np.full((self.num_points, 3), [0.4, 0.4, 0.4], dtype=np.float64)
        uncovered_mask = first_cover == -1
        pt_colors[uncovered_mask] = [1.0, 0.0, 0.0]
        for i, c in enumerate(colors[:n_cand]):
            mask = first_cover == i
            if not np.any(mask):
                continue
            pt_colors[mask] = list(c)

        paths.append(
            create_points_prim(
                stage, f"{base_path}/points", self.target_points, colors=pt_colors,
                point_size=point_size,
            )
        )

        covered = int((~uncovered_mask).sum())
        logger.info(
            "[SetCoverVisualizer] phase visibility: %d / %d points covered (%d candidates).",
            covered,
            self.num_points,
            n_cand,
        )
        return paths

    def add_phase_selected(
        self,
        stage,
        base_path: str,
        result: OptimizationResult,
    ) -> list[str]:
        paths: list[str] = []
        model_vis = ModelVisualizer(self.mesh)
        paths.append(model_vis.add_mesh(stage, f"{base_path}/mesh"))

        positions_np = _to_numpy(result.positions)
        rotations_np = _to_numpy(result.rotations)
        vis_map_np = _to_numpy(result.visibility_map).astype(bool)

        covered_mask = vis_map_np.any(axis=0) if vis_map_np.size else np.zeros(self.num_points, dtype=bool)
        uncovered_indices = np.where(~covered_mask)[0]

        if len(uncovered_indices) > 0:
            paths.append(
                create_points_prim(
                    stage,
                    f"{base_path}/uncovered",
                    self.target_points[uncovered_indices],
                    colors=(1.0, 0.0, 0.0),
                )
            )

        colors = generate_tab20_colors(max(result.num_viewpoints, 1))

        for i in range(result.num_viewpoints):
            color = tuple(colors[i % len(colors)])
            pos = positions_np[i]
            rot = rotations_np[i]
            vis = np.where(vis_map_np[i])[0]

            paths += add_viewpoint_geometry(
                stage, f"{base_path}/vp_{i}", pos, rot, self.frustum_params, color
            )
            if len(vis) > 0:
                paths.append(
                    create_points_prim(
                        stage,
                        f"{base_path}/vp_{i}/visible",
                        self.target_points[vis],
                        colors=color,
                    )
                )

        logger.info(
            "[SetCoverVisualizer] phase selected: %d VPs, coverage=%.2f%%",
            result.num_viewpoints,
            result.total_coverage * 100,
        )
        return paths

    def add_solution(self, stage, base_path: str, result: OptimizationResult) -> list[str]:
        return self.add_phase_selected(stage, base_path, result)
