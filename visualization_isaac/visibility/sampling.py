"""Visualize free-space sampling and resampling progression (Isaac Sim variant)."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import trimesh

from visibility.core.types import FrustumParams

from .._helpers import generate_tab20_colors
from .._usd_primitives import create_points_prim, create_sphere_prim
from .frustum_utils import add_viewpoint_geometry
from .model import ModelVisualizer

logger = logging.getLogger(__name__)


def _to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "get"):
        return x.get()
    return np.asarray(x)


class SamplingVisualizer:
    def __init__(
        self,
        mesh: trimesh.Trimesh,
        target_points: np.ndarray,
        normals: np.ndarray,
        frustum_params: FrustumParams,
    ):
        self.mesh = mesh
        self.target_points = target_points
        self.normals = normals
        self.num_points = len(target_points)
        self.frustum_params = frustum_params

    def add_free_space(
        self,
        stage,
        base_path: str,
        outside_positions: np.ndarray,
        outside_weights: np.ndarray,
        inside_positions: np.ndarray,
        inside_weights: np.ndarray,
        outside_colormap: str = "Reds",
        inside_colormap: str = "Blues",
        point_size: float = 0.02,
    ) -> list[str]:
        paths: list[str] = []

        model_vis = ModelVisualizer(self.mesh)
        paths.append(model_vis.add_mesh(stage, f"{base_path}/mesh"))

        for tag, positions, weights, cmap_name in [
            ("outside", outside_positions, outside_weights, outside_colormap),
            ("inside", inside_positions, inside_weights, inside_colormap),
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

            paths.append(
                create_points_prim(
                    stage, f"{base_path}/{tag}", positions, colors=colors, point_size=point_size
                )
            )

        logger.info(
            "[SamplingVisualizer] Free-space heatmap: %d outside, %d inside points",
            len(outside_positions),
            len(inside_positions),
        )

        return paths

    def add_resampling_phase1(
        self, stage, base_path: str, normal_vis_map, normal_candidates, point_size: float = 0.05
    ) -> list[str]:
        paths: list[str] = []

        model_vis = ModelVisualizer(self.mesh)
        paths.append(model_vis.add_mesh(stage, f"{base_path}/mesh"))

        n_normal = len(normal_candidates)
        raw_blues = plt.cm.Blues(np.linspace(0.4, 0.9, max(1, n_normal)))

        normal_covered: set[int] = set()
        for visible_indices in normal_vis_map.values():
            normal_covered.update(visible_indices.tolist())

        for i in range(n_normal):
            pos = np.asarray(normal_candidates[i][0])
            rotation = normal_candidates[i][1]
            color = tuple(raw_blues[i][:3])

            paths += add_viewpoint_geometry(
                stage, f"{base_path}/normal_vp_{i}", pos, rotation, self.frustum_params, color
            )

            visible_indices = normal_vis_map.get(i, np.array([], dtype=int))
            if len(visible_indices) > 0:
                paths.append(
                    create_points_prim(
                        stage,
                        f"{base_path}/normal_vp_{i}/visible",
                        self.target_points[visible_indices],
                        colors=color,
                        point_size=point_size,
                    )
                )

        uncovered_indices = set(range(self.num_points)) - normal_covered
        if uncovered_indices:
            paths.append(
                create_points_prim(
                    stage,
                    f"{base_path}/uncovered",
                    self.target_points[list(uncovered_indices)],
                    colors=(1.0, 0.0, 0.0),
                    point_size=point_size,
                )
            )

        coverage_pct = len(normal_covered) / self.num_points * 100
        logger.info("Phase 1: %d normal VPs, coverage=%.1f%%", n_normal, coverage_pct)

        return paths

    def add_resampling_phase2_step(
        self,
        stage,
        base_path: str,
        step_idx: int,
        normal_candidates,
        targeted_candidates,
        targeted_vis_map,
        cumulative_covered: set[int],
        point_size: float = 0.02,
    ) -> list[str]:
        """``cumulative_covered`` is the set already covered before this step; caller updates it after."""
        BLUE = (0.0, 0.4, 0.8)
        ORANGE = (1.0, 0.5, 0.0)
        RED = (1.0, 0.0, 0.0)
        GREEN = (0.0, 1.0, 0.0)
        GRAY = (0.5, 0.5, 0.5)

        paths: list[str] = []

        model_vis = ModelVisualizer(self.mesh)
        paths.append(model_vis.add_mesh(stage, f"{base_path}/mesh"))

        for i in range(len(normal_candidates)):
            pos = np.asarray(normal_candidates[i][0])
            paths.append(
                create_sphere_prim(
                    stage, f"{base_path}/normal_marker_{i}", position=pos, radius=0.2, color=BLUE
                )
            )

        for j in range(step_idx):
            pos = np.asarray(targeted_candidates[j][0])
            paths.append(
                create_sphere_prim(
                    stage,
                    f"{base_path}/targeted_marker_{j}",
                    position=pos,
                    radius=0.2,
                    color=ORANGE,
                )
            )

        cur_pos = np.asarray(targeted_candidates[step_idx][0])
        cur_orient = targeted_candidates[step_idx][1]
        paths += add_viewpoint_geometry(
            stage,
            f"{base_path}/current_vp",
            cur_pos,
            cur_orient,
            self.frustum_params,
            ORANGE,
            sphere_radius=0.2,
        )

        cur_visible = targeted_vis_map.get(step_idx, np.array([], dtype=int))
        newly_covered = set(cur_visible.tolist()) - cumulative_covered
        still_uncovered = set(range(self.num_points)) - cumulative_covered - newly_covered

        if cumulative_covered:
            paths.append(
                create_points_prim(
                    stage,
                    f"{base_path}/already_covered",
                    self.target_points[list(cumulative_covered)],
                    colors=GRAY,
                    point_size=point_size,
                )
            )

        if newly_covered:
            paths.append(
                create_points_prim(
                    stage,
                    f"{base_path}/newly_covered",
                    self.target_points[list(newly_covered)],
                    colors=GREEN,
                    point_size=point_size,
                )
            )

        if still_uncovered:
            paths.append(
                create_points_prim(
                    stage,
                    f"{base_path}/still_uncovered",
                    self.target_points[list(still_uncovered)],
                    colors=RED,
                    point_size=point_size,
                )
            )

        cur_coverage_pct = (len(cumulative_covered) + len(newly_covered)) / self.num_points * 100
        logger.info(
            "Targeted VP %d/%d: +%d pts, coverage=%.1f%%",
            step_idx + 1,
            len(targeted_candidates),
            len(newly_covered),
            cur_coverage_pct,
        )

        return paths

    def add_resampling_phase3(
        self,
        stage,
        base_path: str,
        normal_vis_map,
        normal_candidates,
        targeted_vis_map,
        targeted_candidates,
        point_size: float = 0.02,
    ) -> list[str]:
        BLUE = (0.0, 0.4, 0.8)
        ORANGE = (1.0, 0.5, 0.0)
        RED = (1.0, 0.0, 0.0)

        paths: list[str] = []

        model_vis = ModelVisualizer(self.mesh)
        paths.append(model_vis.add_mesh(stage, f"{base_path}/mesh"))

        n_normal = len(normal_candidates)
        all_candidates = list(normal_candidates) + list(targeted_candidates)
        all_vis_map: dict[int, np.ndarray] = {}
        for i, vis_indices in normal_vis_map.items():
            all_vis_map[i] = vis_indices
        for i, vis_indices in targeted_vis_map.items():
            all_vis_map[n_normal + i] = vis_indices

        total_covered: set[int] = set()
        for vis_indices in all_vis_map.values():
            total_covered.update(vis_indices.tolist())

        vp_colors = generate_tab20_colors(len(all_candidates))

        uncovered_final = set(range(self.num_points)) - total_covered
        if uncovered_final:
            paths.append(
                create_points_prim(
                    stage,
                    f"{base_path}/uncovered",
                    self.target_points[list(uncovered_final)],
                    colors=RED,
                    point_size=point_size,
                )
            )

        for i, candidate in enumerate(all_candidates):
            pos = np.asarray(candidate[0])
            rotation = candidate[1]
            is_targeted = i >= n_normal
            base_color = ORANGE if is_targeted else BLUE
            vis_color = tuple(vp_colors[i % len(vp_colors)])

            paths += add_viewpoint_geometry(
                stage, f"{base_path}/vp_{i}", pos, rotation, self.frustum_params, base_color
            )

            visible_indices = all_vis_map.get(i, np.array([], dtype=int))
            if len(visible_indices) > 0:
                paths.append(
                    create_points_prim(
                        stage,
                        f"{base_path}/vp_{i}/visible",
                        self.target_points[visible_indices],
                        colors=vis_color,
                        point_size=point_size,
                    )
                )

        final_coverage_pct = len(total_covered) / self.num_points * 100
        logger.info(
            "Final: %d normal + %d targeted VPs, coverage=%.1f%%",
            n_normal,
            len(targeted_candidates),
            final_coverage_pct,
        )

        return paths

    def add_candidates(
        self,
        stage,
        base_path: str,
        positions: np.ndarray,
        rotmats: np.ndarray,
        visibility_map: np.ndarray | None = None,
        sphere_radius: float = 0.10,
        point_size: float = 0.02,
        candidate_color: tuple = (0.2, 0.55, 1.0),
    ) -> list[str]:
        """Mesh + per-candidate frustum (and optional visibility colouring). Accepts NumPy or CuPy."""
        paths: list[str] = []
        model_vis = ModelVisualizer(self.mesh, self.target_points)
        paths.append(model_vis.add_mesh(stage, f"{base_path}/mesh"))

        positions = _to_numpy(positions)
        rotmats = _to_numpy(rotmats)
        n = len(positions)
        if n == 0:
            logger.warning("[SamplingVisualizer] add_candidates: no candidates supplied.")
            return paths

        colors = generate_tab20_colors(max(n, 1)) if visibility_map is not None else None

        for i, (pos, R) in enumerate(zip(positions, rotmats, strict=False)):
            color = tuple(colors[i % len(colors)]) if colors is not None else candidate_color
            paths += add_viewpoint_geometry(
                stage,
                f"{base_path}/cand_{i}",
                pos,
                R,
                self.frustum_params,
                color=color,
                sphere_radius=sphere_radius,
            )

        if visibility_map is not None:
            V = _to_numpy(visibility_map).astype(bool)
            first_cover = np.full(self.num_points, -1, dtype=np.int64)
            if V.size:
                covers = np.argmax(V, axis=0)
                any_cover = V.any(axis=0)
                first_cover[any_cover] = covers[any_cover]
            pt_colors = np.full((self.num_points, 3), [0.4, 0.4, 0.4], dtype=np.float64)
            uncovered = first_cover == -1
            pt_colors[uncovered] = [1.0, 0.0, 0.0]
            for i, c in enumerate(colors[:n]):
                mask = first_cover == i
                if np.any(mask):
                    pt_colors[mask] = list(c)
            paths.append(
                create_points_prim(
                    stage,
                    f"{base_path}/points",
                    self.target_points,
                    colors=pt_colors,
                    point_size=point_size,
                )
            )
            covered = int((~uncovered).sum())
            logger.info(
                "[SamplingVisualizer] add_candidates: %d candidates, %d/%d points covered.",
                n,
                covered,
                self.num_points,
            )
        else:
            paths.append(
                create_points_prim(
                    stage,
                    f"{base_path}/points",
                    self.target_points,
                    colors=(0.5, 0.5, 0.5),
                    point_size=point_size,
                )
            )
            logger.info("[SamplingVisualizer] add_candidates: %d candidates (no visibility).", n)

        return paths
