import logging
import numpy as np
from time import time as get_time
from typing import List, Tuple

from ..core.types import ViewpointResult, OptimizationResult
from ..core.base import VisibilityQuery

logger = logging.getLogger(__name__)


class GreedyOptimizer:
    """
    Implements a standard greedy set-cover optimization.
    Works with any VisibilityQuery implementation.
    """

    def __init__(self, visibility_query: VisibilityQuery):
        self.query = visibility_query
        self.num_points = visibility_query.num_points
        logger.info("[GreedyOptimizer] Initialized with %s.", type(visibility_query).__name__)

    def optimize(self, candidates: List[Tuple[np.ndarray, np.ndarray]],
                 target_coverage: float = 0.95,
                 max_viewpoints: int = 50,
                 precomputed_visibility_map=None) -> OptimizationResult:
        """Standard greedy set-cover optimization."""
        logger.info("[GreedyOptimizer] Starting standard greedy optimization with %d candidates.", len(candidates))
        start_time = get_time()

        if self.num_points == 0 or not candidates:
            return OptimizationResult("Greedy_Empty", [], 0.0, 0, 0.0, [], 0.0, 0.0, 0.0)

        if precomputed_visibility_map is not None:
            initial_visibility_map = precomputed_visibility_map
            vis_comp_time = 0.0
        else:
            initial_visibility_map, vis_comp_time = self.query.compute_visibility_for_all_candidates(candidates)

        uncovered = set(range(self.num_points))
        selected_viewpoints: List[ViewpointResult] = []
        remaining_candidate_indices = set(initial_visibility_map.keys())
        target_uncovered_count = int((1.0 - target_coverage) * self.num_points)

        optimization_start_time = get_time()

        while len(uncovered) > target_uncovered_count and len(selected_viewpoints) < max_viewpoints:

            if not remaining_candidate_indices:
                logger.info("  [GreedyOptimizer] No more candidates to check. Breaking.")
                break

            best_candidate_idx = -1
            best_newly_covered = set()
            best_score = 0

            for candidate_idx in remaining_candidate_indices:
                visible_indices = initial_visibility_map[candidate_idx]

                newly_covered = set(visible_indices) & uncovered
                score = len(newly_covered)

                if score > best_score:
                    best_score = score
                    best_candidate_idx = candidate_idx
                    best_newly_covered = newly_covered

            if best_candidate_idx == -1 or best_score == 0:
                logger.info("  [GreedyOptimizer] No candidate provides new coverage. Stopping.")
                break

            best_vp, best_orient = candidates[best_candidate_idx]

            uncovered -= best_newly_covered
            coverage = 1.0 - len(uncovered) / self.num_points

            selected_viewpoints.append(ViewpointResult(
                position=np.asarray(best_vp),
                orientation=np.asarray(best_orient),
                visible_indices=np.array(list(best_newly_covered)),
                coverage_score=best_score / self.num_points,
                computation_time=0.0,
            ))

            remaining_candidate_indices.remove(best_candidate_idx)

            logger.info("  [GreedyOptimizer] Selected VP %d: +%d points, Total coverage=%.1f%%",
                        len(selected_viewpoints), best_score, coverage * 100)

        optimization_time = get_time() - optimization_start_time
        total_time = get_time() - start_time
        coverage = 1.0 - len(uncovered) / self.num_points
        redundancy = self.query.compute_redundancy(selected_viewpoints)

        return OptimizationResult(
            method_name="Greedy_Standard",
            viewpoints=selected_viewpoints,
            total_coverage=coverage,
            num_viewpoints=len(selected_viewpoints),
            total_time=total_time,
            coverage_per_viewpoint=[vp.coverage_score for vp in selected_viewpoints],
            redundancy=redundancy,
            visibility_computation_time=vis_comp_time,
            optimization_time=optimization_time,
            visibility_map=initial_visibility_map,
        )
