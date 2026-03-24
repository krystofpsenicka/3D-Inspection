import logging
import numpy as np
import cupy as cp
from time import time as get_time
from typing import List, Tuple

from ..core.types import ViewpointResult, OptimizationResult
from ..core.base import VisibilityQueryBase

logger = logging.getLogger(__name__)


class GreedyOptimizerCuda:
    """GPU-accelerated greedy set-cover optimizer using CuPy.

    Builds a dense visibility matrix on GPU and performs each greedy
    iteration as a matrix-vector multiply + argmax.
    """

    def __init__(self, visibility_query: VisibilityQueryBase):
        self.query = visibility_query
        self.num_points = visibility_query.num_points
        logger.info("[GreedyOptimizerCuda] Initialized with %s.", type(visibility_query).__name__)

    def optimize(self, candidates: List[Tuple[np.ndarray, np.ndarray]],
                 target_coverage: float = 0.95,
                 max_viewpoints: int = 50) -> OptimizationResult:
        """GPU greedy set-cover optimization."""
        logger.info("[GreedyOptimizerCuda] Starting GPU greedy optimization with %d candidates.",
                    len(candidates))
        start_time = get_time()

        if self.num_points == 0 or not candidates:
            return OptimizationResult("GreedyCuda_Empty", [], 0.0, 0, 0.0, [], 0.0, 0.0, 0.0)

        # Pre-compute visibility for all candidates
        initial_visibility_map, vis_comp_time = \
            self.query.compute_visibility_batch(candidates)

        n_candidates = len(candidates)

        # Build dense visibility matrix on GPU (n_candidates x n_points, uint8)
        V = cp.zeros((n_candidates, self.num_points), dtype=cp.uint8)
        for i in range(n_candidates):
            visible_indices = initial_visibility_map[i]
            if len(visible_indices) > 0:
                V[i, visible_indices] = 1

        logger.info("[GreedyOptimizerCuda] Visibility matrix on GPU: %s (%.1f MB)",
                    V.shape, V.nbytes / 1e6)

        # Greedy loop on GPU
        uncovered = cp.ones(self.num_points, dtype=cp.uint8)
        candidate_mask = cp.ones(n_candidates, dtype=cp.bool_)
        total_covered = 0
        target_covered = int(target_coverage * self.num_points)

        selected_viewpoints: List[ViewpointResult] = []
        optimization_start_time = get_time()

        while total_covered < target_covered and len(selected_viewpoints) < max_viewpoints:
            if not cp.any(candidate_mask):
                logger.info("  [GreedyOptimizerCuda] No more candidates. Breaking.")
                break

            # GPU matmul: score each candidate by number of newly covered points
            # V @ uncovered gives (n_candidates,) — count of uncovered points each can see
            scores = V.astype(cp.float32) @ uncovered.astype(cp.float32)
            scores[~candidate_mask] = 0

            best = int(cp.argmax(scores))
            best_score = int(scores[best])

            if best_score == 0:
                logger.info("  [GreedyOptimizerCuda] No candidate provides new coverage. Stopping.")
                break

            # Update coverage
            newly_covered_mask = V[best] & uncovered
            uncovered &= ~V[best]
            candidate_mask[best] = False

            total_covered = self.num_points - int(cp.sum(uncovered))
            coverage = total_covered / self.num_points

            best_vp, best_orient = candidates[best]

            # Store the *full* visibility set, not just the incremental contribution
            full_visible_indices = cp.where(V[best])[0].get()
            selected_viewpoints.append(ViewpointResult(
                position=np.asarray(best_vp),
                orientation=np.asarray(best_orient),
                visible_indices=full_visible_indices,
                coverage_score=len(full_visible_indices) / self.num_points,
                computation_time=0.0,
            ))

            logger.info("  [GreedyOptimizerCuda] Selected VP %d: +%d points, Total coverage=%.1f%%",
                        len(selected_viewpoints), best_score, coverage * 100)

        optimization_time = get_time() - optimization_start_time
        total_time = get_time() - start_time
        coverage = total_covered / self.num_points
        redundancy = self.query.compute_redundancy(selected_viewpoints)

        return OptimizationResult(
            method_name="GreedyCuda",
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
