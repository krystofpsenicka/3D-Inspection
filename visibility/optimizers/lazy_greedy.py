"""
Lazy Greedy Set-Cover Optimizer (Minoux 1978).

Produces the exact same solution as the standard greedy optimizer but with
90-99% fewer visibility evaluations.  Uses a max-heap approach: compute initial
gains, pop top, re-evaluate, select if still best, otherwise push back.
"""

from __future__ import annotations

import heapq
import logging
import numpy as np
from time import time as get_time
from typing import Dict, List, Tuple

from ..core.types import ViewpointResult, OptimizationResult
from ..core.base import VisibilityQuery

logger = logging.getLogger(__name__)


class LazyGreedyOptimizer:
    """Lazy greedy set-cover optimizer (CPU).

    Exact same solution as standard greedy, but skips most re-evaluations
    via the Minoux lazy-evaluation trick: marginal gains can only decrease
    as points get covered, so a stale gain that is still the maximum must
    be fresh.
    """

    def __init__(self, visibility_query: VisibilityQuery):
        self.query = visibility_query
        self.num_points = visibility_query.num_points

    def optimize(
        self,
        candidates: List[Tuple[np.ndarray, np.ndarray]],
        target_coverage: float = 0.95,
        max_viewpoints: int = 50,
        precomputed_visibility_map=None,
    ) -> OptimizationResult:
        logger.info("[LazyGreedy] Starting with %d candidates.", len(candidates))
        start_time = get_time()

        if self.num_points == 0 or not candidates:
            return OptimizationResult(
                "LazyGreedy_Empty", [], 0.0, 0, 0.0, [], 0.0, 0.0, 0.0,
            )

        # Pre-compute visibility for all candidates
        if precomputed_visibility_map is not None:
            vis_map = precomputed_visibility_map
            vis_comp_time = 0.0
        else:
            vis_map, vis_comp_time = self.query.compute_visibility_for_all_candidates(
                candidates
            )
        n_cand = len(candidates)

        # Convert to sets for fast intersection
        vis_sets: List[set] = [
            set(vis_map[i]) for i in range(n_cand)
        ]

        # Initialise the max-heap with (negative gain, candidate index, generation)
        # Generation tracks whether the score is current; a candidate is only
        # valid if its generation equals the global generation counter.
        uncovered = set(range(self.num_points))
        target_uncovered = int((1.0 - target_coverage) * self.num_points)

        heap: list = []
        for i in range(n_cand):
            gain = len(vis_sets[i] & uncovered)
            heapq.heappush(heap, (-gain, i, 0))

        selected_viewpoints: List[ViewpointResult] = []
        generation = 0
        evaluations = 0
        optimization_start = get_time()

        while (
            len(uncovered) > target_uncovered
            and len(selected_viewpoints) < max_viewpoints
            and heap
        ):
            generation += 1
            while heap:
                neg_gain, idx, gen = heapq.heappop(heap)
                if gen == generation:
                    # Already re-evaluated this round; this is the best
                    best_idx = idx
                    best_gain = -neg_gain
                    break
                # Re-evaluate
                evaluations += 1
                fresh_gain = len(vis_sets[idx] & uncovered)
                heapq.heappush(heap, (-fresh_gain, idx, generation))
            else:
                break

            if best_gain == 0:
                logger.info("  [LazyGreedy] No candidate provides new coverage.")
                break

            newly_covered = vis_sets[best_idx] & uncovered
            uncovered -= newly_covered
            coverage = 1.0 - len(uncovered) / self.num_points

            best_vp, best_orient = candidates[best_idx]
            selected_viewpoints.append(ViewpointResult(
                position=np.asarray(best_vp),
                orientation=np.asarray(best_orient),
                visible_indices=np.array(list(newly_covered)),
                coverage_score=best_gain / self.num_points,
                computation_time=0.0,
            ))

            # Remove from future consideration
            vis_sets[best_idx] = set()

            logger.info("  [LazyGreedy] VP %d: +%d pts, coverage=%.1f%%",
                        len(selected_viewpoints), best_gain, coverage * 100)

        optimization_time = get_time() - optimization_start
        total_time = get_time() - start_time
        coverage = 1.0 - len(uncovered) / self.num_points
        redundancy = self.query.compute_redundancy(selected_viewpoints)

        total_evals = evaluations + n_cand  # initial + lazy re-evals
        logger.info("  [LazyGreedy] Done: %d evaluations (vs %d standard greedy).",
                    total_evals, n_cand * len(selected_viewpoints))

        return OptimizationResult(
            method_name="LazyGreedy",
            viewpoints=selected_viewpoints,
            total_coverage=coverage,
            num_viewpoints=len(selected_viewpoints),
            total_time=total_time,
            coverage_per_viewpoint=[
                vp.coverage_score for vp in selected_viewpoints
            ],
            redundancy=redundancy,
            visibility_computation_time=vis_comp_time,
            optimization_time=optimization_time,
            visibility_map=vis_map,
        )
