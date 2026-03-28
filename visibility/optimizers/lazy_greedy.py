"""
Lazy Greedy Set-Cover Optimizer (Minoux 1978).

Produces the exact same solution as the standard greedy optimizer but with
90-99% fewer visibility evaluations.  Uses a max-heap approach: compute initial
gains, pop top, re-evaluate, select if still best, otherwise push back.

Uses NumPy boolean masks for coverage tracking (fast on large point sets).
"""

from __future__ import annotations

import heapq
import logging
import numpy as np
from time import time as get_time
from typing import List

from ..core.types import ViewpointResult, OptimizationResult
from ..core.base import VisibilityQueryBase

logger = logging.getLogger(__name__)


class LazyGreedyOptimizer:
    """Lazy greedy set-cover optimizer (CPU).

    Exact same solution as standard greedy, but skips most re-evaluations
    via the Minoux lazy-evaluation trick: marginal gains can only decrease
    as points get covered, so a stale gain that is still the maximum must
    be fresh.
    """

    def __init__(self, visibility_query: VisibilityQueryBase):
        self.query = visibility_query
        self.num_points = visibility_query.num_points

    def optimize(
        self,
        positions,
        rotmats,
        target_coverage: float = 0.95,
        max_viewpoints: int = 50,
        precomputed_visibility_map=None,
    ) -> OptimizationResult:
        """Lazy greedy set-cover optimization.

        Parameters
        ----------
        positions : array (N, 3)
            Candidate positions (numpy or CuPy).
        rotmats : array (N, 3, 3)
            Candidate rotation matrices (numpy or CuPy).
        """
        n_cand = len(positions)
        logger.info("[LazyGreedy] Starting with %d candidates.", n_cand)
        start_time = get_time()

        if self.num_points == 0 or n_cand == 0:
            return OptimizationResult(
                "LazyGreedy_Empty", [], 0.0, 0, 0.0, [], 0.0, 0.0, 0.0,
            )

        # Pre-compute visibility for all candidates
        if precomputed_visibility_map is not None:
            vis_map = precomputed_visibility_map
            vis_comp_time = 0.0
        else:
            vis_map, vis_comp_time = self.query.compute_visibility_batch(
                positions, rotmats
            )

        # Build per-candidate boolean visibility masks for fast scoring
        vis_masks: List[np.ndarray] = [np.zeros(0)] * n_cand
        for i in range(n_cand):
            mask = np.zeros(self.num_points, dtype=np.bool_)
            vis_idx = vis_map[i]
            if len(vis_idx) > 0:
                mask[vis_idx] = True
            vis_masks[i] = mask

        # Initialise the max-heap with (negative gain, candidate index, generation)
        uncovered = np.ones(self.num_points, dtype=np.bool_)
        target_uncovered = int((1.0 - target_coverage) * self.num_points)

        heap: list = []
        for i in range(n_cand):
            gain = int(np.count_nonzero(vis_masks[i] & uncovered))
            heapq.heappush(heap, (-gain, i, 0))

        selected_viewpoints: List[ViewpointResult] = []
        generation = 0
        evaluations = 0
        optimization_start = get_time()

        while (
            int(uncovered.sum()) > target_uncovered
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
                fresh_gain = int(np.count_nonzero(vis_masks[idx] & uncovered))
                heapq.heappush(heap, (-fresh_gain, idx, generation))
            else:
                break

            if best_gain == 0:
                logger.info("  [LazyGreedy] No candidate provides new coverage.")
                break

            uncovered &= ~vis_masks[best_idx]
            total_covered = self.num_points - int(uncovered.sum())
            coverage = total_covered / self.num_points

            pos_i = positions[best_idx]
            rot_i = rotmats[best_idx]
            if hasattr(pos_i, 'get'):  # CuPy array
                pos_i = pos_i.get()
                rot_i = rot_i.get()

            # Store the *full* visibility set (all points visible from this
            # viewpoint), not just the incremental contribution.
            full_visible = vis_map[best_idx]
            selected_viewpoints.append(ViewpointResult(
                position=np.asarray(pos_i),
                orientation=np.asarray(rot_i),
                visible_indices=np.asarray(full_visible),
                coverage_score=len(full_visible) / self.num_points,
                computation_time=0.0,
            ))

            # Invalidate this candidate's mask so it can't be re-selected
            vis_masks[best_idx] = np.zeros(self.num_points, dtype=np.bool_)

            logger.info("  [LazyGreedy] VP %d: +%d pts, coverage=%.1f%%",
                        len(selected_viewpoints), best_gain, coverage * 100)

        optimization_time = get_time() - optimization_start
        total_time = get_time() - start_time
        total_covered = self.num_points - int(uncovered.sum())
        coverage = total_covered / self.num_points
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
