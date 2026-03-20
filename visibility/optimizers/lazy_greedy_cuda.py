"""
GPU-accelerated Lazy Greedy Set-Cover Optimizer (Minoux 1978).

Same lazy evaluation trick as the CPU version but with CuPy for the
dense visibility matrix operations.
"""

from __future__ import annotations

import logging
import numpy as np
import cupy as cp
from time import time as get_time
from typing import List, Tuple

from ..core.types import ViewpointResult, OptimizationResult
from ..core.base import VisibilityQuery

logger = logging.getLogger(__name__)


class LazyGreedyOptimizerCuda:
    """GPU-accelerated lazy greedy set-cover optimizer."""

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
        logger.info("[LazyGreedyCuda] Starting with %d candidates.", len(candidates))
        start_time = get_time()

        if self.num_points == 0 or not candidates:
            return OptimizationResult(
                "LazyGreedyCuda_Empty", [], 0.0, 0, 0.0, [], 0.0, 0.0, 0.0,
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

        # Build dense visibility matrix on GPU (n_candidates x n_points)
        V = cp.zeros((n_cand, self.num_points), dtype=cp.uint8)
        for i in range(n_cand):
            vis_idx = vis_map[i]
            if len(vis_idx) > 0:
                V[i, cp.asarray(vis_idx, dtype=cp.int64)] = 1

        logger.info("[LazyGreedyCuda] Visibility matrix: %s (%.1f MB)",
                    V.shape, V.nbytes / 1e6)

        uncovered = cp.ones(self.num_points, dtype=cp.uint8)
        candidate_active = cp.ones(n_cand, dtype=cp.bool_)
        total_covered = 0
        target_covered = int(target_coverage * self.num_points)

        # Compute initial gains on GPU
        gains = (V.astype(cp.float32) @ uncovered.astype(cp.float32))
        # Track whether gain is stale
        stale = cp.ones(n_cand, dtype=cp.bool_)

        selected_viewpoints: List[ViewpointResult] = []
        evaluations = 0
        optimization_start = get_time()

        while total_covered < target_covered and len(selected_viewpoints) < max_viewpoints:
            if not cp.any(candidate_active):
                break

            # Find candidate with max stale gain
            masked_gains = cp.where(candidate_active, gains, cp.float32(-1.0))

            while True:
                best = int(cp.argmax(masked_gains))
                best_gain = float(masked_gains[best])

                if best_gain <= 0:
                    break

                if not stale[best]:
                    # Gain is fresh — select this candidate
                    break

                # Re-evaluate gain
                evaluations += 1
                fresh_gain = float(
                    cp.sum(V[best].astype(cp.float32) * uncovered.astype(cp.float32))
                )
                gains[best] = fresh_gain
                masked_gains[best] = fresh_gain
                stale[best] = False

                # If still the best, we'll select it on next iteration

            if best_gain <= 0:
                logger.info("  [LazyGreedyCuda] No candidate provides new coverage.")
                break

            # Select this candidate
            newly_covered_mask = V[best] & uncovered
            uncovered &= ~V[best]
            candidate_active[best] = False
            # Mark all remaining as stale (coverage changed)
            stale[:] = True

            newly_covered_indices = cp.where(newly_covered_mask)[0].get()
            total_covered = self.num_points - int(cp.sum(uncovered))
            coverage = total_covered / self.num_points

            best_vp, best_orient = candidates[best]
            selected_viewpoints.append(ViewpointResult(
                position=np.asarray(best_vp),
                orientation=np.asarray(best_orient),
                visible_indices=newly_covered_indices,
                coverage_score=int(best_gain) / self.num_points,
                computation_time=0.0,
            ))

            logger.info("  [LazyGreedyCuda] VP %d: +%d pts, coverage=%.1f%%",
                        len(selected_viewpoints), int(best_gain), coverage * 100)

        optimization_time = get_time() - optimization_start
        total_time = get_time() - start_time
        coverage = total_covered / self.num_points
        redundancy = self.query.compute_redundancy(selected_viewpoints)

        logger.info("  [LazyGreedyCuda] Done: %d evaluations (vs %d standard).",
                    evaluations + n_cand, n_cand * len(selected_viewpoints))

        return OptimizationResult(
            method_name="LazyGreedyCuda",
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
