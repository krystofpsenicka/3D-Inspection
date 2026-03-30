"""GPU-accelerated lazy greedy set-cover optimizer — Minoux 1978."""

from __future__ import annotations

import logging
import cupy as cp
from typing import Optional, Tuple

from .base import IterativeSetCoverOptimizer

logger = logging.getLogger(__name__)


class LazyGreedySetCoverCuda(IterativeSetCoverOptimizer):
    """GPU lazy greedy set-cover (Minoux 1978).

    Maintains stale-gain flags on GPU. Re-evaluates only the current
    top candidate each iteration.
    """

    def __init__(self, num_points: int, positions: cp.ndarray,
                 rotmats: cp.ndarray, visibility_map: cp.ndarray):
        self.num_points = num_points
        self.positions = positions
        self.rotmats = rotmats
        self.V = visibility_map  # (N, M) uint8 on GPU
        n_cand = len(positions)
        self.uncovered = cp.ones(num_points, dtype=cp.uint8)
        self.candidate_active = cp.ones(n_cand, dtype=cp.bool_)
        self._last_idx = -1

        # Initial gains
        self.gains = self.V.astype(cp.float32) @ self.uncovered.astype(cp.float32)
        self.stale = cp.ones(n_cand, dtype=cp.bool_)

        logger.info("[LazyGreedySetCoverCuda] Initialized with %d candidates, %d points.",
                    n_cand, num_points)

    def select_next(self) -> Optional[Tuple[cp.ndarray, cp.ndarray, cp.ndarray]]:
        if not cp.any(self.candidate_active):
            return None

        masked_gains = cp.where(self.candidate_active, self.gains, cp.float32(-1.0))

        while True:
            best = int(cp.argmax(masked_gains))
            best_gain = float(masked_gains[best])

            if best_gain <= 0:
                return None

            if not self.stale[best]:
                self._last_idx = best
                vis = cp.where(self.V[best])[0]
                return (self.positions[best], self.rotmats[best], vis)

            # Re-evaluate gain
            fresh_gain = float(
                cp.sum(self.V[best].astype(cp.float32) * self.uncovered.astype(cp.float32))
            )
            self.gains[best] = fresh_gain
            masked_gains[best] = fresh_gain
            self.stale[best] = False

    def commit_selection(self, visible_indices: cp.ndarray):
        assert self._last_idx != -1, "commit_selection called before select_next"
        if len(visible_indices) > 0:
            self.uncovered[visible_indices] = 0
        self.candidate_active[self._last_idx] = False
        self.stale[:] = True
        self._last_idx = -1
