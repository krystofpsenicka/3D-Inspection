"""GPU-accelerated greedy set-cover optimizer."""

import logging
import cupy as cp
from typing import Optional, Tuple

from .base import IterativeSetCoverOptimizer

logger = logging.getLogger(__name__)


class GreedySetCoverCuda(IterativeSetCoverOptimizer):
    """GPU-accelerated greedy set-cover optimizer.

    Scores candidates via GPU matrix-vector multiplication.
    """

    def __init__(self, num_points: int, positions: cp.ndarray, rotmats: cp.ndarray,
                 visibility_map: cp.ndarray):
        self.num_points = num_points
        self.positions = positions
        self.rotmats = rotmats
        self.V = visibility_map  # (N, M) uint8 on GPU
        n_cand = len(positions)
        self.uncovered = cp.ones(num_points, dtype=cp.uint8)
        self.candidate_active = cp.ones(n_cand, dtype=cp.bool_)
        self._last_idx = -1
        logger.info("[GreedySetCoverCuda] Initialized with %d candidates, %d points.",
                    n_cand, num_points)

    def select_next(self) -> Optional[Tuple[cp.ndarray, cp.ndarray, cp.ndarray]]:
        if not cp.any(self.candidate_active):
            return None

        scores = self.V.astype(cp.float32) @ self.uncovered.astype(cp.float32)
        scores[~self.candidate_active] = 0

        best = int(cp.argmax(scores))
        if int(scores[best]) == 0:
            return None

        self._last_idx = best
        vis = cp.where(self.V[best])[0]

        return (self.positions[best], self.rotmats[best], vis)

    def commit_selection(self, visible_indices: cp.ndarray):
        assert self._last_idx != -1, "commit_selection called before select_next"
        if len(visible_indices) > 0:
            self.uncovered[visible_indices] = 0
        self.candidate_active[self._last_idx] = False
