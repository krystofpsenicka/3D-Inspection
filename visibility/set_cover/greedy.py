"""Standard greedy set-cover optimizer (CPU)."""

import logging
import cupy as cp
import numpy as np
from typing import Optional, Tuple

from .base import IterativeSetCoverOptimizer

logger = logging.getLogger(__name__)


class GreedySetCover(IterativeSetCoverOptimizer):
    """Standard greedy set-cover optimizer.

    Scores each candidate using matrix-vector multiplication and selects the
    one covering the most uncovered points. Operates on numpy internally,
    converts to/from cupy at the boundary.
    """

    def __init__(self, num_points: int, positions: np.ndarray,
                 rotmats: np.ndarray, visibility_map: np.ndarray):
        self.num_points = num_points
        self.positions = positions
        self.rotmats = rotmats
        self.V = visibility_map  # (N, M) bool
        n_cand = len(positions)
        self.uncovered = np.ones(num_points, dtype=np.bool_)
        self.candidate_active = np.ones(n_cand, dtype=np.bool_)
        self._last_idx = -1
        logger.info("[GreedySetCover] Initialized with %d candidates, %d points.",
                    n_cand, num_points)

    def select_next(self) -> Optional[Tuple[cp.ndarray, cp.ndarray, cp.ndarray]]:
        if not np.any(self.candidate_active):
            return None

        scores = self.V.astype(np.float32) @ self.uncovered.astype(np.float32)
        scores[~self.candidate_active] = 0

        best = int(np.argmax(scores))
        if scores[best] == 0:
            return None

        self._last_idx = best
        vis = np.where(self.V[best])[0]

        return (cp.asarray(self.positions[best]),
                cp.asarray(self.rotmats[best]),
                cp.asarray(vis))

    def commit_selection(self, visible_indices: cp.ndarray):
        assert self._last_idx != -1, "commit_selection called before select_next"
        vis_np = visible_indices.get()
        if len(vis_np) > 0:
            self.uncovered[vis_np] = False
        self.candidate_active[self._last_idx] = False
