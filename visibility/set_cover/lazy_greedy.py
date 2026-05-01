"""Lazy greedy set-cover optimizer  --  Minoux 1978 (CPU)."""

from __future__ import annotations

import heapq
import logging

import cupy as cp
import numpy as np

from .base import IterativeSetCoverOptimizer

logger = logging.getLogger(__name__)


class LazyGreedySetCover(IterativeSetCoverOptimizer):
    """CPU lazy greedy set-cover (Minoux 1978).

    Same solution as standard greedy but avoids most re-evaluations
    by using a max-heap. Operates on numpy internally, converts to/from
    cupy at the boundary.
    """

    def __init__(
        self,
        num_points: int,
        positions: np.ndarray,
        rotmats: np.ndarray,
        visibility_map: np.ndarray,
    ):
        self.num_points = num_points
        self.positions = positions
        self.rotmats = rotmats
        self.V = visibility_map.astype(np.bool_)  # (N, M) bool
        n_cand = len(positions)
        self.uncovered = np.ones(num_points, dtype=np.bool_)
        self._last_idx = -1
        self._generation = 0

        # Initialise the min-heap: (negative gain, candidate index, generation)
        self._heap: list = []
        for i in range(n_cand):
            gain = int(np.count_nonzero(self.V[i] & self.uncovered))
            heapq.heappush(self._heap, (-gain, i, self._generation))

        logger.info(
            "[LazyGreedySetCover] Initialized with %d candidates, %d points.", n_cand, num_points
        )

    def select_next(self) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray] | None:
        if not self._heap:
            return None

        self._generation += 1

        while self._heap:
            neg_gain, idx, entry_gen = heapq.heappop(self._heap)
            if entry_gen == self._generation:
                if -neg_gain == 0:
                    return None
                self._last_idx = idx
                vis = np.where(self.V[idx])[0]
                return (
                    cp.asarray(self.positions[idx]),
                    cp.asarray(self.rotmats[idx]),
                    cp.asarray(vis),
                )
            # Re-evaluate with current uncovered state
            fresh_gain = int(np.count_nonzero(self.V[idx] & self.uncovered))
            heapq.heappush(self._heap, (-fresh_gain, idx, self._generation))

        return None

    def commit_selection(self, visible_indices: cp.ndarray):
        assert self._last_idx != -1, "commit_selection called before select_next"
        vis_np = visible_indices.get()
        if len(vis_np) > 0:
            self.uncovered[vis_np] = False
        # Zero out selected candidate's row so it scores 0 if re-evaluated
        self.V[self._last_idx] = False
