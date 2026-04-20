"""Expansion set-cover optimizer — composes an inner optimizer with a refinement sampler."""

import logging
import cupy as cp
from typing import Optional, Tuple

from .base import IterativeSetCoverOptimizer
from ..sampling.samplers.expansion import ExpansionSampler

logger = logging.getLogger(__name__)


class ExpansionIterativeSetCover(IterativeSetCoverOptimizer):
    """Set-cover optimizer with viewpoint expansion/refinement.

    Wraps an inner ``IterativeSetCoverOptimizer`` and an
    ``ExpansionSampler``.  Each iteration selects the best candidate
    via the inner optimizer, then refines it with the sampler before
    committing the selection.

    The refinement is guarded by a marginal-gain comparison on the current
    uncovered set (tracked here), so each committed viewpoint covers at
    least as many uncovered points as the inner optimizer's pick. This
    upper-bounds the total viewpoint count by the inner optimizer's count.

    Inherits ``optimize()`` from the base class.
    """

    def __init__(self, inner_optimizer: IterativeSetCoverOptimizer,
                 sampler: ExpansionSampler):
        self.inner = inner_optimizer
        self.sampler = sampler
        self.num_points = inner_optimizer.num_points
        self.positions = inner_optimizer.positions
        self.rotmats = inner_optimizer.rotmats
        self._uncovered_mask = cp.ones(self.num_points, dtype=cp.bool_)

    @property
    def last_selected_index(self) -> int:
        return self.inner.last_selected_index

    def select_next(self) -> Optional[Tuple[cp.ndarray, cp.ndarray, cp.ndarray]]:
        result = self.inner.select_next()
        if result is None:
            return None
        pos, rot, vis = result
        ref_pos, ref_rot, ref_vis = self.sampler.refine(
            pos, rot, vis, self._uncovered_mask)
        return (ref_pos, ref_rot, ref_vis)

    def commit_selection(self, visible_indices: cp.ndarray):
        if len(visible_indices) > 0:
            self._uncovered_mask[visible_indices] = False
        self.inner.commit_selection(visible_indices)
