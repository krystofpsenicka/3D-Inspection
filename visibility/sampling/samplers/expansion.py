"""Expansion samplers for refining viewpoints in ExpansionIterativeSetCover.

Provides an ABC and two adapter subclasses that wrap existing samplers
with search-space restriction to refine a given viewpoint.

All expansion samplers operate on GPU (CuPy arrays) exclusively.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Tuple

import cupy as cp

from ...core.constants import DEFAULT_K_COVERAGE
from .base import ProbabilisticSampler
from .optimizing import OptimizingSampler
from ...visibility.base_cuda import VisibilityQueryCuda

logger = logging.getLogger(__name__)


class ExpansionSampler(ABC):
    """Base for expansion/refinement samplers used by ExpansionIterativeSetCover."""

    @abstractmethod
    def refine(self, position: cp.ndarray, rotation: cp.ndarray,
               visible_indices: cp.ndarray,
               uncovered_mask: cp.ndarray
               ) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        """Refine a viewpoint by searching for a better one nearby.

        A refined viewpoint is only returned when its marginal gain on the
        current uncovered set strictly exceeds the original's marginal gain;
        otherwise the originals are returned unchanged. This guarantees the
        expansion optimizer's viewpoint count is upper-bounded by the base
        greedy it wraps.

        Args:
            position:        (3,) CuPy array.
            rotation:        (3, 3) CuPy rotation matrix.
            visible_indices: (K,) CuPy int64 array of visible point indices.
            uncovered_mask:  (M,) CuPy bool array — True for points still
                             uncovered at this iteration.

        Returns:
            (position, rotation, visible_indices) — the refined viewpoint,
            or the originals if no strict improvement was found.
        """


class ProbabilisticExpansionSampler(ExpansionSampler):
    """Adapter that wraps a ProbabilisticSampler for expansion refinement."""

    def __init__(self, sampler: ProbabilisticSampler,
                 visibility_query: VisibilityQueryCuda,
                 n_samples: int = 20, radius: float = 0.5):
        self.sampler = sampler
        self.query = visibility_query
        self.n_samples = n_samples
        self.radius = radius

    def refine(self, position, rotation, visible_indices, uncovered_mask):
        self.sampler.restrict_to_sphere(position, self.radius)
        try:
            positions_gpu, rotmats_gpu = self.sampler.sample(self.n_samples)
        finally:
            self.sampler.clear_restriction()

        if len(positions_gpu) == 0:
            return position, rotation, visible_indices

        V, _ = self.query.compute_visibility_batch(positions_gpu, rotmats_gpu)

        # Marginal gain over current uncovered set
        V_bool = V.astype(cp.bool_)
        gains = (V_bool & uncovered_mask[cp.newaxis, :]).sum(axis=1)
        original_gain = int(uncovered_mask[visible_indices].sum())

        best = int(cp.argmax(gains))
        if int(gains[best]) <= original_gain:
            return position, rotation, visible_indices

        best_vis = cp.where(V_bool[best])[0]
        return positions_gpu[best], rotmats_gpu[best], best_vis


class OptimizingExpansionSampler(ExpansionSampler):
    """Adapter that wraps an OptimizingSampler for expansion refinement."""

    def __init__(self, sampler: OptimizingSampler,
                 visibility_query: VisibilityQueryCuda,
                 radius: float = 0.5):
        self.sampler = sampler
        self.query = visibility_query
        self.radius = radius

    def refine(self, position, rotation, visible_indices, uncovered_mask):
        self.sampler.restrict_to_sphere(position, self.radius)
        try:
            # Drive the inner optimizer's deficit so it rewards ONLY currently
            # uncovered points (deficit = k for uncovered, 0 for covered).
            coverage_count_gpu = cp.where(
                uncovered_mask, 0, DEFAULT_K_COVERAGE
            ).astype(cp.int32)

            result_pos, result_rot, _ = self.sampler.sample_optimized(
                n_rounds=1,
                coverage_count_gpu=coverage_count_gpu,
                visibility_query=self.query,
            )
        finally:
            self.sampler.clear_restriction()

        if len(result_pos) == 0:
            return position, rotation, visible_indices

        V, _ = self.query.compute_visibility_batch(
            result_pos[:1], result_rot[:1])
        refined_vis = cp.where(V[0].astype(cp.bool_))[0]

        refined_gain = int(uncovered_mask[refined_vis].sum())
        original_gain = int(uncovered_mask[visible_indices].sum())
        if refined_gain <= original_gain:
            return position, rotation, visible_indices

        return result_pos[0], result_rot[0], refined_vis
