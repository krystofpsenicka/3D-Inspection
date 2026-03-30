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

from .base import ProbabilisticSampler
from .optimizing import OptimizingSampler
from ...core.base_cuda import VisibilityQueryCuda

logger = logging.getLogger(__name__)


class ExpansionSampler(ABC):
    """Base for expansion/refinement samplers used by ExpansionIterativeSetCover."""

    @abstractmethod
    def refine(self, position: cp.ndarray, orientation: cp.ndarray,
               visible_indices: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        """Refine a viewpoint by searching for a better one nearby.

        Args:
            position:        (3,) CuPy array.
            orientation:     (3, 3) CuPy rotation matrix.
            visible_indices: (K,) CuPy int64 array of visible point indices.

        Returns:
            (position, orientation, visible_indices) — the refined viewpoint,
            or the originals if no improvement was found.
        """


class ProbabilisticExpansionSampler(ExpansionSampler):
    """Adapter that wraps a ProbabilisticSampler for expansion refinement.

    Restricts the sampler's search space to a sphere, samples N candidates,
    evaluates visibility for all, and returns the best one.
    """

    def __init__(self, sampler: ProbabilisticSampler,
                 visibility_query: VisibilityQueryCuda,
                 n_samples: int = 20, radius: float = 0.5):
        self.sampler = sampler
        self.query = visibility_query
        self.n_samples = n_samples
        self.radius = radius

    def refine(self, position, orientation, visible_indices):
        self.sampler.restrict_to_sphere(cp.asnumpy(position), self.radius)
        try:
            positions_gpu, rotmats_gpu = self.sampler.sample(self.n_samples)
        finally:
            self.sampler.clear_restriction()

        if len(positions_gpu) == 0:
            return position, orientation, visible_indices

        V, _ = self.query.compute_visibility_batch(positions_gpu, rotmats_gpu)

        counts = V.sum(axis=1)
        best = int(cp.argmax(counts))
        if int(counts[best]) <= len(visible_indices):
            return position, orientation, visible_indices

        best_vis = cp.where(V[best])[0]
        return positions_gpu[best], rotmats_gpu[best], best_vis


class OptimizingExpansionSampler(ExpansionSampler):
    """Adapter that wraps an OptimizingSampler for expansion refinement.

    Restricts the optimizer's search space to a sphere (bounding box
    approximation) and runs one round of optimization.
    """

    def __init__(self, sampler: OptimizingSampler,
                 visibility_query: VisibilityQueryCuda,
                 radius: float = 0.5):
        self.sampler = sampler
        self.query = visibility_query
        self.radius = radius

    def refine(self, position, orientation, visible_indices):
        self.sampler.restrict_to_sphere(cp.asnumpy(position), self.radius)
        try:
            coverage_count_gpu = cp.zeros(self.query.num_points, dtype=cp.int32)
            if len(visible_indices) > 0:
                coverage_count_gpu[visible_indices] = 1

            result_pos, result_rot = self.sampler.sample_optimized(
                n_rounds=1,
                coverage_count_gpu=coverage_count_gpu,
                visibility_query=self.query,
            )
        finally:
            self.sampler.clear_restriction()

        if len(result_pos) == 0:
            return position, orientation, visible_indices

        V, _ = self.query.compute_visibility_batch(
            result_pos[:1], result_rot[:1])
        new_vis = cp.where(V[0])[0]

        if len(new_vis) <= len(visible_indices):
            return position, orientation, visible_indices

        return result_pos[0], result_rot[0], new_vis
