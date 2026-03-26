"""Targeted viewpoint sampling biased toward uncovered surface regions."""

import logging

import cupy as cp
import numpy as np
from typing import List, Tuple

from ...core.constants import (
    NORM_EPS, DEFAULT_MAX_DIR_NOISE_RAD,
    GPU_NN_CHUNK_SIZE, PROXIMITY_KNN_FRACTION,
    TARGETED_PROXIMITY_SIGMA_FACTOR, DEFAULT_K_COVERAGE,
)
from .uniform import UniformViewpointSampler

logger = logging.getLogger(__name__)


class TargetedViewpointSampler(UniformViewpointSampler):
    """Sample viewpoints biased toward uncovered surface regions.

    Inherits uniform sampling capabilities and adds proximity-weighted
    targeted sampling.  Supports batch mode (all at once) and iterative
    mode (sample one, update coverage counts, repeat).
    """

    def sample_targeted(self, uncovered_indices, num_candidates: int,
                        side: str = "outside",
                        min_distance: float | None = None,
                        max_distance_offset: float = 0.95,
                        max_dir_noise_rad: float = DEFAULT_MAX_DIR_NOISE_RAD,
                        curvature_weighting: bool = False,
                        iterative: bool = False,
                        visibility_query=None,
                        k_coverage: int = DEFAULT_K_COVERAGE,
                        coverage_count_gpu=None,
                        ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Sample candidates biased toward uncovered surface regions.

        Re-uses cached feasible positions, re-weights by proximity to the
        under-covered points, and orients viewing directions toward them.

        Args:
            uncovered_indices: 1-D array of target-point indices not yet
                               fully covered (count < k).
            num_candidates:    max number of new candidates to generate.
            side:              "outside" or "inside".
            curvature_weighting: also apply curvature bias to base weights.
            iterative:         if True, sample one viewpoint at a time,
                               compute visibility, update coverage counts,
                               repeat.  Requires *visibility_query*.
            visibility_query:  required when iterative=True. Must implement
                               ``compute_visibility_batch(candidates)``.
            k_coverage:        coverage-redundancy target — a point is
                               considered covered when seen by >= k viewpoints.
                               Only used in iterative mode.
            coverage_count_gpu: optional (num_points,) CuPy int32 — per-point
                               coverage count.  If provided and iterative=True,
                               updated in-place.  If None, a fresh array is
                               built from *uncovered_indices*.

        Returns:
            List of (position, orientation) tuples, where orientation is a quaternion.
        """
        if iterative:
            return self._sample_targeted_iterative(
                uncovered_indices, num_candidates, side, min_distance,
                max_distance_offset, max_dir_noise_rad, curvature_weighting,
                visibility_query, k_coverage, coverage_count_gpu)

        return self._sample_targeted_batch(
            uncovered_indices, num_candidates, side, min_distance,
            max_distance_offset, max_dir_noise_rad, curvature_weighting)

    def _sample_targeted_batch(self, uncovered_indices, num_candidates,
                               side, min_distance, max_distance_offset,
                               max_dir_noise_rad, curvature_weighting):
        """Sample all targeted candidates in one batch."""
        centers_gpu, base_weights_gpu, coarse_res = self._get_cached_free_space(
            side, min_distance, max_distance_offset, curvature_weighting)
        if int(len(centers_gpu)) == 0:
            return []

        uncovered_pts_gpu = self._target_points_gpu[cp.asarray(uncovered_indices)]
        sigma = TARGETED_PROXIMITY_SIGMA_FACTOR * coarse_res
        prox_w = self._compute_proximity_weights(
            centers_gpu, uncovered_pts_gpu, sigma)

        blended = base_weights_gpu * prox_w
        blended_sum = float(blended.sum())
        if blended_sum < NORM_EPS:
            logger.warning("[TargetedSampler] All blended weights are zero.")
            return []
        blended = blended / blended_sum

        return self._sample_from_free_space(
            centers_gpu, blended, coarse_res, num_candidates,
            max_dir_noise_rad=max_dir_noise_rad,
            direction_targets_gpu=uncovered_pts_gpu,
            curvature_weighting=curvature_weighting,
        )

    def _sample_targeted_iterative(self, uncovered_indices, num_candidates,
                                   side, min_distance, max_distance_offset,
                                   max_dir_noise_rad, curvature_weighting,
                                   visibility_query, k_coverage,
                                   coverage_count_gpu):
        """Sample one viewpoint at a time, updating coverage counts after each."""
        if visibility_query is None:
            raise ValueError("visibility_query is required for iterative targeted sampling.")

        centers_gpu, base_weights_gpu, coarse_res = self._get_cached_free_space(
            side, min_distance, max_distance_offset, curvature_weighting)
        if int(len(centers_gpu)) == 0:
            return []

        # Initialize coverage count array if not provided
        if coverage_count_gpu is None:
            coverage_count_gpu = cp.full(
                self.num_points, k_coverage, dtype=cp.int32)
            coverage_count_gpu[cp.asarray(uncovered_indices)] = 0

        all_candidates = []

        for i in range(num_candidates):
            under_k_mask = coverage_count_gpu < k_coverage
            if not cp.any(under_k_mask):
                logger.info("[TargetedSampler] All points k=%d-covered "
                            "after %d iterations.", k_coverage, i)
                break

            under_k_indices = cp.where(under_k_mask)[0]
            uncovered_pts_gpu = self._target_points_gpu[under_k_indices]
            sigma = TARGETED_PROXIMITY_SIGMA_FACTOR * coarse_res
            prox_w = self._compute_proximity_weights(
                centers_gpu, uncovered_pts_gpu, sigma)

            blended = base_weights_gpu * prox_w
            blended_sum = float(blended.sum())
            if blended_sum < NORM_EPS:
                logger.warning("[TargetedSampler] All blended weights zero "
                               "at iteration %d.", i)
                break
            blended = blended / blended_sum

            cands = self._sample_from_free_space(
                centers_gpu, blended, coarse_res, 1,
                max_dir_noise_rad=max_dir_noise_rad,
                direction_targets_gpu=uncovered_pts_gpu,
                curvature_weighting=curvature_weighting,
            )
            if not cands:
                break

            all_candidates.extend(cands)

            # Compute visibility and update coverage counts
            vis_map, _ = visibility_query.compute_visibility_batch(cands)
            for vis_indices in vis_map.values():
                if len(vis_indices) > 0:
                    coverage_count_gpu[cp.asarray(vis_indices)] += 1

        n_remaining = int((coverage_count_gpu < k_coverage).sum())
        logger.info("[TargetedSampler] Iterative (k=%d): generated %d/%d "
                    "candidates, %d points remain under-covered.",
                    k_coverage, len(all_candidates), num_candidates,
                    n_remaining)
        return all_candidates

    def _compute_proximity_weights(self, feasible_gpu, uncovered_gpu, sigma,
                                   k_fraction=PROXIMITY_KNN_FRACTION):
        """KNN proximity weighting toward uncovered surface regions.

        For each feasible viewpoint, finds the K nearest uncovered points and
        sums their exponential proximity contributions.

        Args:
            feasible_gpu:  (N, 3) CuPy array — feasible viewpoint positions.
            uncovered_gpu: (U, 3) CuPy array — uncovered surface points.
            sigma:         length scale for exponential decay.
            k_fraction:    fraction of total points to use as K.

        Returns:
            (N,) CuPy float32 — proximity weights (higher = closer to uncovered).
        """
        n_query = len(feasible_gpu)
        n_uncovered = len(uncovered_gpu)
        k = max(1, int(k_fraction * self.num_points))
        k = min(k, n_uncovered)

        uncovered_sq = cp.sum(uncovered_gpu ** 2, axis=1)  # (U,)
        weights = cp.empty(n_query, dtype=cp.float32)

        for start in range(0, n_query, GPU_NN_CHUNK_SIZE):
            end = min(start + GPU_NN_CHUNK_SIZE, n_query)
            q = feasible_gpu[start:end]  # (chunk, 3)
            q_sq = cp.sum(q ** 2, axis=1, keepdims=True)  # (chunk, 1)
            dist_sq = q_sq + uncovered_sq[cp.newaxis, :] - 2.0 * q @ uncovered_gpu.T
            cp.maximum(dist_sq, 0.0, out=dist_sq)

            # K nearest uncovered indices
            knn_idx = cp.argpartition(dist_sq, k, axis=1)[:, :k]  # (chunk, k)
            knn_dist = cp.sqrt(
                dist_sq[cp.arange(len(q))[:, cp.newaxis], knn_idx])  # (chunk, k)

            # Sum of exponential proximity contributions
            weights[start:end] = cp.sum(cp.exp(-knn_dist / sigma), axis=1)

        return weights
