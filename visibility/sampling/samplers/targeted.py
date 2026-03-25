"""Targeted viewpoint sampling biased toward uncovered surface regions."""

import logging

import cupy as cp
import numpy as np
from typing import List, Tuple

from ...core.constants import (
    NORM_EPS, DEFAULT_MAX_DIR_NOISE_RAD,
    GPU_NN_CHUNK_SIZE, PROXIMITY_KNN_FRACTION,
    TARGETED_PROXIMITY_SIGMA_FACTOR,
)
from .uniform import UniformViewpointSampler

logger = logging.getLogger(__name__)


class TargetedViewpointSampler(UniformViewpointSampler):
    """Sample viewpoints biased toward uncovered surface regions.

    Inherits uniform sampling capabilities and adds proximity-weighted
    targeted sampling.  Supports batch mode (all at once) and iterative
    mode (sample one, update uncovered set, repeat).
    """

    def sample_targeted(self, uncovered_indices, num_candidates: int,
                        side: str = "outside",
                        min_distance: float | None = None,
                        max_distance_offset: float = 0.95,
                        max_dir_noise_rad: float = DEFAULT_MAX_DIR_NOISE_RAD,
                        curvature_weighting: bool = False,
                        iterative: bool = False,
                        visibility_query=None,
                        ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Sample candidates biased toward uncovered surface regions.

        Re-uses cached feasible positions, re-weights by proximity to the
        uncovered points, and orients viewing directions toward them.

        Args:
            uncovered_indices: 1-D array of target-point indices not yet covered.
            num_candidates:    how many new candidates to generate.
            side:              "outside" or "inside".
            curvature_weighting: also apply curvature bias to base weights.
            iterative:         if True, sample one viewpoint at a time,
                               compute visibility, update uncovered set, repeat.
                               Requires *visibility_query*.
            visibility_query:  required when iterative=True. Must implement
                               ``compute_visibility_batch(candidates)``.

        Returns:
            List of (position, orientation) tuples, where orientation is a quaternion.
        """
        if iterative:
            return self._sample_targeted_iterative(
                uncovered_indices, num_candidates, side, min_distance,
                max_distance_offset, max_dir_noise_rad, curvature_weighting,
                visibility_query)

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
                                   visibility_query):
        """Sample one viewpoint at a time, updating uncovered set after each."""
        if visibility_query is None:
            raise ValueError("visibility_query is required for iterative targeted sampling.")

        centers_gpu, base_weights_gpu, coarse_res = self._get_cached_free_space(
            side, min_distance, max_distance_offset, curvature_weighting)
        if int(len(centers_gpu)) == 0:
            return []

        uncovered = set(np.asarray(uncovered_indices).tolist())
        all_candidates = []

        for i in range(num_candidates):
            if not uncovered:
                logger.info("[TargetedSampler] All points covered after %d iterations.", i)
                break

            uncovered_arr = np.array(sorted(uncovered), dtype=np.int64)
            uncovered_pts_gpu = self._target_points_gpu[cp.asarray(uncovered_arr)]
            sigma = TARGETED_PROXIMITY_SIGMA_FACTOR * coarse_res
            prox_w = self._compute_proximity_weights(
                centers_gpu, uncovered_pts_gpu, sigma)

            blended = base_weights_gpu * prox_w
            blended_sum = float(blended.sum())
            if blended_sum < NORM_EPS:
                logger.warning("[TargetedSampler] All blended weights zero at iteration %d.", i)
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

            # Compute visibility for the new viewpoint and update uncovered set
            vis_map, _ = visibility_query.compute_visibility_batch(cands)
            for vis_indices in vis_map.values():
                uncovered -= set(vis_indices.tolist())

        logger.info("[TargetedSampler] Iterative: generated %d/%d candidates, "
                    "%d points remain uncovered.",
                    len(all_candidates), num_candidates, len(uncovered))
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
