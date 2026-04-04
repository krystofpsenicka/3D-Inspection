"""Targeted viewpoint sampling biased toward uncovered surface regions."""

import logging

import cupy as cp
from typing import Tuple

from ...core.constants import (
    NORM_EPS, DEFAULT_MAX_DIR_NOISE_RAD,
    PROXIMITY_KNN_FRACTION,
    TARGETED_PROXIMITY_SIGMA_FACTOR, DEFAULT_K_COVERAGE,
)
from .weighted import WeightedViewpointSampler
from ...core.base import VisibilityQueryBase
from ..utils.proximity import compute_knn_proximity_weights

logger = logging.getLogger(__name__)


class TargetedViewpointSampler(WeightedViewpointSampler):
    """Sample viewpoints biased toward uncovered surface regions.

    Inherits weighted free space sampling capabilities and adds proximity-weighted
    targeted sampling with configurable batch size per iteration.
    """

    def sample(self, uncovered_indices: cp.ndarray, num_candidates: int,
                        side: str = "outside",
                        min_distance: float | None = None,
                        max_distance_offset: float = 0.95,
                        max_dir_noise_rad: float = DEFAULT_MAX_DIR_NOISE_RAD,
                        curvature_weighting: bool = False,
                        visibility_query: VisibilityQueryBase = None,
                        k_coverage: int = DEFAULT_K_COVERAGE,
                        coverage_count_gpu: cp.ndarray | None = None,
                        samples_per_iteration: int | None = None,
                        ) -> Tuple[cp.ndarray, cp.ndarray]:
        """Sample candidates biased toward uncovered surface regions.

        Returns:
            (positions_gpu, rotmats_gpu) — CuPy arrays (N,3) and (N,3,3) on GPU.
        """
        # Default to iterative mode with 1 sample per iteration when visibility_query is provided
        if samples_per_iteration is None:
            samples_per_iteration = 1 if visibility_query is not None else num_candidates

        return self._sample_targeted_iterative(
            uncovered_indices, num_candidates, side, min_distance,
            max_distance_offset, max_dir_noise_rad, curvature_weighting,
            visibility_query, k_coverage, coverage_count_gpu,
            samples_per_iteration)

    def _sample_targeted_iterative(self, uncovered_indices: cp.ndarray, num_candidates: int,
                                   side: str, min_distance: float | None, max_distance_offset: float,
                                   max_dir_noise_rad: float, curvature_weighting: bool,
                                   visibility_query, k_coverage: int,
                                   coverage_count_gpu, samples_per_iteration: int):
        """Sample viewpoints in batches, optionally updating coverage after each."""
        centers_gpu, base_weights_gpu, coarse_res = self.get_feasible_sampling_data(
            side, min_distance, max_distance_offset, curvature_weighting)
        if int(len(centers_gpu)) == 0:
            return cp.empty((0, 3), dtype=cp.float32), cp.empty((0, 3, 3), dtype=cp.float32)

        # Initialize coverage tracking only when visibility_query is provided
        if visibility_query is not None:
            if coverage_count_gpu is None:
                coverage_count_gpu = cp.full(
                    self.num_points, k_coverage, dtype=cp.int32)
                coverage_count_gpu[uncovered_indices] = 0

        all_pos_list = []
        all_rot_list = []
        remaining = num_candidates

        while remaining > 0:
            # Determine uncovered points to target for this iteration
            if visibility_query is not None:
                under_k_mask = coverage_count_gpu < k_coverage
                if not cp.any(under_k_mask):
                    logger.info("[TargetedSampler] All points k=%d-covered "
                                "after %d candidates.", k_coverage,
                                num_candidates - remaining)
                    break
                under_k_indices = cp.where(under_k_mask)[0]
                uncovered_pts_gpu = self.target_points[under_k_indices]
            else:
                uncovered_pts_gpu = self.target_points[uncovered_indices]

            sigma = TARGETED_PROXIMITY_SIGMA_FACTOR * coarse_res
            prox_w = self._compute_uncovered_proximity_weights(
                centers_gpu, uncovered_pts_gpu, sigma)

            blended = base_weights_gpu * prox_w
            blended_sum = float(blended.sum())
            if blended_sum < NORM_EPS:
                logger.warning("[TargetedSampler] All blended weights zero "
                               "at candidate %d.", num_candidates - remaining)
                break
            blended = blended / blended_sum

            n_this = min(samples_per_iteration, remaining)
            batch_pos_gpu, batch_rot_gpu = self._sample_from_free_space(
                centers_gpu, blended, coarse_res, n_this,
                max_dir_noise_rad=max_dir_noise_rad,
                direction_targets_gpu=uncovered_pts_gpu,
                curvature_weighting=curvature_weighting,
            )
            if len(batch_pos_gpu) == 0:
                break

            all_pos_list.append(batch_pos_gpu)
            all_rot_list.append(batch_rot_gpu)
            remaining -= len(batch_pos_gpu)

            # Only update coverage if another iteration follows
            if visibility_query is not None and remaining > 0:
                V_batch, _ = visibility_query.compute_visibility_batch(
                    batch_pos_gpu, batch_rot_gpu)
                coverage_count_gpu += V_batch.astype(cp.int32).sum(axis=0)

        if not all_pos_list:
            return cp.empty((0, 3), dtype=cp.float32), cp.empty((0, 3, 3), dtype=cp.float32)

        positions_gpu = cp.concatenate(all_pos_list)
        rotmats_gpu = cp.concatenate(all_rot_list)

        if visibility_query is not None:
            n_remaining = int((coverage_count_gpu < k_coverage).sum())
            logger.info("[TargetedSampler] Iterative (k=%d): generated %d/%d "
                        "candidates, %d points remain under-covered.",
                        k_coverage, len(positions_gpu), num_candidates,
                        n_remaining)
        else:
            logger.info("[TargetedSampler] Batch: generated %d/%d candidates.",
                        len(positions_gpu), num_candidates)

        return positions_gpu, rotmats_gpu

    def _compute_uncovered_proximity_weights(self, feasible_gpu: cp.ndarray,
                                             uncovered_gpu: cp.ndarray, sigma: float,
                                             k_fraction=PROXIMITY_KNN_FRACTION):
        """KNN proximity weighting toward uncovered surface regions."""
        k = max(1, int(k_fraction * self.num_points))
        return compute_knn_proximity_weights(feasible_gpu, uncovered_gpu, k, sigma)
