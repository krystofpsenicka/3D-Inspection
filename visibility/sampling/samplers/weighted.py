"""Viewpoint sampling with sdf² weighting and optional curvature bias."""

import logging
import time

import cupy as cp
from typing import Tuple

import numpy as np

from shared.geometry import directions_rolls_to_rotmats

from ...core.constants import DEFAULT_MAX_DIR_NOISE_RAD, NORM_EPS
from ..utils.direction import knn_centroid_direction, apply_angular_noise
from .base import ViewpointSamplerBase

logger = logging.getLogger(__name__)


class WeightedViewpointSampler(ViewpointSamplerBase):
    """Sample viewpoints from free space with sdf² distance weighting.

    Optionally applies curvature weighting to bias toward geometrically
    complex surface regions.
    """

    def sample(self, num_candidates: int, side: str = "outside",
               min_distance: float | None = None,
               max_distance_offset: float = 0.95,
               curvature_weighting: bool = False,
               max_dir_noise_rad: float = DEFAULT_MAX_DIR_NOISE_RAD,
               ) -> Tuple[cp.ndarray, cp.ndarray]:
        """Sample candidate viewpoints from free space (GPU, OG-based).

        Args:
            num_candidates:      number of viewpoints to generate.
            side:                "outside" or "inside" the mesh.
            min_distance:        minimum clearance from mesh surface (default: 2×collision_radius).
            max_distance_offset: fraction of frustum_far for max distance.
            curvature_weighting: bias weights toward high-curvature regions.
            max_dir_noise_rad:   max angular noise for viewing direction.

        Returns:
            (positions_gpu, rotmats_gpu) — CuPy arrays (N,3) and (N,3,3) on GPU.
        """
        if self.num_points == 0:
            return cp.empty((0, 3), dtype=cp.float32), cp.empty((0, 3, 3), dtype=cp.float32)

        logger.info("[WeightedViewpointSampler] Sampling %d viewpoints from %s mesh "
                    "(curvature_weighting=%s) ...",
                    num_candidates, side.upper(), curvature_weighting)

        centers_gpu, weights_gpu, coarse_res = self.get_feasible_sampling_data(
            side, min_distance, max_distance_offset, curvature_weighting)

        logger.info("[WeightedViewpointSampler] %s free space: %d feasible voxel centers",
                    side.capitalize(), int(len(centers_gpu)))

        t0 = time.perf_counter()
        result = self._sample_from_free_space(
            centers_gpu, weights_gpu, coarse_res, num_candidates,
            max_dir_noise_rad=max_dir_noise_rad,
            curvature_weighting=curvature_weighting)
        dt = time.perf_counter() - t0
        logger.info("[WeightedViewpointSampler] _sample_from_free_space: %.3fs", dt)
        return result

    # ── Free-space sampling ────────────────────────────────────

    def _sample_from_free_space(self, centers_gpu: cp.ndarray, weights_gpu: cp.ndarray,
                                coarse_res: float, num_candidates: int,
                                max_dir_noise_rad: float = 0.0,
                                direction_targets_gpu=None,
                                curvature_weighting: bool = False,
                                ) -> Tuple[cp.ndarray, cp.ndarray]:
        """GPU-accelerated viewpoint sampling from feasible positions.

        Returns:
            (positions_gpu, rotmats_gpu) — CuPy arrays (N,3) and (N,3,3) on GPU.
        """
        n_feasible = int(len(centers_gpu))
        if n_feasible == 0:
            logger.warning("[WeightedViewpointSampler] No feasible positions — returning empty.")
            return cp.empty((0, 3), dtype=cp.float32), cp.empty((0, 3, 3), dtype=cp.float32)

        # 1. Weighted random sample (GPU)
        cdf = cp.cumsum(weights_gpu)
        cdf /= cdf[-1]
        rand_vals = cp.random.uniform(0, 1, size=num_candidates, dtype=cp.float32)
        indices = cp.searchsorted(cdf, rand_vals)
        indices = cp.clip(indices, 0, n_feasible - 1)
        sampled_gpu = centers_gpu[indices]

        # 2. Sub-voxel jitter on GPU
        jitter = cp.random.uniform(
            -coarse_res / 2, coarse_res / 2,
            size=(num_candidates, 3), dtype=cp.float32
        )
        sampled_gpu = sampled_gpu + jitter

        # 3. K-nearest-neighbor centroid for viewing direction (GPU)
        dir_targets = (direction_targets_gpu if direction_targets_gpu is not None
                       else self._target_points_gpu)
        normals_gpu = (cp.asarray(self.normals, dtype=cp.float32)
                       if curvature_weighting else None)
        base_dirs = knn_centroid_direction(
            sampled_gpu, dir_targets, normals_gpu=normals_gpu)
        norms = cp.linalg.norm(base_dirs, axis=1, keepdims=True)
        norms = cp.maximum(norms, NORM_EPS)
        base_dirs = base_dirs / norms

        # 4. Angular noise via Rodrigues rotation (GPU)
        directions_gpu = apply_angular_noise(base_dirs, max_dir_noise_rad)

        # 5. Random roll + GPU-vectorized rotation construction
        rolls_gpu = cp.random.uniform(0, 2 * cp.pi, size=num_candidates, dtype=cp.float32)
        rotmats_gpu = directions_rolls_to_rotmats(directions_gpu, rolls_gpu)

        logger.info("[WeightedViewpointSampler] Generated %d candidates from free space (GPU).",
                    num_candidates)
        return sampled_gpu, rotmats_gpu
