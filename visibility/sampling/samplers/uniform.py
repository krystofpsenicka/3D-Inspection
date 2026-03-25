"""Uniform viewpoint sampling with sdf² weighting and optional curvature bias."""

import logging
import time

import cupy as cp
from typing import List, Tuple

import numpy as np

from ...core.constants import DEFAULT_MAX_DIR_NOISE_RAD
from .base import ViewpointSamplerBase

logger = logging.getLogger(__name__)


class UniformViewpointSampler(ViewpointSamplerBase):
    """Sample viewpoints uniformly from free space with sdf² distance weighting.

    Optionally applies curvature weighting to bias toward geometrically
    complex surface regions.
    """

    def sample(self, num_candidates: int, side: str = "outside",
               min_distance: float | None = None,
               max_distance_offset: float = 0.95,
               curvature_weighting: bool = False,
               max_dir_noise_rad: float = DEFAULT_MAX_DIR_NOISE_RAD,
               ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Sample candidate viewpoints from free space (GPU, OG-based).

        Args:
            num_candidates:      number of viewpoints to generate.
            side:                "outside" or "inside" the mesh.
            min_distance:        minimum clearance from mesh surface (default: 2×collision_radius).
            max_distance_offset: fraction of frustum_far for max distance.
            curvature_weighting: bias weights toward high-curvature regions.
            max_dir_noise_rad:   max angular noise for viewing direction.

        Returns:
            List of (position, quaternion) tuples.
        """
        if self.num_points == 0:
            return []

        logger.info("[UniformSampler] Sampling %d viewpoints from %s mesh "
                    "(curvature_weighting=%s) ...",
                    num_candidates, side.upper(), curvature_weighting)

        centers_gpu, weights_gpu, coarse_res = self._get_cached_free_space(
            side, min_distance, max_distance_offset, curvature_weighting)

        logger.info("[UniformSampler] %s free space: %d feasible positions",
                    side.capitalize(), int(len(centers_gpu)))

        t0 = time.perf_counter()
        result = self._sample_from_free_space(
            centers_gpu, weights_gpu, coarse_res, num_candidates,
            max_dir_noise_rad=max_dir_noise_rad,
            curvature_weighting=curvature_weighting)
        dt = time.perf_counter() - t0
        logger.info("[UniformSampler] _sample_from_free_space: %.3fs", dt)
        return result
