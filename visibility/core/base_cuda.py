import logging
import numpy as np
import cupy as cp
from typing import Tuple

from .types import FrustumParams
from .base import VisibilityQueryBase, get_frustum_basis_from_rotation

logger = logging.getLogger(__name__)


class VisibilityQueryCuda(VisibilityQueryBase):
    """GPU-accelerated base class for visibility queries.

    Transfers target points and normals to GPU as CuPy arrays and provides
    brute-force GPU frustum culling.
    """

    def __init__(self, target_points: np.ndarray,
                 normals: np.ndarray, frustum_params: FrustumParams):
        super().__init__(frustum_params, num_points=len(target_points))

        # Keep numpy references for set-cover optimizers
        self.target_points = target_points
        self.normals = normals

        # Transfer to GPU (float64 for the same precision as CPU frustum culling)
        self.gpu_points = cp.asarray(target_points, dtype=cp.float64)
        self.gpu_normals = cp.asarray(normals, dtype=cp.float64)

        logger.info("[VisibilityQueryCuda] GPU arrays ready (%d points)", self.num_points)

    def points_in_frustum_gpu(self, viewpoint: np.ndarray,
                               orientation: np.ndarray) -> np.ndarray:
        """Brute-force frustum culling on GPU."""
        forward, right, up = get_frustum_basis_from_rotation(orientation)

        vp_gpu = cp.asarray(viewpoint, dtype=cp.float64)
        dir_gpu = cp.asarray(forward, dtype=cp.float64)

        vp_vectors = self.gpu_points - vp_gpu  # (N, 3)
        proj_distance = vp_vectors @ dir_gpu   # (N,)

        mask = (proj_distance >= self.frustum_params.near) & \
               (proj_distance <= self.frustum_params.far)

        right_gpu = cp.asarray(right, dtype=cp.float64)
        up_gpu = cp.asarray(up, dtype=cp.float64)

        tan_half_fov = np.tan(self.frustum_params.fov_y / 2.0)
        max_size_v = proj_distance * tan_half_fov
        max_size_h = max_size_v * self.frustum_params.aspect

        lateral_right = cp.abs(vp_vectors @ right_gpu)
        lateral_up = cp.abs(vp_vectors @ up_gpu)

        mask &= (lateral_right < max_size_h) & (lateral_up < max_size_v)

        return cp.where(mask)[0].get()  # back to CPU
