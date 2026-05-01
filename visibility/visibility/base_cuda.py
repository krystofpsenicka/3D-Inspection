import logging

import cupy as cp
import numpy as np

from ..core.constants import CUDA_BLOCK_SIZE
from ..core.types import FrustumParams
from .base import VisibilityQueryBase

logger = logging.getLogger(__name__)

# ── Batch frustum culling CUDA kernel ────────────────────────────────────────
# One thread per (viewpoint, point) pair.  Loads one VP + one target point from
# global memory, computes 3 dot-products in registers, writes 1 byte.
_BATCH_FRUSTUM_KERNEL = cp.RawKernel(
    r"""
extern "C" __global__
void batch_frustum_cull(
    const float* points,       // (M, 3)
    const float* viewpoints,   // (N, 3)
    const float* rotmats,      // (N, 9) columns-contiguous [fwd|right|up]
    unsigned char* mask,       // (N * M) output
    int N, int M,
    float near, float far, float tan_half_fov, float aspect
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N * M) return;
    int vp_i = idx / M;
    int pt_j = idx % M;

    float vx = viewpoints[vp_i*3],   vy = viewpoints[vp_i*3+1], vz = viewpoints[vp_i*3+2];
    float fx = rotmats[vp_i*9+0],    fy = rotmats[vp_i*9+1],    fz = rotmats[vp_i*9+2];
    float rx = rotmats[vp_i*9+3],    ry = rotmats[vp_i*9+4],    rz = rotmats[vp_i*9+5];
    float ux = rotmats[vp_i*9+6],    uy = rotmats[vp_i*9+7],    uz = rotmats[vp_i*9+8];

    float dx = points[pt_j*3] - vx;
    float dy = points[pt_j*3+1] - vy;
    float dz = points[pt_j*3+2] - vz;

    float proj = dx*fx + dy*fy + dz*fz;
    if (proj < near || proj > far) { mask[idx] = 0; return; }

    float max_v = proj * tan_half_fov;
    float max_h = max_v * aspect;
    float lat_r = fabsf(dx*rx + dy*ry + dz*rz);
    float lat_u = fabsf(dx*ux + dy*uy + dz*uz);

    mask[idx] = (lat_r < max_h && lat_u < max_v) ? 1 : 0;
}
""",
    "batch_frustum_cull",
)


class VisibilityQueryCuda(VisibilityQueryBase):
    """Base class for visibility queries."""

    def __init__(
        self, target_points: cp.ndarray, normals: cp.ndarray, frustum_params: FrustumParams
    ):
        super().__init__(frustum_params, num_points=len(target_points))

        # float64 for same precision as CPU frustum culling
        self.gpu_points = target_points.astype(cp.float64)
        self.gpu_normals = normals.astype(cp.float64)

        logger.info("[VisibilityQueryCuda] GPU arrays ready (%d points)", self.num_points)

    def points_in_frustum_gpu(
        self, viewpoint_gpu: cp.ndarray, rotmat_gpu: cp.ndarray
    ) -> cp.ndarray:
        """Brute-force frustum culling on GPU.

        Args:
            viewpoint_gpu: (3,) CuPy array -- viewpoint position.
            rotmat_gpu:    (3, 3) CuPy array -- rotation matrix.

        Returns:
            CuPy int64 array of indices passing the frustum test.
        """
        forward_gpu = rotmat_gpu[:, 0].astype(cp.float64)
        right_gpu = rotmat_gpu[:, 1].astype(cp.float64)
        up_gpu = rotmat_gpu[:, 2].astype(cp.float64)
        vp_gpu = viewpoint_gpu.astype(cp.float64)

        vp_vectors = self.gpu_points - vp_gpu  # (N, 3)
        proj_distance = vp_vectors @ forward_gpu  # (N,)

        mask = (proj_distance >= self.frustum_params.near) & (
            proj_distance <= self.frustum_params.far
        )

        tan_half_fov = np.tan(self.frustum_params.fov_y / 2.0)
        max_size_v = proj_distance * tan_half_fov
        max_size_h = max_size_v * self.frustum_params.aspect

        lateral_right = cp.abs(vp_vectors @ right_gpu)
        lateral_up = cp.abs(vp_vectors @ up_gpu)

        mask &= (lateral_right < max_size_h) & (lateral_up < max_size_v)

        return cp.where(mask)[0]

    def batch_points_in_frustum_gpu(
        self, positions_gpu: cp.ndarray, rotmats_gpu: cp.ndarray
    ) -> cp.ndarray:
        """Batch frustum culling via fused CUDA kernel.

        Args:
            positions_gpu: (N, 3) CuPy float32 -- viewpoint positions.
            rotmats_gpu:   (N, 3, 3) CuPy float32 -- rotation matrices.

        Returns:
            (N, M) CuPy uint8 mask.
        """
        N = len(positions_gpu)
        M = self.num_points

        # Ensure arrays are contiguous for pointer arithmetic in CUDA kernel.
        positions_f32 = cp.ascontiguousarray(cp.asarray(positions_gpu, dtype=cp.float32))
        points_f32 = cp.ascontiguousarray(self.gpu_points.astype(cp.float32))

        # Flatten rotation matrices.
        rotmats_f32 = cp.ascontiguousarray(
            cp.asarray(rotmats_gpu, dtype=cp.float32).transpose(0, 2, 1)
        ).reshape(N, 9)

        mask = cp.zeros(N * M, dtype=cp.uint8)
        total_threads = N * M
        grid_size = (total_threads + CUDA_BLOCK_SIZE - 1) // CUDA_BLOCK_SIZE

        _BATCH_FRUSTUM_KERNEL(
            (grid_size,),
            (CUDA_BLOCK_SIZE,),
            (
                points_f32,
                positions_f32,
                rotmats_f32,
                mask,
                np.int32(N),
                np.int32(M),
                np.float32(self.frustum_params.near),
                np.float32(self.frustum_params.far),
                np.float32(np.tan(self.frustum_params.fov_y / 2.0)),
                np.float32(self.frustum_params.aspect),
            ),
        )

        return mask.reshape(N, M)

    def compute_visibility_batch(self, positions, rotmats) -> tuple[cp.ndarray, float]:
        """Compute visibility for a batch of viewpoints.

        Subclasses must override this to provide an efficient batch implementation.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement compute_visibility_batch")
