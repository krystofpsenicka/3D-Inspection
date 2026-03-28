import logging
import numpy as np
import cupy as cp
import torch
import open3d as o3d
from time import time as get_time
from typing import Tuple

from ..core.types import FrustumParams
from ..core.base_cuda import VisibilityQueryCuda
from ..core.constants import NORM_EPS, RAYCAST_TOLERANCE

logger = logging.getLogger(__name__)


class RaycastingVisibilityQueryCuda(VisibilityQueryCuda):
    """GPU-accelerated raycasting visibility using Triro (OptiX).

    Uses NVIDIA OptiX via Triro for hardware-accelerated ray-mesh intersection.
    Requires OptiX SDK >= 7.7 and triro installed.
    """

    def __init__(self, mesh: o3d.geometry.TriangleMesh, target_points: np.ndarray,
                 normals: np.ndarray, frustum_params: FrustumParams):
        super().__init__(target_points, normals, frustum_params)

        from triro.ray.ray_optix import RayMeshIntersector

        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        triangles = np.asarray(mesh.triangles, dtype=np.int32)

        vertices_torch = torch.from_numpy(vertices).cuda()
        triangles_torch = torch.from_numpy(triangles).cuda()

        self.intersector = RayMeshIntersector(vertices=vertices_torch, faces=triangles_torch)
        logger.info("[RaycastingCuda] Initialized Triro OptiX RayMeshIntersector (GPU BVH)")

    def compute_visibility(self, viewpoint_gpu: cp.ndarray,
                           rotmat_gpu: cp.ndarray) -> Tuple[cp.ndarray, float]:
        """Compute visible points using OptiX GPU raycasting via Triro.

        Args:
            viewpoint_gpu: (3,) CuPy array -- viewpoint position on GPU.
            rotmat_gpu:    (3, 3) CuPy array -- rotation matrix on GPU.

        Returns:
            (visible_indices, comp_time) where visible_indices is CuPy int64.
        """
        start = get_time()

        candidate_indices = self.points_in_frustum_gpu(viewpoint_gpu, rotmat_gpu)

        if len(candidate_indices) == 0:
            return cp.array([], dtype=cp.int64), get_time() - start

        num_candidates = len(candidate_indices)

        # Build rays on GPU (float32 -- OptiX works in float32 anyway)
        candidate_points_gpu = self.gpu_points[candidate_indices].astype(cp.float32)

        vp_gpu = viewpoint_gpu.astype(cp.float32)
        origins_gpu = cp.tile(vp_gpu, (num_candidates, 1))
        vectors_gpu = candidate_points_gpu - origins_gpu
        distances_gpu = cp.linalg.norm(vectors_gpu, axis=1)
        ray_dirs_gpu = vectors_gpu / (distances_gpu[:, cp.newaxis] + NORM_EPS)

        # CuPy -> torch (zero-copy)
        origins_torch = torch.as_tensor(origins_gpu, device='cuda')
        ray_dirs_torch = torch.as_tensor(ray_dirs_gpu, device='cuda')

        # Cast rays using Triro
        hit, _, _, location, _ = self.intersector.intersects_closest(
            origins_torch, ray_dirs_torch
        )

        # Occlusion check on GPU: visible if no hit, or hit is beyond the target point
        hit_gpu = cp.from_dlpack(hit)
        location_gpu = cp.from_dlpack(location)

        is_visible = cp.ones(num_candidates, dtype=cp.bool_)
        if cp.any(hit_gpu):
            t_hit = cp.linalg.norm(location_gpu[hit_gpu] - origins_gpu[hit_gpu], axis=1)
            is_visible[hit_gpu] = (t_hit >= distances_gpu[hit_gpu] - RAYCAST_TOLERANCE)

        visible_indices = candidate_indices[is_visible]

        comp_time = get_time() - start
        return visible_indices, comp_time

    def compute_visibility_batch(self, positions, rotmats) -> Tuple[cp.ndarray, float]:
        """Batch raycasting: single CUDA frustum kernel + single OptiX call.

        Args:
            positions: (N, 3) CuPy float32 -- viewpoint positions.
            rotmats:   (N, 3, 3) CuPy float32 -- rotation matrices.

        Returns:
            ``(V, total_time)`` where *V* is ``(N, M)`` uint8 CuPy array.
        """
        start = get_time()
        N, M = len(positions), self.num_points
        V = cp.zeros((N, M), dtype=cp.uint8)

        # 1. Batch frustum culling (CUDA kernel)
        frustum_mask = self.batch_points_in_frustum_gpu(positions, rotmats)

        # 2. Extract all candidate (viewpoint, point) pairs
        vp_idx, pt_idx = cp.where(frustum_mask)
        del frustum_mask  # free (N*M) bytes
        if len(vp_idx) == 0:
            return V, get_time() - start

        # 3. Build rays for all pairs (float32 for OptiX)
        positions_f32 = cp.asarray(positions, dtype=cp.float32)
        origins = positions_f32[vp_idx]
        targets = self.gpu_points[pt_idx].astype(cp.float32)
        vectors = targets - origins
        distances = cp.linalg.norm(vectors, axis=1)
        ray_dirs = vectors / (distances[:, cp.newaxis] + NORM_EPS)

        # 4. Single OptiX call for ALL rays
        hit, _, _, location, _ = self.intersector.intersects_closest(
            torch.as_tensor(origins, device='cuda'),
            torch.as_tensor(ray_dirs, device='cuda'))

        # 5. Batch occlusion check
        hit_gpu = cp.from_dlpack(hit)
        loc_gpu = cp.from_dlpack(location)
        is_visible = cp.ones(len(vp_idx), dtype=cp.bool_)
        if cp.any(hit_gpu):
            t_hit = cp.linalg.norm(loc_gpu[hit_gpu] - origins[hit_gpu], axis=1)
            is_visible[hit_gpu] = (t_hit >= distances[hit_gpu] - RAYCAST_TOLERANCE)

        # 6. Scatter into V
        V[vp_idx[is_visible], pt_idx[is_visible]] = 1

        total_time = get_time() - start
        logger.info("[RaycastingCuda] Batch visibility for %d VPs: %d rays, %.2fs",
                    N, len(vp_idx), total_time)
        return V, total_time
