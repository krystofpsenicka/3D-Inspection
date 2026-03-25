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

    def compute_visibility(self, viewpoint: np.ndarray,
                           orientation: np.ndarray) -> Tuple[np.ndarray, float]:
        """Compute visible points using OptiX GPU raycasting via Triro."""
        start = get_time()

        candidate_indices = self.points_in_frustum_gpu(viewpoint, orientation)

        if len(candidate_indices) == 0:
            return np.array([]), get_time() - start

        num_candidates = len(candidate_indices)

        # Build rays on GPU (float32 — OptiX works in float32 anyway)
        candidate_indices_gpu = cp.asarray(candidate_indices)
        candidate_points_gpu = self.gpu_points[candidate_indices_gpu].astype(cp.float32)

        vp_gpu = cp.asarray(viewpoint, dtype=cp.float32)
        origins_gpu = cp.tile(vp_gpu, (num_candidates, 1))
        vectors_gpu = candidate_points_gpu - origins_gpu
        distances_gpu = cp.linalg.norm(vectors_gpu, axis=1)
        ray_dirs_gpu = vectors_gpu / (distances_gpu[:, cp.newaxis] + NORM_EPS)

        # CuPy -> torch (zero-copy, already float32)
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

        visible_indices = candidate_indices[cp.asnumpy(is_visible)]

        comp_time = get_time() - start
        return visible_indices, comp_time
