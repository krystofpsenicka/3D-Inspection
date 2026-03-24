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
        """Check visibility using OptiX GPU raycasting via Triro."""
        start = get_time()

        candidate_indices = self.points_in_frustum_gpu(viewpoint, orientation)

        if len(candidate_indices) == 0:
            return np.array([]), get_time() - start

        num_candidates = len(candidate_indices)

        # Build rays on GPU
        candidate_indices_gpu = cp.asarray(candidate_indices)
        candidate_points_gpu = self.gpu_points[candidate_indices_gpu]  # (C, 3) float64

        vp_gpu = cp.asarray(viewpoint, dtype=cp.float64)
        origins_gpu = cp.tile(vp_gpu, (num_candidates, 1))
        vectors_gpu = candidate_points_gpu - origins_gpu
        distances_gpu = cp.linalg.norm(vectors_gpu, axis=1)
        ray_dirs_gpu = vectors_gpu / (distances_gpu[:, cp.newaxis] + NORM_EPS)

        # CuPy -> torch (float32 required by OptiX)
        origins_torch = torch.as_tensor(origins_gpu.astype(cp.float32), device='cuda')
        ray_dirs_torch = torch.as_tensor(ray_dirs_gpu.astype(cp.float32), device='cuda')

        # Cast rays via Triro
        hit, _, _, location, _ = self.intersector.intersects_closest(
            origins_torch, ray_dirs_torch
        )

        # Compute t_hit for rays that hit — bring to CPU for comparison
        hit_np = hit.cpu().numpy()
        location_np = location.cpu().numpy()
        distances_np = distances_gpu.get()
        origins_np = origins_gpu.get()

        is_visible = np.ones(num_candidates, dtype=bool)

        hit_mask = hit_np
        if np.any(hit_mask):
            hit_locations = location_np[hit_mask]
            hit_origins = origins_np[hit_mask]
            t_hit = np.linalg.norm(hit_locations - hit_origins, axis=1)
            hit_distances = distances_np[hit_mask]
            is_visible[hit_mask] = (t_hit >= hit_distances - RAYCAST_TOLERANCE)

        visible_indices = candidate_indices[is_visible]

        comp_time = get_time() - start
        return visible_indices, comp_time
