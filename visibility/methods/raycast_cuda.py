import numpy as np
import torch
import open3d as o3d
from numpy.linalg import norm
from time import time as get_time
from typing import Tuple

from ..core.types import FrustumParams
from ..core.base_cuda import VisibilityQueryCuda


class RaycastingVisibilityQueryCuda(VisibilityQueryCuda):
    """GPU-accelerated raycasting visibility using Triro (OptiX).

    Uses NVIDIA OptiX via Triro for hardware-accelerated ray-mesh intersection.
    Requires OptiX SDK >= 7.7 and triro installed.
    """

    def __init__(self, mesh: o3d.geometry.TriangleMesh, target_points: np.ndarray,
                 normals: np.ndarray, frustum_params: FrustumParams,
                 frustum_method: str = "bruteforce"):
        super().__init__(mesh, target_points, normals, frustum_params,
                         frustum_method=frustum_method)

        from triro.ray.ray_optix import RayMeshIntersector

        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        triangles = np.asarray(mesh.triangles, dtype=np.int32)

        vertices_torch = torch.from_numpy(vertices).cuda()
        triangles_torch = torch.from_numpy(triangles).cuda()

        self.intersector = RayMeshIntersector(vertices=vertices_torch, faces=triangles_torch)
        print("[RaycastingCuda] Initialized Triro OptiX RayMeshIntersector (GPU BVH)")

    def compute_visibility(self, viewpoint: np.ndarray,
                           direction: np.ndarray) -> Tuple[np.ndarray, float]:
        """Check visibility using OptiX GPU raycasting via Triro."""
        start = get_time()

        candidate_indices = self.points_in_frustum_gpu(viewpoint, direction)

        if len(candidate_indices) == 0:
            return np.array([]), get_time() - start

        candidate_points = self.target_points[candidate_indices]
        num_candidates = len(candidate_indices)

        # Build ray origins and directions (float64 for precision parity with CPU)
        origins_np = np.tile(viewpoint, (num_candidates, 1))
        vectors = candidate_points - origins_np
        distances = np.linalg.norm(vectors, axis=1)
        ray_dirs_np = vectors / (distances[:, np.newaxis] + 1e-12)

        # Convert to PyTorch CUDA tensors (float32 required by OptiX)
        origins_torch = torch.from_numpy(origins_np.astype(np.float32)).cuda()
        ray_dirs_torch = torch.from_numpy(ray_dirs_np.astype(np.float32)).cuda()

        # Cast rays via Triro
        hit, _, _, location, _ = self.intersector.intersects_closest(
            origins_torch, ray_dirs_torch
        )

        # Compute t_hit for rays that hit
        hit_np = hit.cpu().numpy()
        location_np = location.cpu().numpy()

        TOLERANCE = 1e-4
        is_visible = np.ones(num_candidates, dtype=bool)

        hit_mask = hit_np
        if np.any(hit_mask):
            hit_locations = location_np[hit_mask]
            hit_origins = origins_np[hit_mask]
            t_hit = np.linalg.norm(hit_locations - hit_origins, axis=1)
            hit_distances = distances[hit_mask]
            is_visible[hit_mask] = (t_hit >= hit_distances - TOLERANCE)

        visible_indices = candidate_indices[is_visible]

        comp_time = get_time() - start
        return visible_indices, comp_time
