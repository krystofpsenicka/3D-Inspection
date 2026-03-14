import numpy as np
import cupy as cp
import open3d as o3d
from numpy.linalg import norm
from time import time as get_time
from typing import List, Dict, Tuple

from .types import FrustumParams, ViewpointResult
from .base import VisibilityQuery, get_frustum_basis, get_frustum_bounding_sphere


class VisibilityQueryCuda(VisibilityQuery):
    """GPU-accelerated base class for visibility queries.

    Transfers target points and normals to GPU as CuPy arrays and provides
    two frustum culling strategies: brute-force GPU and hybrid KDTree+GPU.
    """

    def __init__(self, mesh: o3d.geometry.TriangleMesh, target_points: np.ndarray,
                 normals: np.ndarray, frustum_params: FrustumParams,
                 frustum_method: str = "bruteforce"):
        super().__init__(mesh, target_points, normals, frustum_params)
        self.frustum_method = frustum_method

        # Transfer to GPU (float64 for precision parity with CPU frustum culling)
        self.gpu_points = cp.asarray(self.target_points, dtype=cp.float64)
        self.gpu_normals = cp.asarray(self.normals, dtype=cp.float64)

        print(f"[VisibilityQueryCuda] GPU arrays ready ({self.num_points} points, "
              f"frustum_method={frustum_method})")

    def points_in_frustum_bruteforce_gpu(self, viewpoint: np.ndarray,
                                          direction: np.ndarray) -> np.ndarray:
        """Brute-force frustum test on GPU — tests all points in one pass."""
        direction = direction / norm(direction)
        vp_gpu = cp.asarray(viewpoint, dtype=cp.float64)
        dir_gpu = cp.asarray(direction, dtype=cp.float64)

        vp_vectors = self.gpu_points - vp_gpu  # (N, 3)
        proj_distance = vp_vectors @ dir_gpu   # (N,)

        mask = (proj_distance >= self.frustum_params.near) & \
               (proj_distance <= self.frustum_params.far)

        right, up = get_frustum_basis(direction)
        right_gpu = cp.asarray(right, dtype=cp.float64)
        up_gpu = cp.asarray(up, dtype=cp.float64)

        tan_half_fov = np.tan(self.frustum_params.fov_y / 2.0)
        max_size = proj_distance * tan_half_fov

        lateral_right = cp.abs(vp_vectors @ right_gpu)
        lateral_up = cp.abs(vp_vectors @ up_gpu)

        mask &= (lateral_right < max_size) & (lateral_up < max_size)

        return cp.where(mask)[0].get()  # back to CPU

    def points_in_frustum_kdtree_gpu(self, viewpoint: np.ndarray,
                                      direction: np.ndarray) -> np.ndarray:
        """Hybrid KDTree + GPU: coarse CPU ball query, then fine GPU frustum test."""
        direction = direction / norm(direction)

        # Step 1: CPU KDTree ball query (coarse filter)
        center, radius = get_frustum_bounding_sphere(viewpoint, direction, self.frustum_params)
        candidate_indices = self.kdtree.query_ball_point(center, radius)

        if len(candidate_indices) == 0:
            return np.array([], dtype=int)

        candidate_indices = np.array(candidate_indices)

        # Step 2: GPU fine filter on the subset
        candidate_points_gpu = self.gpu_points[candidate_indices]
        vp_gpu = cp.asarray(viewpoint, dtype=cp.float64)
        dir_gpu = cp.asarray(direction, dtype=cp.float64)

        vp_vectors = candidate_points_gpu - vp_gpu
        proj_distance = vp_vectors @ dir_gpu

        mask = (proj_distance >= self.frustum_params.near) & \
               (proj_distance <= self.frustum_params.far)

        right, up = get_frustum_basis(direction)
        right_gpu = cp.asarray(right, dtype=cp.float64)
        up_gpu = cp.asarray(up, dtype=cp.float64)

        tan_half_fov = np.tan(self.frustum_params.fov_y / 2.0)
        max_size = proj_distance * tan_half_fov

        lateral_right = cp.abs(vp_vectors @ right_gpu)
        lateral_up = cp.abs(vp_vectors @ up_gpu)

        mask &= (lateral_right < max_size) & (lateral_up < max_size)

        return candidate_indices[cp.where(mask)[0].get()]

    def points_in_frustum_gpu(self, viewpoint: np.ndarray,
                               direction: np.ndarray) -> np.ndarray:
        """Dispatch to the configured frustum culling strategy."""
        if self.frustum_method == "bruteforce":
            return self.points_in_frustum_bruteforce_gpu(viewpoint, direction)
        elif self.frustum_method == "kdtree":
            return self.points_in_frustum_kdtree_gpu(viewpoint, direction)
        else:
            raise ValueError(f"Unknown frustum_method: {self.frustum_method}")

    def compute_visibility_for_all_candidates(
        self, candidates: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[Dict[Tuple[Tuple[float, ...], Tuple[float, ...]], np.ndarray], float]:
        """Pre-computes visibility for all candidates using GPU methods."""
        start_time = get_time()
        visibility_map = {}

        for i, (vp, direction) in enumerate(candidates):
            if (i + 1) % 100 == 0:
                print(f"  [VisibilityQueryCuda] ... computed {i+1} / {len(candidates)} candidates")

            visible_indices, _ = self.compute_visibility(vp, direction)
            visibility_map[(tuple(vp), tuple(direction))] = visible_indices

        total_time = get_time() - start_time
        print(f"[VisibilityQueryCuda] Visibility computation finished in {total_time:.2f}s")
        return visibility_map, total_time
