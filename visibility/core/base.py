import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from numpy.linalg import norm
from time import time as get_time
from typing import List, Dict, Tuple

from .types import FrustumParams, ViewpointResult


def get_frustum_basis(direction):
    """Calculate orthonormal basis for frustum."""
    direction = direction / norm(direction)

    if 1.0 - np.abs(direction[2]) < 1e-6:
        temp_up = np.array([1.0, 0.0, 0.0])
    else:
        temp_up = np.array([0.0, 0.0, 1.0])

    right = np.cross(direction, temp_up)
    right = right / norm(right)
    up = np.cross(right, direction)

    return right, up


def get_frustum_bounding_sphere(viewpoint, direction, params):
    """Calculate bounding sphere for frustum (for spatial query)."""
    center = viewpoint + direction * (params.near + params.far) / 2.0

    half_depth = (params.far - params.near) / 2.0
    tan_half_fov = np.tan(params.fov_y / 2.0)
    far_half_size = params.far * tan_half_fov
    radius = np.sqrt(half_depth**2 + 2 * far_half_size**2)

    return center, radius


class VisibilityQuery:
    """Abstract base class for all visibility query implementations."""

    def __init__(self, mesh: o3d.geometry.TriangleMesh, target_points: np.ndarray,
                 normals: np.ndarray, frustum_params: FrustumParams):
        self.mesh = mesh
        self.target_points = target_points
        self.normals = normals
        self.frustum_params = frustum_params
        self.num_points = len(target_points)
        self.kdtree = KDTree(self.target_points)
        print(f"[VisibilityQuery] Initialized base query for {self.num_points} target points.")

    def compute_visibility(self, viewpoint: np.ndarray, direction: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Computes the visible indices and computation time from a single viewpoint.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement compute_visibility")

    def points_in_frustum_with_kdtree(self, viewpoint, direction):
        """Frustum culling with KD-tree spatial query."""
        center, radius = get_frustum_bounding_sphere(viewpoint, direction, self.frustum_params)
        candidate_indices = self.kdtree.query_ball_point(center, radius)

        if len(candidate_indices) == 0:
            return np.array([], dtype=int)

        candidate_points = self.target_points[candidate_indices]

        vp_vectors = candidate_points - viewpoint
        proj_distance = np.dot(vp_vectors, direction)

        mask = (proj_distance >= self.frustum_params.near) & (proj_distance <= self.frustum_params.far)

        right, up = get_frustum_basis(direction)
        tan_half_fov = np.tan(self.frustum_params.fov_y / 2.0)
        max_size = proj_distance * tan_half_fov

        lateral_right = np.abs(np.dot(vp_vectors, right))
        lateral_up = np.abs(np.dot(vp_vectors, up))

        mask &= (lateral_right < max_size) & (lateral_up < max_size)

        return np.array(candidate_indices)[mask]

    def compute_visibility_for_all_candidates(self, candidates: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[Dict[Tuple[Tuple[float, ...], Tuple[float, ...]], np.ndarray], float]:
        """Pre-computes and returns visibility for a list of candidates."""
        start_time = get_time()
        visibility_map = {}

        for i, (vp, direction) in enumerate(candidates):
            if (i + 1) % 100 == 0:
                print(f"  [VisibilityQuery] ... computed {i+1} / {len(candidates)} candidates")

            visible_indices, _ = self.compute_visibility(vp, direction)
            visibility_map[(tuple(vp), tuple(direction))] = visible_indices

        total_time = get_time() - start_time
        print(f"[VisibilityQuery] Visibility computation finished in {total_time:.2f}s")
        return visibility_map, total_time

    def compute_redundancy(self, viewpoints: List[ViewpointResult]) -> float:
        """Compute average coverage redundancy."""
        if self.num_points == 0 or not viewpoints:
            return 0

        coverage_count = np.zeros(self.num_points)
        for vp in viewpoints:
            if len(vp.visible_indices) > 0:
                coverage_count[vp.visible_indices] += 1

        covered_points = coverage_count[coverage_count > 0]
        if len(covered_points) == 0:
            return 0

        return np.mean(covered_points)
