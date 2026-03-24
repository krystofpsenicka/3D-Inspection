import logging
import numpy as np
from abc import ABC, abstractmethod
from scipy.spatial import KDTree
from numpy.linalg import norm
from time import time as get_time
from typing import List, Dict, Tuple

from .types import FrustumParams, ViewpointResult

logger = logging.getLogger(__name__)


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


def get_frustum_basis_from_quaternion(q_wxyz):
    """Extract (forward, right, up) from quaternion [qw,qx,qy,qz], +X is forward."""
    from scipy.spatial.transform import Rotation as R

    rot = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    mat = rot.as_matrix()
    return mat[:, 0], mat[:, 1], mat[:, 2]  # forward, right, up


def get_frustum_bounding_sphere(viewpoint, direction, params):
    """Calculate bounding sphere for frustum (for spatial query)."""
    center = viewpoint + direction * (params.near + params.far) / 2.0

    half_depth = (params.far - params.near) / 2.0
    tan_half_fov = np.tan(params.fov_y / 2.0)
    far_half_h = params.far * tan_half_fov
    far_half_w = far_half_h * params.aspect
    radius = np.sqrt(half_depth**2 + far_half_w**2 + far_half_h**2)

    return center, radius


class VisibilityQueryBase(ABC):
    """Abstract base class for all visibility query implementations."""

    def __init__(self, frustum_params: FrustumParams, num_points: int):
        self.frustum_params = frustum_params
        self.num_points = num_points

    @abstractmethod
    def compute_visibility(self, viewpoint: np.ndarray,
                           orientation: np.ndarray) -> Tuple[np.ndarray, float]:
        """Computes visible indices and computation time from a single viewpoint.

        Args:
            viewpoint: 3D position.
            orientation: quaternion [qw,qx,qy,qz].

        Returns:
            visible_indices: array of visible point indices.
            computation_time: time taken to compute visibility for this viewpoint.
        """
        ...

    def compute_visibility_batch(self, candidates: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[Dict[int, np.ndarray], float]:
        """Computes visibility for a batch of viewpoints.

        Returns ``(visibility_map, total_time)`` where ``visibility_map``
        maps candidate index -> visible point indices array.
        """
        start_time = get_time()
        visibility_map: Dict[int, np.ndarray] = {}
        class_name = type(self).__name__

        for i, (vp, orientation) in enumerate(candidates):
            if (i + 1) % 100 == 0:
                logger.info("  [%s] ... computed %d / %d candidates",
                            class_name, i + 1, len(candidates))

            visible_indices, _ = self.compute_visibility(vp, orientation)
            visibility_map[i] = visible_indices

        total_time = get_time() - start_time
        logger.info("[%s] Visibility computation for %d candidates finished in %.2fs",
                    class_name, len(candidates), total_time)
        return visibility_map, total_time

    def compute_redundancy(self, viewpoints: List[ViewpointResult]) -> float:
        """Compute mean coverage redundancy."""
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


class VisibilityQuery(VisibilityQueryBase):
    """CPU visibility query base with numpy arrays and KDTree."""

    def __init__(self, target_points: np.ndarray,
                 normals: np.ndarray, frustum_params: FrustumParams):
        super().__init__(frustum_params, num_points=len(target_points))
        self.target_points = target_points
        self.normals = normals
        self.kdtree = KDTree(self.target_points)
        logger.info("[VisibilityQuery] Initialized base query for %d target points.", self.num_points)

    def compute_visibility(self, viewpoint: np.ndarray, orientation: np.ndarray) -> Tuple[np.ndarray, float]:
        raise NotImplementedError("Subclasses must implement compute_visibility")

    def points_in_frustum_with_kdtree(self, viewpoint, orientation):
        """Frustum culling with KD-tree."""
        forward, right, up = get_frustum_basis_from_quaternion(orientation)

        center, radius = get_frustum_bounding_sphere(viewpoint, forward, self.frustum_params)
        candidate_indices = self.kdtree.query_ball_point(center, radius)

        if len(candidate_indices) == 0:
            return np.array([], dtype=int)

        candidate_points = self.target_points[candidate_indices]

        vp_vectors = candidate_points - viewpoint
        proj_distance = np.dot(vp_vectors, forward)

        mask = (proj_distance >= self.frustum_params.near) & (proj_distance <= self.frustum_params.far)

        tan_half_fov = np.tan(self.frustum_params.fov_y / 2.0)
        max_size_v = proj_distance * tan_half_fov
        max_size_h = max_size_v * self.frustum_params.aspect

        lateral_right = np.abs(np.dot(vp_vectors, right))
        lateral_up = np.abs(np.dot(vp_vectors, up))

        mask &= (lateral_right < max_size_h) & (lateral_up < max_size_v)

        return np.array(candidate_indices)[mask]
