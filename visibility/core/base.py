import logging
import numpy as np
from abc import ABC, abstractmethod
from scipy.spatial import KDTree
from numpy.linalg import norm
from time import time as get_time
from typing import Tuple

from .types import FrustumParams

logger = logging.getLogger(__name__)


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
    def compute_visibility(self, viewpoint, orientation) -> Tuple[np.ndarray, float]:
        """Compute visible indices from a single viewpoint."""

    @abstractmethod
    def compute_visibility_batch(self, positions, orientations) -> Tuple[np.ndarray, float]:
        """Compute visibility for a batch of viewpoints.

        Args:
            positions:    (N, 3) array of viewpoint positions.
            orientations: (N, 3, 3) array of rotation matrices.

        Returns ``(visibility_matrix, total_time)`` where ``visibility_matrix``
        is an (N, M) array; entry [i, j] indicates whether point j is visible
        from viewpoint i.
        """


class VisibilityQuery(VisibilityQueryBase):
    """CPU visibility query base with numpy arrays and KDTree."""

    def __init__(self, target_points: np.ndarray,
                 normals: np.ndarray, frustum_params: FrustumParams):
        super().__init__(frustum_params, num_points=len(target_points))
        self.target_points = target_points
        self.normals = normals
        self.kdtree = KDTree(self.target_points)
        logger.info("[VisibilityQuery] Initialized base query for %d target points.", self.num_points)

    def compute_visibility(self, viewpoint: np.ndarray, rotmat: np.ndarray) -> Tuple[np.ndarray, float]:
        raise NotImplementedError("Subclasses must implement compute_visibility")

    def compute_visibility_batch(self, positions, orientations) -> Tuple[np.ndarray, float]:
        """CPU batch visibility via per-viewpoint loop.

        Returns ``(visibility_matrix, total_time)`` where ``visibility_matrix``
        is an (N, M) bool array.
        """
        start_time = get_time()
        class_name = type(self).__name__

        n = len(positions)
        matrix = np.zeros((n, self.num_points), dtype=np.bool_)
        for i in range(n):
            if (i + 1) % 100 == 0:
                logger.info("  [%s] ... computed %d / %d candidates",
                            class_name, i + 1, n)

            visible_indices, _ = self.compute_visibility(positions[i], orientations[i])
            if len(visible_indices) > 0:
                matrix[i, visible_indices] = True

        total_time = get_time() - start_time
        logger.info("[%s] Visibility computation for %d candidates finished in %.2fs",
                    class_name, n, total_time)
        return matrix, total_time

    def points_in_frustum_with_kdtree(self, viewpoint: np.ndarray, rotmat: np.ndarray):
        """Frustum culling with KD-tree."""
        forward, right, up = rotmat[:, 0], rotmat[:, 1], rotmat[:, 2]

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
