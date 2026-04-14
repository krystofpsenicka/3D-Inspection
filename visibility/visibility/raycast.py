import logging
import numpy as np
import open3d as o3d
from numpy.linalg import norm
import time
from typing import Tuple

from ..core.types import FrustumParams
from .base import VisibilityQuery
from ..core.constants import NORM_EPS, RAYCAST_TOLERANCE

logger = logging.getLogger(__name__)


class RaycastingVisibilityQuery(VisibilityQuery):
    """
    Implements ground-truth visibility using Open3D's RaycastingScene.
    """

    def __init__(self, mesh: o3d.geometry.TriangleMesh, target_points: np.ndarray,
                 normals: np.ndarray, frustum_params: FrustumParams):
        super().__init__(target_points, normals, frustum_params)

        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        logger.info("[RaycastingQuery] Initialized O3D RaycastingScene for occlusion checks.")

    def compute_visibility(self, viewpoint: np.ndarray, rotation: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Checks visibility using the RaycastingScene: we cast a ray from the
        viewpoint to each target point. If the ray hits anything *before* it
        hits the target point, the target is occluded.
        """
        start = time.perf_counter()

        candidate_indices = self.points_in_frustum_with_kdtree(viewpoint, rotation)

        if len(candidate_indices) == 0:
            return np.array([], dtype=np.intp), time.perf_counter() - start

        candidate_points = self.target_points[candidate_indices]
        num_candidates = len(candidate_indices)

        origins = np.tile(viewpoint, (num_candidates, 1))
        vectors = candidate_points - origins

        distances = norm(vectors, axis=1)
        directions = vectors / (distances[:, np.newaxis] + NORM_EPS)

        rays = np.hstack([origins, directions])
        rays_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

        ans = self.scene.cast_rays(rays_tensor)
        t_hit = ans['t_hit'].numpy()

        is_visible_mask = (t_hit >= distances - RAYCAST_TOLERANCE)

        visible_indices = candidate_indices[is_visible_mask]

        comp_time = time.perf_counter() - start
        return visible_indices, comp_time
