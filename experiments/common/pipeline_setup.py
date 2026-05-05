"""Cached mesh loading, OG building, surface sampling."""

from __future__ import annotations

import logging
import os
import sys

import cupy as cp
import numpy as np
import open3d as o3d

from .config import FrustumConfig, ModelConfig

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logger = logging.getLogger(__name__)


class DegenerateNormalsError(Exception):
    pass


def _check_normals(normals_np: np.ndarray, threshold_deg: float = 30.0) -> bool:
    """True if normals have sufficient angular diversity."""
    if len(normals_np) < 10:
        return True
    mean_normal = normals_np.mean(axis=0)
    norm = np.linalg.norm(mean_normal)
    if norm < 1e-8:
        return True  # cancel out → diverse
    mean_normal /= norm
    dots = np.clip(normals_np @ mean_normal, -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(np.abs(dots)))
    spread = np.percentile(angles_deg, 90)
    if spread < threshold_deg:
        logger.warning(
            "  Degenerate normals: 90th-pct angle spread = %.1f° < %.1f° threshold",
            spread, threshold_deg,
        )
        return False
    return True


class PipelineContext:
    """Per-model cache for mesh, OG, surface samples, visibility queries, samplers."""

    def __init__(self, model: ModelConfig):
        self.model = model
        self._raw_tm = None
        self._o3d_mesh = None
        self._sampling_og = None
        self._target_points = None
        self._normals = None
        self._surface_seed = None
        self._vis_query_cache: dict = {}
        self._sampler_cache: dict = {}

    def load_mesh(self):
        if self._raw_tm is not None:
            return self._raw_tm, self._o3d_mesh

        from shared.mesh_loader import load_and_transform_mesh

        logger.info(
            "Loading mesh: %s (target_length=%.1f)", self.model.name, self.model.target_length
        )
        self._raw_tm = load_and_transform_mesh(
            self.model.mesh_path, self.model.target_length, self.model.mesh_pose,
        )
        self._o3d_mesh = o3d.geometry.TriangleMesh()
        self._o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(self._raw_tm.vertices))
        self._o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(self._raw_tm.faces))
        self._o3d_mesh.compute_vertex_normals()

        logger.info(
            "  Mesh bounds: %s → %s",
            self._raw_tm.bounds[0].round(2), self._raw_tm.bounds[1].round(2),
        )
        return self._raw_tm, self._o3d_mesh

    def sample_surface(self, num_points: int | None = None, seed: int = 42):
        """Sample surface points + normals. Cached per (seed, num_points)."""
        n = num_points or self.model.num_surface_points
        if (
            self._target_points is not None
            and self._surface_seed == seed
            and len(self._target_points) == n
        ):
            return self._target_points, self._normals

        _, o3d_mesh = self.load_mesh()
        from shared.surface_sampler import SurfacePointSampler

        logger.info("Sampling %d surface points (seed=%d) ...", n, seed)
        sampler = SurfacePointSampler()
        pts_np, norms_np = sampler.sample(o3d_mesh, n, seed=seed)

        if not _check_normals(norms_np):
            raise DegenerateNormalsError(
                f"Model '{self.model.name}' has degenerate normals."
            )

        self._target_points = cp.asarray(pts_np, dtype=cp.float32)
        self._normals = cp.asarray(norms_np, dtype=cp.float32)
        self._surface_seed = seed
        # Invalidate caches built for the old point set
        self._vis_query_cache.clear()
        self._sampler_cache.clear()
        logger.info("  Sampled %d points. Normals OK.", n)
        return self._target_points, self._normals

    def build_sampling_og(self):
        """Surface-only OG for viewpoint sampling. Cached."""
        if self._sampling_og is not None:
            return self._sampling_og

        _, o3d_mesh = self.load_mesh()
        from visibility.sampling.utils.sampling_grid_builder import build_sampling_occupancy_grid

        min_clearance = 2 * self.model.collision_radius
        logger.info(
            "Building sampling OG (res=%.2f, clearance=%.2f) ...",
            self.model.voxel_resolution, min_clearance,
        )
        self._sampling_og, _, _ = build_sampling_occupancy_grid(
            mesh=o3d_mesh,
            frustum_far=self.model.frustum.far,
            min_clearance=min_clearance,
            resolution=self.model.voxel_resolution,
        )
        logger.info(
            "  OG shape: %s  free=%d", self._sampling_og.grid.shape, self._sampling_og.num_free
        )
        return self._sampling_og

    def build_visibility_query(self, method: str = "raycast"):
        """Visibility query. Cached per method."""
        if method in self._vis_query_cache:
            return self._vis_query_cache[method]

        _, o3d_mesh = self.load_mesh()
        # Use cached points so the query matches the caller's sample_surface() call
        if self._target_points is not None:
            target_points, normals = self._target_points, self._normals
        else:
            target_points, normals = self.sample_surface()
        from visibility.core.types import FrustumParams

        frustum_params = FrustumParams(
            fov_y=self.model.frustum.fov_y_rad,
            aspect=self.model.frustum.aspect,
            near=self.model.frustum.near,
            far=self.model.frustum.far,
        )

        if method == "raycast":
            from visibility.visibility.raycast_cuda import RaycastingVisibilityQueryCuda

            query = RaycastingVisibilityQueryCuda(
                mesh=o3d_mesh,
                target_points=target_points,
                normals=normals,
                frustum_params=frustum_params,
            )
        elif method == "epsilon":
            from visibility.visibility.epsilon_cuda import EpsilonVisibilityQueryCuda

            query = EpsilonVisibilityQueryCuda(
                target_points=target_points,
                normals=normals,
                frustum_params=frustum_params,
            )
        else:
            raise ValueError(f"Unknown visibility method: {method}")

        self._vis_query_cache[method] = query
        return query

    def build_sampler(self, strategy: str = "targeted"):
        """Viewpoint sampler. Cached per strategy."""
        if strategy in self._sampler_cache:
            return self._sampler_cache[strategy]

        _, o3d_mesh = self.load_mesh()
        if self._target_points is not None:
            target_points, normals = self._target_points, self._normals
        else:
            target_points, normals = self.sample_surface()
        og = self.build_sampling_og()

        if strategy in ("weighted", "targeted"):
            from visibility.sampling import TargetedViewpointSampler

            sampler = TargetedViewpointSampler(
                mesh=o3d_mesh,
                target_points=target_points,
                normals=normals,
                frustum_far=self.model.frustum.far,
                collision_radius=self.model.collision_radius,
                occupancy_grid=og,
            )
        elif strategy == "optimizing":
            from visibility.sampling import CMAESBackend, OptimizingSampler

            base_sampler = self.build_sampler("targeted")
            sampler = OptimizingSampler(
                mesh=o3d_mesh,
                target_points=target_points,
                normals=normals,
                frustum_far=self.model.frustum.far,
                collision_radius=self.model.collision_radius,
                occupancy_grid=og,
                backend=CMAESBackend(),
                random_sampler=base_sampler,
            )
        else:
            raise ValueError(f"Unknown sampler strategy: {strategy}")

        self._sampler_cache[strategy] = sampler
        return sampler

    @property
    def mesh_bounds(self):
        tm, _ = self.load_mesh()
        return np.asarray(tm.bounds[0], dtype=float), np.asarray(tm.bounds[1], dtype=float)
