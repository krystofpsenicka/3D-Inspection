"""Base class for viewpoint sampling with shared GPU state."""

import logging
import time
from abc import ABC, abstractmethod

import cupy as cp
import open3d as o3d
from typing import Tuple
from ..utils.sampling_grid_builder import build_sampling_occupancy_grid, build_sdf_grid
from ..utils.sampling_space_builder import build_sampling_space
from ...core.types import Side

logger = logging.getLogger(__name__)


class ViewpointSamplerBase(ABC):
    """
    Base class for viewpoint sampling.

    Provides shared state: occupancy grid, SDF grid, and cached
    feasible sampling data. All arrays are GPU-resident (CuPy).

    Subclasses implement specific sampling strategies.
    """

    def __init__(self, mesh: o3d.geometry.TriangleMesh, target_points: cp.ndarray,
                 normals: cp.ndarray, frustum_far: float,
                 collision_radius: float = 0.0,
                 occupancy_grid=None,
                 free_space_resolution: float = 0.5):
        self.mesh = mesh
        self.target_points = target_points    # CuPy float32
        self.normals = normals                # CuPy float32
        self.frustum_far = frustum_far
        self.num_points = len(target_points)
        self.collision_radius = collision_radius
        self.min_clearance = 2.0 * collision_radius
        self.free_space_resolution = free_space_resolution
        self._occupancy_grid = occupancy_grid

        if self.num_points == 0:
            logger.warning("[ViewpointSamplerBase] No target points available.")

        if occupancy_grid is None:
            logger.info("[ViewpointSamplerBase] No occupancy grid provided — building surface-only OG from mesh...")
            self._occupancy_grid = build_sampling_occupancy_grid(mesh, frustum_far, self.min_clearance)

        if not mesh.is_watertight():
            logger.warning(
                "[ViewpointSamplerBase] Mesh is not watertight — SDF signs may be "
                "unreliable. Free-space sampling results could be incorrect."
            )

        # Sphere restriction for expansion sampling
        self._sphere_center = None   # CuPy float32 (3,) or None
        self._sphere_radius = None   # float or None

        # Cache for feasible free-space data
        self._feasible_cache = {}

        t0 = time.perf_counter()
        self._sdf_grid_gpu = build_sdf_grid(self._occupancy_grid)
        dt = time.perf_counter() - t0
        gpu_mb = cp.get_default_memory_pool().used_bytes() / 1e6
        logger.info("[ViewpointSamplerBase] GPU init: %.2fs (SDF precompute). "
                    "GPU memory: %.1f MB", dt, gpu_mb)

    # ── Sphere restriction ──────────────────────────────────────

    def restrict_to_sphere(self, center: cp.ndarray, radius: float):
        """Restrict subsequent sampling to a sphere around *center*."""
        self._sphere_center = center.astype(cp.float32)
        self._sphere_radius = radius

    def clear_restriction(self):
        """Remove any active sphere restriction."""
        self._sphere_center = None
        self._sphere_radius = None

    # ── Feasible data access ────────────────────────────────────

    def get_feasible_sampling_data(self, side: Side = Side.OUTSIDE,
                                   min_distance: float | None = None,
                                   max_distance_offset: float = 0.95,
                                   curvature_weighting: bool = False,
                                   ) -> Tuple[cp.ndarray, cp.ndarray, float]:
        """Return cached (centers_gpu, weights_gpu, coarse_res) for the given side."""
        if min_distance is None:
            min_distance = self.min_clearance
        max_distance = max_distance_offset * self.frustum_far

        key = (side, curvature_weighting)
        if key not in self._feasible_cache:
            t0 = time.perf_counter()
            self._feasible_cache[key] = build_sampling_space(
                self._occupancy_grid, self._sdf_grid_gpu,
                self.target_points, self.normals,
                self.free_space_resolution,
                side, min_distance, max_distance,
                curvature_weighting=curvature_weighting,
            )
            dt = time.perf_counter() - t0
            logger.info("[ViewpointSamplerBase] build_sampling_space: %.3fs", dt)

        centers_gpu, weights_gpu, coarse_res = self._feasible_cache[key]

        if self._sphere_center is not None:
            dists = cp.linalg.norm(centers_gpu - self._sphere_center, axis=1)
            mask = dists <= self._sphere_radius
            centers_gpu = centers_gpu[mask]
            weights_gpu = weights_gpu[mask]
            w_sum = cp.sum(weights_gpu)
            if w_sum > 0:
                weights_gpu = weights_gpu / w_sum

        return centers_gpu, weights_gpu, coarse_res


class ProbabilisticSampler(ViewpointSamplerBase, ABC):
    """Base class for samplers that generate N random candidate viewpoints."""

    @abstractmethod
    def sample(self, num_candidates: int, **kwargs) -> Tuple[cp.ndarray, cp.ndarray]:
        """Sample candidate viewpoints.

        Returns:
            (positions_gpu, rotmats_gpu) — CuPy arrays (N, 3) and (N, 3, 3).
        """
