"""Base class for viewpoint sampling with shared GPU state."""

import logging
import time

import cupy as cp
import numpy as np
import open3d as o3d
from typing import Tuple
from ..utils.sampling_grid_builder import build_sampling_occupancy_grid, build_sdf_grid
from ..utils.sampling_space_builder import build_sampling_space

logger = logging.getLogger(__name__)


class ViewpointSamplerBase:
    """
    Base class for viewpoint sampling.

    Provides shared state: occupancy grid, SDF grid, and cached
    feasible sampling data.

    Subclasses implement specific sampling strategies.
    """

    def __init__(self, mesh: o3d.geometry.TriangleMesh, target_points: np.ndarray,
                 normals: np.ndarray, frustum_far: float,
                 collision_radius: float = 0.0,
                 occupancy_grid=None,
                 free_space_resolution: float = 0.5):
        self.mesh = mesh
        self.target_points = target_points
        self.normals = normals
        self.frustum_far = frustum_far
        self.num_points = len(target_points)
        self.collision_radius = collision_radius
        self.min_clearance = 2.0 * collision_radius
        self.free_space_resolution = free_space_resolution
        self._occupancy_grid = occupancy_grid

        if self.num_points == 0:
            logger.warning("[ViewpointSamplerBase] No target points available.")

        # Auto-build OG when none provided
        if occupancy_grid is None:
            logger.info("[ViewpointSamplerBase] No occupancy grid provided — building surface-only OG from mesh...")
            self._occupancy_grid = build_sampling_occupancy_grid(mesh, frustum_far, self.min_clearance)

        # Watertight warning
        if not mesh.is_watertight():
            logger.warning(
                "[ViewpointSamplerBase] Mesh is not watertight — SDF signs may be "
                "unreliable. Free-space sampling results could be incorrect."
            )

        # Cache for feasible free-space data: (side, curvature_weighting) -> (centers_gpu, weights_gpu, coarse_res)
        self._feasible_cache = {}

        self._sdf_grid = None          # CPU float32 array
        self._sdf_grid_gpu = None      # CuPy array
        self._og_grid_gpu = None       # CuPy bool array (inflated)
        self._og_origin_gpu = None     # CuPy float32 (3,)
        self._target_points_gpu = None # CuPy float32 (M, 3)

        t0 = time.perf_counter()
        self._sdf_grid = build_sdf_grid(self._occupancy_grid)
        self._sdf_grid_gpu = cp.asarray(self._sdf_grid)
        self._og_grid_gpu = cp.asarray(self._occupancy_grid.grid)
        self._og_origin_gpu = cp.asarray(self._occupancy_grid.origin, dtype=cp.float32)
        self._og_resolution = float(self._occupancy_grid.resolution)
        self._target_points_gpu = cp.asarray(target_points, dtype=cp.float32)
        dt = time.perf_counter() - t0
        gpu_mb = cp.get_default_memory_pool().used_bytes() / 1e6
        logger.info("[ViewpointSamplerBase] GPU init: %.2fs (SDF precompute + transfer). "
                    "GPU memory: %.1f MB", dt, gpu_mb)

    # ── Feasible data access ────────────────────────────────────

    def get_feasible_sampling_data(self, side: str = "outside",
                                   min_distance: float | None = None,
                                   max_distance_offset: float = 0.95,
                                   curvature_weighting: bool = False,
                                   ) -> Tuple[cp.ndarray, cp.ndarray, float]:
        """Return cached (centers_gpu, weights_gpu, coarse_res) for the given side.

        Weights are normalized sdf² and curvature biased probabilities on GPU.
        Triggers the computation if not already cached.
        """
        if min_distance is None:
            min_distance = self.min_clearance
        max_distance = max_distance_offset * self.frustum_far

        key = (side, curvature_weighting)
        if key not in self._feasible_cache:
            t0 = time.perf_counter()
            self._feasible_cache[key] = build_sampling_space(
                self._occupancy_grid, self._sdf_grid_gpu,
                self._target_points_gpu, self.normals,
                self.free_space_resolution,
                side, min_distance, max_distance,
                curvature_weighting=curvature_weighting,
            )
            dt = time.perf_counter() - t0
            logger.info("[ViewpointSamplerBase] build_sampling_space: %.3fs", dt)

        return self._feasible_cache[key]
