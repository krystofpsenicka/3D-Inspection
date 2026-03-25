"""Base class for viewpoint sampling with shared GPU infrastructure."""

import logging
import time

import cupy as cp
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from typing import List, Tuple

from ...core.constants import NORM_EPS, GPU_NN_CHUNK_SIZE
from ..utils.occupancy import build_occupancy_grid, precompute_sdf_grid
from ..utils.free_space import build_free_space, sample_from_free_space

logger = logging.getLogger(__name__)


class ViewpointSamplerBase:
    """
    Base class for viewpoint sampling (GPU-only).

    Provides shared infrastructure: occupancy grid, SDF grid, GPU collision
    helpers, and the core free-space sampling pipeline.

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
            logger.warning("[ViewpointSampler] No target points available.")

        # Mesh bounding box
        vertices = np.asarray(mesh.vertices)
        self.mesh_bbox_min = vertices.min(axis=0)
        self.mesh_bbox_max = vertices.max(axis=0)

        # KDTree for nearest-surface queries (CPU fallback)
        self._kdtree = KDTree(target_points)

        # Lazy caches for free-space data
        self._feasible_outside = None
        self._feasible_inside = None
        self._feasible_outside_curv = None
        self._feasible_inside_curv = None

        # Raycasting scene (kept for CPU fallback and legacy methods)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        logger.info("[ViewpointSampler] Created O3D RaycastingScene.")

        # Auto-build OG when none provided
        if occupancy_grid is None:
            logger.info("[ViewpointSampler] No occupancy grid provided — building surface-only OG from mesh...")
            occupancy_grid = build_occupancy_grid(mesh, frustum_far, self.min_clearance)
            self._occupancy_grid = occupancy_grid

        # Watertight warning
        if not mesh.is_watertight():
            logger.warning(
                "[ViewpointSampler] Mesh is not watertight — SDF signs may be "
                "unreliable. Free-space sampling results could be incorrect."
            )

        self._sdf_grid = None          # CPU float32 array
        self._sdf_grid_gpu = None      # CuPy array
        self._og_grid_gpu = None       # CuPy bool array (inflated)
        self._og_origin_gpu = None     # CuPy float32 (3,)
        self._target_points_gpu = None # CuPy float32 (M, 3)

        t0 = time.perf_counter()
        self._sdf_grid = precompute_sdf_grid(self._occupancy_grid)
        self._sdf_grid_gpu = cp.asarray(self._sdf_grid)
        self._og_grid_gpu = cp.asarray(occupancy_grid.grid)
        self._og_origin_gpu = cp.asarray(occupancy_grid.origin, dtype=cp.float32)
        self._og_resolution = float(occupancy_grid.resolution)
        self._target_points_gpu = cp.asarray(target_points, dtype=cp.float32)
        dt = time.perf_counter() - t0
        gpu_mb = cp.get_default_memory_pool().used_bytes() / 1e6
        logger.info("[ViewpointSampler] GPU init: %.2fs (SDF precompute + transfer). "
                    "GPU memory: %.1f MB", dt, gpu_mb)

    # ── SDF / collision helpers ─────────────────────────────────────────

    def _sdf_lookup(self, positions_gpu):
        """Look up SDF values for (N, 3) world positions on GPU.

        Out-of-bounds positions get +inf (exterior).
        """
        ijk = cp.floor(
            (positions_gpu - self._og_origin_gpu) / self._og_resolution
        ).astype(cp.int32)

        shape = cp.asarray(self._sdf_grid_gpu.shape, dtype=cp.int32)
        in_bounds = (
            (ijk[:, 0] >= 0) & (ijk[:, 0] < shape[0]) &
            (ijk[:, 1] >= 0) & (ijk[:, 1] < shape[1]) &
            (ijk[:, 2] >= 0) & (ijk[:, 2] < shape[2])
        )

        sdf = cp.full(len(positions_gpu), cp.inf, dtype=cp.float32)
        if cp.any(in_bounds):
            valid_ijk = ijk[in_bounds]
            sdf[in_bounds] = self._sdf_grid_gpu[
                valid_ijk[:, 0], valid_ijk[:, 1], valid_ijk[:, 2]
            ]
        return sdf

    def _is_free(self, positions_gpu):
        """Check OG collision for (N, 3) world positions on GPU.

        Out-of-bounds → False (not free).
        """
        ijk = cp.floor(
            (positions_gpu - self._og_origin_gpu) / self._og_resolution
        ).astype(cp.int32)

        shape = cp.asarray(self._og_grid_gpu.shape, dtype=cp.int32)
        in_bounds = (
            (ijk[:, 0] >= 0) & (ijk[:, 0] < shape[0]) &
            (ijk[:, 1] >= 0) & (ijk[:, 1] < shape[1]) &
            (ijk[:, 2] >= 0) & (ijk[:, 2] < shape[2])
        )

        result = cp.zeros(len(positions_gpu), dtype=cp.bool_)
        if cp.any(in_bounds):
            valid_ijk = ijk[in_bounds]
            result[in_bounds] = ~self._og_grid_gpu[
                valid_ijk[:, 0], valid_ijk[:, 1], valid_ijk[:, 2]
            ]
        return result

    def _nearest_neighbor(self, query_gpu, targets_gpu):
        """Brute-force nearest neighbor on GPU using squared-distance decomposition.

        Returns (distances, indices) for each query point.
        Chunks queries to limit memory usage.
        """
        n_query = len(query_gpu)

        # Pre-compute ||b||² for all targets
        targets_sq = cp.sum(targets_gpu ** 2, axis=1)  # (M,)

        indices = cp.empty(n_query, dtype=cp.int32)
        distances = cp.empty(n_query, dtype=cp.float32)

        for start in range(0, n_query, GPU_NN_CHUNK_SIZE):
            end = min(start + GPU_NN_CHUNK_SIZE, n_query)
            q = query_gpu[start:end]  # (chunk, 3)
            q_sq = cp.sum(q ** 2, axis=1, keepdims=True)  # (chunk, 1)
            dist_sq = q_sq + targets_sq[cp.newaxis, :] - 2.0 * q @ targets_gpu.T
            cp.maximum(dist_sq, 0.0, out=dist_sq)
            chunk_idx = cp.argmin(dist_sq, axis=1)
            chunk_dist = cp.sqrt(dist_sq[cp.arange(len(chunk_idx)), chunk_idx])
            indices[start:end] = chunk_idx
            distances[start:end] = chunk_dist

        return distances, indices

    # ── Free-space pipeline (delegates to free_space module) ───────────

    def _build_free_space(self, side: str, min_dist: float, max_dist: float,
                          curvature_weighting: bool = False):
        """Build feasible positions. Returns (centers_gpu, weights_gpu, coarse_res)."""
        return build_free_space(
            self._occupancy_grid, self._sdf_grid_gpu,
            self._og_origin_gpu, self._og_resolution,
            self._target_points_gpu, self.normals,
            self.free_space_resolution,
            side, min_dist, max_dist,
            curvature_weighting=curvature_weighting,
            sdf_lookup_fn=self._sdf_lookup,
        )

    def _sample_from_free_space(self, centers_gpu, weights_gpu,
                                coarse_res: float, num_candidates: int,
                                max_dir_noise_rad: float = 0.0,
                                direction_targets_gpu=None,
                                curvature_weighting: bool = False,
                                ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """GPU-accelerated viewpoint sampling from feasible positions."""
        return sample_from_free_space(
            centers_gpu, weights_gpu, coarse_res, num_candidates,
            self._target_points_gpu, self.normals,
            max_dir_noise_rad=max_dir_noise_rad,
            direction_targets_gpu=direction_targets_gpu,
            curvature_weighting=curvature_weighting,
        )

    # ── Public: feasible data access ────────────────────────────────────

    def get_feasible_data(self, side: str = "outside",
                          min_distance: float | None = None,
                          max_distance_offset: float = 0.95,
                          curvature_weighting: bool = False,
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (positions [N,3], weights [N]) for the feasible free-space region.

        Weights are normalized sdf² sampling probabilities.
        Triggers the build if not already cached.
        """
        min_d = min_distance if min_distance is not None else self.min_clearance
        max_d = max_distance_offset * self.frustum_far

        cache_attr = (f"_feasible_{side}_curv" if curvature_weighting
                      else f"_feasible_{side}")
        cached = getattr(self, cache_attr)

        if cached is None:
            cached = self._build_free_space(side, min_d, max_d,
                                            curvature_weighting=curvature_weighting)
            setattr(self, cache_attr, cached)

        centers_gpu, weights_gpu, _ = cached
        return cp.asnumpy(centers_gpu), cp.asnumpy(weights_gpu)

    # ── Internal: get or build cached free space ──────────────────────

    def _get_cached_free_space(self, side: str, min_distance: float | None,
                                max_distance_offset: float,
                                curvature_weighting: bool):
        """Get or build cached free-space data. Returns (centers_gpu, weights_gpu, coarse_res)."""
        min_d = min_distance if min_distance is not None else self.min_clearance
        max_d = max_distance_offset * self.frustum_far

        cache_attr = (f"_feasible_{side}_curv" if curvature_weighting
                      else f"_feasible_{side}")
        cached = getattr(self, cache_attr)

        if cached is None:
            t0 = time.perf_counter()
            cached = self._build_free_space(side, min_d, max_d,
                                            curvature_weighting=curvature_weighting)
            dt = time.perf_counter() - t0
            logger.info("[ViewpointSampler] _build_free_space: %.3fs", dt)
            setattr(self, cache_attr, cached)

        return cached
