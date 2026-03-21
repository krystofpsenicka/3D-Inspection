# DEPRECATED: This module is a stale copy.
# The canonical implementation is visibility/sampling/base.py which includes
# curvature weighting, targeted resampling, and sample_with_resampling().
# Use: from visibility.sampling import ViewpointSampler

import logging
import time
from math import ceil

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import KDTree
from typing import List, Tuple, Optional

from .types import FrustumParams
from .constants import NORM_EPS, DEFAULT_MAX_DIR_NOISE_RAD, KNN_DIRECTION_K, GPU_NN_CHUNK_SIZE

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False


class ViewpointSampler:
    """
    Generates candidate viewpoints for 3D inspection (GPU-only).

    Requires an occupancy grid and CuPy.  Uses the occupancy grid to
    identify collision-free voxels, then samples with distance² weighting
    and sub-voxel jitter.  A precomputed EDT-based SDF grid provides O(1)
    signed-distance lookups on GPU.
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

        # Raycasting scene (kept for CPU fallback and legacy methods)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        logger.info("[ViewpointSampler] Created O3D RaycastingScene.")

        # ── GPU setup ─────────────────────────────────────────────────────
        if not _CUPY_AVAILABLE:
            raise RuntimeError(
                "ViewpointSampler requires CuPy for GPU-accelerated sampling."
            )

        # Auto-build OG when none provided
        if occupancy_grid is None:
            logger.info("[ViewpointSampler] No occupancy grid provided — building surface-only OG from mesh...")
            occupancy_grid = self._build_occupancy_grid()
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
        self._precompute_sdf_grid()
        self._sdf_grid_gpu = cp.asarray(self._sdf_grid)
        self._og_grid_gpu = cp.asarray(occupancy_grid.grid)
        self._og_origin_gpu = cp.asarray(occupancy_grid.origin, dtype=cp.float32)
        self._og_resolution = float(occupancy_grid.resolution)
        self._target_points_gpu = cp.asarray(target_points, dtype=cp.float32)
        dt = time.perf_counter() - t0
        gpu_mb = cp.get_default_memory_pool().used_bytes() / 1e6
        logger.info("[ViewpointSampler] GPU init: %.2fs (SDF precompute + transfer). "
                    "GPU memory: %.1f MB", dt, gpu_mb)

    # ── Auto-build occupancy grid ─────────────────────────────────────

    def _build_occupancy_grid(self, resolution: float = 0.10):
        """Build a surface-only OccupancyGrid from the Open3D mesh.

        Used when the caller does not supply an occupancy grid.
        Follows the same voxelization pattern as VRP/core/occupancy_grid.py.
        """
        from shared.occupancy_grid import OccupancyGrid, inflate_grid

        vertices = np.asarray(self.mesh.vertices)
        faces = np.asarray(self.mesh.triangles)
        tm = trimesh.Trimesh(vertices=vertices, faces=faces)

        # Grid bounds: extend by frustum_far + min_clearance so the SDF
        # covers the full sampling shell around the mesh.
        padding = self.frustum_far + self.min_clearance
        bounds_min = vertices.min(axis=0) - padding
        bounds_max = vertices.max(axis=0) + padding
        origin = bounds_min.copy()
        grid_shape = np.ceil((bounds_max - bounds_min) / resolution).astype(int)
        grid_shape = np.maximum(grid_shape, 1)

        # Surface-only voxelization
        vg_surface = tm.voxelized(pitch=resolution)
        vg_filled = vg_surface.fill()

        def _map_voxels_to_grid(vg):
            """Map a trimesh VoxelGrid into the padded grid."""
            result = np.zeros(grid_shape, dtype=bool)
            vox_matrix = vg.matrix
            vox_world_origin = np.asarray(vg.transform[:3, 3])
            vox_origin_ijk = np.floor(
                (vox_world_origin - origin) / resolution
            ).astype(int)
            dst_min = np.maximum(vox_origin_ijk, 0)
            src_min = np.maximum(-vox_origin_ijk, 0)
            dst_max = np.minimum(vox_origin_ijk + np.array(vox_matrix.shape), grid_shape)
            src_max = src_min + (dst_max - dst_min)
            result[
                dst_min[0]:dst_max[0],
                dst_min[1]:dst_max[1],
                dst_min[2]:dst_max[2],
            ] = vox_matrix[
                src_min[0]:src_max[0],
                src_min[1]:src_max[1],
                src_min[2]:src_max[2],
            ]
            return result

        raw_grid = _map_voxels_to_grid(vg_surface)
        filled_raw_grid = _map_voxels_to_grid(vg_filled)

        logger.info("[ViewpointSampler] Auto-built OG: shape=%s, surface=%d voxels, "
                    "filled=%d voxels, res=%.3fm",
                    grid_shape, int(raw_grid.sum()), int(filled_raw_grid.sum()),
                    resolution)

        # Inflate by min_clearance so free-space voxels are guaranteed
        # to have at least min_clearance distance from the mesh surface.
        inflation_voxels = max(1, ceil(self.min_clearance / resolution))
        inflated = inflate_grid(raw_grid, inflation_voxels)
        logger.info("[ViewpointSampler] Inflated OG: %d occupied, %d free (inflation=%d voxels)",
                    int(inflated.sum()), int((~inflated).sum()), inflation_voxels)

        return OccupancyGrid(
            grid=inflated,
            origin=origin,
            resolution=resolution,
            raw_grid=raw_grid,
            filled_raw_grid=filled_raw_grid,
        )

    # ── SDF precomputation (two-EDT) ──────────────────────────────────

    def _precompute_sdf_grid(self):
        """Build a volumetric SDF grid using the two-EDT method.

        Uses ``og.filled_raw_grid`` (trimesh ray-based flood fill) for robust
        interior detection.  Convention: positive outside, negative inside.
        """
        from scipy.ndimage import distance_transform_edt

        og = self._occupancy_grid

        # Prefer filled_raw_grid (from trimesh vg.fill()); fall back to raw_grid
        if og.filled_raw_grid is not None:
            filled = og.filled_raw_grid
        elif og.raw_grid is not None:
            logger.warning("[ViewpointSampler] filled_raw_grid not available — "
                           "falling back to raw_grid with binary_fill_holes "
                           "(less robust for non-watertight meshes).")
            from scipy.ndimage import binary_fill_holes
            filled = binary_fill_holes(og.raw_grid)
        else:
            logger.warning("[ViewpointSampler] No raw grid available — "
                           "falling back to inflated grid for SDF.")
            filled = og.grid

        t0 = time.perf_counter()
        # Distance from each free (exterior) voxel to nearest occupied voxel
        outside_dist = distance_transform_edt(~filled).astype(np.float32) * og.resolution
        # Distance from each occupied voxel to nearest free voxel
        inside_dist = distance_transform_edt(filled).astype(np.float32) * og.resolution
        # Convention: positive outside, negative inside
        self._sdf_grid = outside_dist - inside_dist
        dt = time.perf_counter() - t0
        logger.info("[ViewpointSampler] SDF grid computed (two-EDT): shape=%s, "
                    "%.2fs, range=[%.2f, %.2f]m",
                    self._sdf_grid.shape, dt,
                    float(self._sdf_grid.min()), float(self._sdf_grid.max()))

    # ── SDF / collision helpers ─────────────────────────────────────────

    def _sdf_lookup(self, positions_gpu):
        """Look up SDF values for (N, 3) world positions on GPU.

        Out-of-bounds positions get +inf (exterior).
        """
        og = self._occupancy_grid
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
        n_target = len(targets_gpu)

        # Pre-compute ||b||² for all targets
        targets_sq = cp.sum(targets_gpu ** 2, axis=1)  # (M,)

        indices = cp.empty(n_query, dtype=cp.int32)
        distances = cp.empty(n_query, dtype=cp.float32)

        for start in range(0, n_query, GPU_NN_CHUNK_SIZE):
            end = min(start + GPU_NN_CHUNK_SIZE, n_query)
            q = query_gpu[start:end]  # (chunk, 3)
            # ||a||²
            q_sq = cp.sum(q ** 2, axis=1, keepdims=True)  # (chunk, 1)
            # ||a - b||² = ||a||² + ||b||² - 2·a·b
            dist_sq = q_sq + targets_sq[cp.newaxis, :] - 2.0 * q @ targets_gpu.T  # (chunk, M)
            # Clamp numerical noise
            cp.maximum(dist_sq, 0.0, out=dist_sq)
            chunk_idx = cp.argmin(dist_sq, axis=1)
            chunk_dist = cp.sqrt(dist_sq[cp.arange(len(chunk_idx)), chunk_idx])
            indices[start:end] = chunk_idx
            distances[start:end] = chunk_dist

        return distances, indices

    def _knn_centroid_direction(self, query_gpu, targets_gpu, k: int = KNN_DIRECTION_K):
        """Compute viewing direction as centroid of K nearest surface points (GPU).

        Returns normalised direction vectors (N, 3).
        """
        n_query = len(query_gpu)
        n_target = len(targets_gpu)
        k = min(k, n_target)

        targets_sq = cp.sum(targets_gpu ** 2, axis=1)  # (M,)
        all_dirs = cp.empty((n_query, 3), dtype=cp.float32)

        for start in range(0, n_query, GPU_NN_CHUNK_SIZE):
            end = min(start + GPU_NN_CHUNK_SIZE, n_query)
            q = query_gpu[start:end]
            q_sq = cp.sum(q ** 2, axis=1, keepdims=True)
            dist_sq = q_sq + targets_sq[cp.newaxis, :] - 2.0 * q @ targets_gpu.T
            cp.maximum(dist_sq, 0.0, out=dist_sq)
            # Get K nearest indices
            knn_idx = cp.argpartition(dist_sq, k, axis=1)[:, :k]
            # Compute centroid of K nearest points for each query
            # Shape: (chunk, k, 3)
            knn_pts = targets_gpu[knn_idx]
            centroid = knn_pts.mean(axis=1)  # (chunk, 3)
            dirs = centroid - q
            all_dirs[start:end] = dirs

        norms = cp.linalg.norm(all_dirs, axis=1, keepdims=True)
        norms = cp.maximum(norms, NORM_EPS)
        return all_dirs / norms

    @staticmethod
    def _apply_angular_noise(directions_gpu, max_angle_rad):
        """GPU Rodrigues rotation — port of CPU _apply_angular_noise."""
        n = len(directions_gpu)
        if max_angle_rad < 1e-8 or n == 0:
            return directions_gpu.copy()

        angles = cp.random.uniform(0, max_angle_rad, size=n)

        rand_vec = cp.random.randn(n, 3, dtype=cp.float32)
        dot = cp.sum(rand_vec * directions_gpu, axis=1, keepdims=True)
        perp = rand_vec - dot * directions_gpu
        perp_norm = cp.linalg.norm(perp, axis=1, keepdims=True)

        degenerate = (perp_norm < 1e-8).ravel()
        if cp.any(degenerate):
            alt = cp.zeros_like(directions_gpu[degenerate])
            alt[:, 0] = -directions_gpu[degenerate, 1]
            alt[:, 1] = directions_gpu[degenerate, 0]
            alt_norm = cp.linalg.norm(alt, axis=1, keepdims=True)
            alt_norm = cp.maximum(alt_norm, NORM_EPS)
            perp[degenerate] = alt / alt_norm
            perp_norm[degenerate] = 1.0
        perp = perp / cp.maximum(perp_norm, NORM_EPS)

        cos_a = cp.cos(angles)[:, cp.newaxis]
        sin_a = cp.sin(angles)[:, cp.newaxis]
        cross = cp.cross(perp, directions_gpu)
        dot_kv = cp.sum(perp * directions_gpu, axis=1, keepdims=True)
        rotated = directions_gpu * cos_a + cross * sin_a + perp * dot_kv * (1 - cos_a)

        rotated_norm = cp.linalg.norm(rotated, axis=1, keepdims=True)
        rotated = rotated / cp.maximum(rotated_norm, NORM_EPS)
        return rotated

    # ── Free-space pipeline ────────────────────────────────────────────

    def _build_free_space(self, side: str, min_dist: float, max_dist: float):
        """Build feasible positions using EDT-SDF grid lookup on GPU.

        Returns (feasible_centers_gpu, weights_gpu, coarse_resolution).
        """
        from shared.occupancy_grid import downsample_occupancy_grid, OccupancyGrid

        og = self._occupancy_grid
        coarse_res = self.free_space_resolution

        # 1. Downsample OG to coarse resolution (CPU — fast)
        coarse_grid, coarse_origin, actual_res = downsample_occupancy_grid(
            og.grid, og.origin, og.resolution, coarse_res
        )
        coarse_og = OccupancyGrid(
            grid=coarse_grid, origin=coarse_origin, resolution=actual_res
        )

        # 2. Extract free voxel centers
        free_ijk = np.argwhere(~coarse_grid)
        if len(free_ijk) == 0:
            logger.warning("[ViewpointSampler] No free voxels in coarse grid.")
            return cp.empty((0, 3), dtype=cp.float32), cp.empty(0, dtype=cp.float32), actual_res

        free_centers = coarse_og.voxel_to_world(free_ijk)
        logger.info("[ViewpointSampler] Coarse grid: %s, %d free voxels",
                    coarse_grid.shape, len(free_ijk))

        # 3. Transfer to GPU and look up SDF
        centers_gpu = cp.asarray(free_centers, dtype=cp.float32)
        sdf = self._sdf_lookup(centers_gpu)

        # 4. Filter by side AND distance range
        if side == "outside":
            mask = (sdf >= min_dist) & (sdf <= max_dist)
        else:  # inside
            mask = (sdf <= -min_dist) & (sdf >= -max_dist)

        feasible_gpu = centers_gpu[mask]
        feasible_sdf = sdf[mask]
        n_feasible = int(len(feasible_gpu))
        logger.info("[ViewpointSampler] After SDF filter (%s): %d / %d positions",
                    side, n_feasible, len(free_centers))

        if n_feasible == 0:
            return feasible_gpu, cp.empty(0, dtype=cp.float32), actual_res

        # 5. Compute sampling weights: w = sdf² (footprint area ∝ d²)
        weights = feasible_sdf ** 2
        weights = weights / weights.sum()

        logger.info("[ViewpointSampler] %s free space: %d feasible positions "
                    "(coarse res %.2fm)", side.capitalize(), n_feasible, actual_res)

        return feasible_gpu, weights, actual_res

    def _sample_from_free_space(self, centers_gpu, weights_gpu,
                                     coarse_res: float, num_candidates: int,
                                     max_dir_noise_rad: float = DEFAULT_MAX_DIR_NOISE_RAD,
                                     ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """GPU-accelerated viewpoint sampling from feasible positions."""
        n_feasible = int(len(centers_gpu))
        if n_feasible == 0:
            logger.warning("[ViewpointSampler] No feasible positions — returning empty.")
            return []

        # 1. Weighted random sample on CPU (CuPy lacks weighted choice)
        weights_cpu = cp.asnumpy(weights_gpu)
        indices_cpu = np.random.choice(n_feasible, num_candidates,
                                       replace=True, p=weights_cpu)
        sampled_gpu = centers_gpu[cp.asarray(indices_cpu)]

        # 2. Sub-voxel jitter on GPU
        jitter = cp.random.uniform(
            -coarse_res / 2, coarse_res / 2,
            size=(num_candidates, 3), dtype=cp.float32
        )
        sampled_gpu = sampled_gpu + jitter

        # 3. K-nearest-neighbor centroid for viewing direction (GPU)
        base_dirs = self._knn_centroid_direction(sampled_gpu, self._target_points_gpu)
        norms = cp.linalg.norm(base_dirs, axis=1, keepdims=True)
        norms = cp.maximum(norms, NORM_EPS)
        base_dirs = base_dirs / norms

        # 4. Angular noise via Rodrigues rotation (GPU)
        directions_gpu = self._apply_angular_noise(base_dirs, max_dir_noise_rad)

        # 5. Transfer back to CPU
        sampled_cpu = cp.asnumpy(sampled_gpu)
        directions_cpu = cp.asnumpy(directions_gpu)

        candidates = list(zip(sampled_cpu, directions_cpu))
        logger.info("[ViewpointSampler] Generated %d candidates from free space (GPU).",
                    len(candidates))
        return candidates


    # ── Public: feasible data access ────────────────────────────────────

    def get_feasible_data(self, side: str = "outside",
                          min_distance: float | None = None,
                          max_distance_offset: float = 0.95,
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (positions [N,3], weights [N]) for the feasible free-space region.

        Weights are normalized sdf² sampling probabilities.
        Triggers the build if not already cached.
        """
        min_d = min_distance if min_distance is not None else self.min_clearance
        max_d = max_distance_offset * self.frustum_far

        cache_attr = f"_feasible_{side}"
        cached = getattr(self, cache_attr)

        if cached is None:
            cached = self._build_free_space(side, min_d, max_d)
            setattr(self, cache_attr, cached)

        centers_gpu, weights_gpu, _ = cached
        return cp.asnumpy(centers_gpu), cp.asnumpy(weights_gpu)

    # ── Public: dispatchers ──────────────────────────────────────────────

    def sample_outside_mesh(self, num_candidates: int,
                            min_distance: float | None = None,
                            max_distance_offset: float = 0.95,
                            ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Sample candidate viewpoints outside the mesh (GPU, OG-based)."""
        min_d = min_distance if min_distance is not None else self.min_clearance
        max_d = max_distance_offset * self.frustum_far
        return self._sample_og_free_space(num_candidates, min_d, max_d, "outside")

    def sample_inside_mesh(self, num_candidates: int,
                           min_distance: float | None = None,
                           max_distance_offset: float = 0.95,
                           ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Sample candidate viewpoints inside the mesh (GPU, OG-based)."""
        min_d = min_distance if min_distance is not None else self.min_clearance
        max_d = max_distance_offset * self.frustum_far
        return self._sample_og_free_space(num_candidates, min_d, max_d, "inside")

    # ── OG-based sampling ──────────────────────────────────────────────

    def _sample_og_free_space(self, num_candidates: int,
                              min_distance: float, max_distance: float,
                              side: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        """OG-based free-space sampling on GPU (works for both inside and outside)."""
        if self.num_points == 0:
            return []

        logger.info("[ViewpointSampler] Sampling %d viewpoints from %s mesh "
                    "(OG free-space GPU, clearance=%.2f m, range=%.2f–%.2f m) …",
                    num_candidates, side.upper(),
                    min_distance, min_distance, max_distance)

        cache_attr = f"_feasible_{side}"
        cached = getattr(self, cache_attr)

        if cached is None:
            t0 = time.perf_counter()
            cached = self._build_free_space(side, min_distance, max_distance)
            dt = time.perf_counter() - t0
            logger.info("[ViewpointSampler] _build_free_space: %.3fs", dt)
            setattr(self, cache_attr, cached)

        centers_gpu, weights_gpu, coarse_res = cached

        logger.info("[ViewpointSampler] %s free space: %d feasible positions",
                    side.capitalize(), int(len(centers_gpu)))

        t0 = time.perf_counter()
        result = self._sample_from_free_space(
            centers_gpu, weights_gpu, coarse_res, num_candidates)
        dt = time.perf_counter() - t0
        logger.info("[ViewpointSampler] _sample_from_free_space: %.3fs", dt)
        return result


