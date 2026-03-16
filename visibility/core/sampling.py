import logging
import time

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from typing import List, Tuple, Optional

from .types import FrustumParams

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False


class ViewpointSampler:
    """
    Generates candidate viewpoints for 3D inspection.

    Two tiers (checked in order):
    1. **OG-based free-space** (occupancy_grid is not None): uses the
       occupancy grid to identify collision-free voxels, then samples
       with distance² weighting and sub-voxel jitter.
    2. **SDF-based free-space** (no OG): builds its own regular 3D grid,
       computes SDF, and samples from feasible positions.

    When an occupancy grid is provided and CuPy is available, the OG-based
    pipeline runs on GPU using a precomputed EDT-based SDF grid for O(1)
    lookups instead of per-point O3D SDF queries.
    """

    def __init__(self, mesh: o3d.geometry.TriangleMesh, target_points: np.ndarray,
                 normals: np.ndarray, frustum_far: float,
                 collision_radius: float = 0.0,
                 occupancy_grid=None,
                 free_space_resolution: float = 0.5,
                 use_gpu: bool = True):
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

        # Watertight warning
        if (collision_radius > 0 or occupancy_grid is not None) and not mesh.is_watertight():
            logger.warning(
                "[ViewpointSampler] Mesh is not watertight — SDF signs may be "
                "unreliable. Free-space sampling results could be incorrect."
            )

        # ── GPU setup ─────────────────────────────────────────────────────
        self._use_gpu = use_gpu and _CUPY_AVAILABLE and occupancy_grid is not None
        self._sdf_grid = None          # CPU float32 array
        self._sdf_grid_gpu = None      # CuPy array
        self._og_grid_gpu = None       # CuPy bool array (inflated)
        self._og_origin_gpu = None     # CuPy float32 (3,)
        self._target_points_gpu = None # CuPy float32 (M, 3)

        if self._use_gpu:
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

    # ── GPU helper methods ────────────────────────────────────────────

    def _sdf_lookup_gpu(self, positions_gpu):
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

    def _is_free_gpu(self, positions_gpu):
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

    def _gpu_nearest_neighbor(self, query_gpu, targets_gpu):
        """Brute-force nearest neighbor on GPU using squared-distance decomposition.

        Returns (distances, indices) for each query point.
        Chunks queries to limit memory usage.
        """
        n_query = len(query_gpu)
        n_target = len(targets_gpu)

        # Pre-compute ||b||² for all targets
        targets_sq = cp.sum(targets_gpu ** 2, axis=1)  # (M,)

        chunk_size = max(1, min(n_query, 500))
        indices = cp.empty(n_query, dtype=cp.int32)
        distances = cp.empty(n_query, dtype=cp.float32)

        for start in range(0, n_query, chunk_size):
            end = min(start + chunk_size, n_query)
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

    @staticmethod
    def _apply_angular_noise_gpu(directions_gpu, max_angle_rad):
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
            alt_norm = cp.maximum(alt_norm, 1e-12)
            perp[degenerate] = alt / alt_norm
            perp_norm[degenerate] = 1.0
        perp = perp / cp.maximum(perp_norm, 1e-12)

        cos_a = cp.cos(angles)[:, cp.newaxis]
        sin_a = cp.sin(angles)[:, cp.newaxis]
        cross = cp.cross(perp, directions_gpu)
        dot_kv = cp.sum(perp * directions_gpu, axis=1, keepdims=True)
        rotated = directions_gpu * cos_a + cross * sin_a + perp * dot_kv * (1 - cos_a)

        rotated_norm = cp.linalg.norm(rotated, axis=1, keepdims=True)
        rotated = rotated / cp.maximum(rotated_norm, 1e-12)
        return rotated

    # ── GPU free-space pipeline ───────────────────────────────────────

    def _build_free_space_gpu(self, side: str, min_dist: float, max_dist: float):
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
        sdf = self._sdf_lookup_gpu(centers_gpu)

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

    def _sample_from_free_space_gpu(self, centers_gpu, weights_gpu,
                                     coarse_res: float, num_candidates: int,
                                     side: str, min_dist: float,
                                     max_dir_noise_rad: float = np.deg2rad(15.0),
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

        # 3. Re-validate: OG collision check + SDF distance
        og_free = self._is_free_gpu(sampled_gpu)
        sdf = self._sdf_lookup_gpu(sampled_gpu)
        if side == "outside":
            sdf_valid = sdf >= min_dist
        else:
            sdf_valid = sdf <= -min_dist

        valid = og_free & sdf_valid
        sampled_gpu = sampled_gpu[valid]
        n_valid = int(len(sampled_gpu))
        logger.info("[ViewpointSampler] After jitter re-validation: %d / %d",
                    n_valid, num_candidates)

        if n_valid == 0:
            return []

        # 4. Nearest-neighbor for viewing direction (GPU brute-force)
        _, nearest_idx = self._gpu_nearest_neighbor(sampled_gpu, self._target_points_gpu)
        nearest_pts = self._target_points_gpu[nearest_idx]
        base_dirs = nearest_pts - sampled_gpu
        norms = cp.linalg.norm(base_dirs, axis=1, keepdims=True)
        norms = cp.maximum(norms, 1e-12)
        base_dirs = base_dirs / norms

        # 5. Angular noise via Rodrigues rotation (GPU)
        directions_gpu = self._apply_angular_noise_gpu(base_dirs, max_dir_noise_rad)

        # 6. Transfer back to CPU
        sampled_cpu = cp.asnumpy(sampled_gpu)
        directions_cpu = cp.asnumpy(directions_gpu)

        candidates = list(zip(sampled_cpu, directions_cpu))
        logger.info("[ViewpointSampler] Generated %d candidates from free space (GPU).",
                    len(candidates))
        return candidates

    # ── Private helpers (CPU) ─────────────────────────────────────────

    def _compute_signed_distances(self, positions: np.ndarray) -> np.ndarray:
        """Batch SDF query via Open3D RaycastingScene."""
        pts_tensor = o3d.core.Tensor(positions.astype(np.float32),
                                     dtype=o3d.core.Dtype.Float32)
        sdf = self.scene.compute_signed_distance(pts_tensor)
        return sdf.numpy()

    # ── OG-based free-space pipeline (CPU) ────────────────────────────

    def _build_free_space(self, side: str, min_dist: float,
                          max_dist: float):
        """
        Build feasible positions using the occupancy grid + SDF.

        Returns (feasible_centers, weights, coarse_resolution).
        """
        from shared.occupancy_grid import downsample_occupancy_grid, OccupancyGrid

        og = self._occupancy_grid
        coarse_res = self.free_space_resolution

        # 1. Downsample OG to coarse resolution
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
            return np.empty((0, 3)), np.empty(0), actual_res
        free_centers = coarse_og.voxel_to_world(free_ijk)
        logger.info("[ViewpointSampler] Coarse grid: %s, %d free voxels",
                    coarse_grid.shape, len(free_ijk))

        # 3. Compute SDF at free voxel centers
        sdf = self._compute_signed_distances(free_centers)

        # 4. Filter by side AND distance range
        if side == "outside":
            mask = (sdf >= min_dist) & (sdf <= max_dist)
        else:  # inside
            mask = (sdf <= -min_dist) & (sdf >= -max_dist)

        feasible = free_centers[mask]
        feasible_sdf = sdf[mask]
        logger.info("[ViewpointSampler] After SDF filter (%s): %d / %d positions",
                    side, len(feasible), len(free_centers))

        if len(feasible) == 0:
            return feasible, np.empty(0), actual_res

        # 5. Compute sampling weights: w = sdf² (footprint area ∝ d²)
        weights = feasible_sdf ** 2
        weights = weights / weights.sum()

        logger.info("[ViewpointSampler] %s free space: %d feasible positions "
                    "(coarse res %.2fm)", side.capitalize(), len(feasible), actual_res)

        return feasible, weights, actual_res

    def _sample_from_free_space(self, centers: np.ndarray, weights: np.ndarray,
                                coarse_res: float, num_candidates: int,
                                side: str, min_dist: float,
                                max_dir_noise_rad: float = np.deg2rad(15.0),
                                ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Sample viewpoints from precomputed feasible positions with sub-voxel
        jitter and angular noise on the viewing direction.
        """
        if len(centers) == 0:
            logger.warning("[ViewpointSampler] No feasible positions — returning empty.")
            return []

        # 1. Weighted random sample (probability ∝ distance²)
        indices = np.random.choice(len(centers), num_candidates,
                                   replace=True, p=weights)
        sampled = centers[indices].copy()

        # 2. Sub-voxel jitter → continuous positions
        jitter = np.random.uniform(-coarse_res / 2, coarse_res / 2,
                                   size=sampled.shape)
        sampled += jitter

        # 3. Re-validate jittered positions: OG collision check
        og = self._occupancy_grid
        og_free = og.is_free_world_batch(sampled)

        # Also re-validate with SDF for distance constraint
        sdf = self._compute_signed_distances(sampled)
        if side == "outside":
            sdf_valid = sdf >= min_dist
        else:
            sdf_valid = sdf <= -min_dist

        valid = og_free & sdf_valid
        sampled = sampled[valid]
        logger.info("[ViewpointSampler] After jitter re-validation: %d / %d",
                    len(sampled), num_candidates)

        if len(sampled) == 0:
            return []

        # 4. Compute viewing direction with angular noise
        _, nearest_idx = self._kdtree.query(sampled)
        nearest_pts = self.target_points[nearest_idx]
        base_dirs = nearest_pts - sampled
        norms = np.linalg.norm(base_dirs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        base_dirs = base_dirs / norms

        # Apply angular noise via Rodrigues rotation
        directions = self._apply_angular_noise(base_dirs, max_dir_noise_rad)

        candidates = list(zip(sampled, directions))
        logger.info("[ViewpointSampler] Generated %d candidates from free space.",
                    len(candidates))
        return candidates

    @staticmethod
    def _apply_angular_noise(directions: np.ndarray,
                             max_angle_rad: float) -> np.ndarray:
        """Rotate each direction by a random angle up to max_angle_rad
        around a random perpendicular axis (vectorized Rodrigues formula)."""
        n = len(directions)
        if max_angle_rad < 1e-8 or n == 0:
            return directions.copy()

        # Random rotation angles uniformly in [0, max_angle_rad]
        angles = np.random.uniform(0, max_angle_rad, size=n)

        # Random perpendicular axes: generate random vectors, subtract
        # projection onto direction, then normalise
        rand_vec = np.random.randn(n, 3)
        # Remove component along direction
        dot = np.sum(rand_vec * directions, axis=1, keepdims=True)
        perp = rand_vec - dot * directions
        perp_norm = np.linalg.norm(perp, axis=1, keepdims=True)
        # Handle degenerate cases (random vector nearly parallel to direction)
        degenerate = (perp_norm < 1e-8).ravel()
        if np.any(degenerate):
            alt = np.zeros_like(directions[degenerate])
            alt[:, 0] = -directions[degenerate, 1]
            alt[:, 1] = directions[degenerate, 0]
            alt_norm = np.linalg.norm(alt, axis=1, keepdims=True)
            alt_norm = np.maximum(alt_norm, 1e-12)
            perp[degenerate] = alt / alt_norm
            perp_norm[degenerate] = 1.0
        perp = perp / np.maximum(perp_norm, 1e-12)

        # Rodrigues: v_rot = v*cos(θ) + (k×v)*sin(θ) + k*(k·v)*(1-cos(θ))
        cos_a = np.cos(angles)[:, np.newaxis]
        sin_a = np.sin(angles)[:, np.newaxis]
        cross = np.cross(perp, directions)
        dot_kv = np.sum(perp * directions, axis=1, keepdims=True)
        rotated = directions * cos_a + cross * sin_a + perp * dot_kv * (1 - cos_a)

        # Re-normalise
        rotated_norm = np.linalg.norm(rotated, axis=1, keepdims=True)
        rotated = rotated / np.maximum(rotated_norm, 1e-12)
        return rotated

    # ── SDF-based free-space pipeline (fallback, no OG) ──────────────────

    def _build_free_space_sdf(self, side: str, min_dist: float,
                              max_dist: float) -> Tuple[np.ndarray, float]:
        """
        Build a regular 3D grid, compute SDF, and filter to feasible
        positions.  Used when no occupancy grid is provided.

        Returns (feasible_positions, resolution).
        """
        MAX_GRID_POINTS = 8_000_000  # ~200^3
        res = self.free_space_resolution

        if side == "outside":
            grid_min = self.mesh_bbox_min - max_dist
            grid_max = self.mesh_bbox_max + max_dist
        else:
            grid_min = self.mesh_bbox_min
            grid_max = self.mesh_bbox_max

        extent = grid_max - grid_min
        nx = int(np.ceil(extent[0] / res)) + 1
        ny = int(np.ceil(extent[1] / res)) + 1
        nz = int(np.ceil(extent[2] / res)) + 1
        total = nx * ny * nz

        if total > MAX_GRID_POINTS:
            volume = float(np.prod(extent))
            res = (volume / MAX_GRID_POINTS) ** (1.0 / 3.0)
            logger.info("[ViewpointSampler] Grid too large (%d pts) — coarsened to res=%.3f",
                        total, res)

        xs = np.arange(grid_min[0], grid_max[0] + res, res)
        ys = np.arange(grid_min[1], grid_max[1] + res, res)
        zs = np.arange(grid_min[2], grid_max[2] + res, res)
        total_points = len(xs) * len(ys) * len(zs)
        logger.info("[ViewpointSampler] Grid: %d x %d x %d = %d points (res=%.2f m)",
                    len(xs), len(ys), len(zs), total_points, res)

        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing='ij')
        grid_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

        sdf = self._compute_signed_distances(grid_points)

        if side == "outside":
            mask = (sdf >= min_dist) & (sdf <= max_dist)
        else:
            mask = (sdf <= -min_dist) & (sdf >= -max_dist)

        feasible = grid_points[mask]
        logger.info("[ViewpointSampler] After SDF filter (%s): %d / %d points",
                    side, len(feasible), len(grid_points))

        return feasible, res

    def _sample_from_free_space_sdf(self, feasible: np.ndarray, resolution: float,
                                    num_candidates: int, side: str,
                                    min_dist: float) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Sample viewpoints from the precomputed SDF free-space grid with
        sub-voxel jitter (legacy fallback).
        """
        if len(feasible) == 0:
            logger.warning("[ViewpointSampler] No feasible positions — returning empty.")
            return []

        replace = num_candidates > len(feasible)
        indices = np.random.choice(len(feasible), num_candidates, replace=replace)
        sampled = feasible[indices].copy()

        jitter = np.random.uniform(-resolution / 2, resolution / 2, size=sampled.shape)
        sampled += jitter

        sdf = self._compute_signed_distances(sampled)
        if side == "outside":
            valid = sdf >= min_dist
        else:
            valid = sdf <= -min_dist
        sampled = sampled[valid]
        logger.info("[ViewpointSampler] After jitter re-validation: %d / %d",
                    len(sampled), num_candidates)

        if len(sampled) == 0:
            return []

        _, nearest_idx = self._kdtree.query(sampled)
        nearest_pts = self.target_points[nearest_idx]
        directions = nearest_pts - sampled
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        directions = directions / norms

        candidates = list(zip(sampled, directions))
        logger.info("[ViewpointSampler] Generated %d candidates from free space.",
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

        if self._occupancy_grid is not None:
            # OG path
            if self._use_gpu:
                if cached is None:
                    cached = self._build_free_space_gpu(side, min_d, max_d)
                    setattr(self, cache_attr, cached)
                centers_gpu, weights_gpu, _ = cached
                return cp.asnumpy(centers_gpu), cp.asnumpy(weights_gpu)
            else:
                if cached is None:
                    cached = self._build_free_space(side, min_d, max_d)
                    setattr(self, cache_attr, cached)
                centers, weights, _ = cached
                return centers, weights
        else:
            # SDF path
            if cached is None:
                cached = self._build_free_space_sdf(side, min_d, max_d)
                setattr(self, cache_attr, cached)
            feasible, _ = cached
            if len(feasible) == 0:
                return feasible, np.empty(0)
            sdf = self._compute_signed_distances(feasible)
            w = sdf ** 2
            w_sum = w.sum()
            if w_sum > 0:
                w = w / w_sum
            return feasible, w

    # ── Public: dispatchers ──────────────────────────────────────────────

    def sample_outside_mesh(self, num_candidates: int,
                            min_distance: float | None = None,
                            max_distance_offset: float = 0.95,
                            ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Sample candidate viewpoints outside the mesh.

        When an occupancy grid is provided, uses OG-based free-space sampling.
        Otherwise uses SDF-based sampling.
        """
        min_d = min_distance if min_distance is not None else self.min_clearance
        max_d = max_distance_offset * self.frustum_far

        if self._occupancy_grid is not None:
            return self._sample_og_free_space(num_candidates, min_d, max_d, "outside")
        else:
            return self._sample_sdf_free_space(num_candidates, min_d, max_d, "outside")

    def sample_inside_mesh(self, num_candidates: int,
                           min_distance: float | None = None,
                           max_distance_offset: float = 0.95,
                           ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Sample candidate viewpoints inside the mesh.

        When an occupancy grid is provided, uses OG-based free-space sampling.
        Otherwise uses SDF-based sampling.
        """
        min_d = min_distance if min_distance is not None else self.min_clearance
        max_d = max_distance_offset * self.frustum_far

        if self._occupancy_grid is not None:
            return self._sample_og_free_space(num_candidates, min_d, max_d, "inside")
        else:
            return self._sample_sdf_free_space(num_candidates, min_d, max_d, "inside")

    # ── OG-based dispatch ────────────────────────────────────────────────

    def _sample_og_free_space(self, num_candidates: int,
                              min_distance: float, max_distance: float,
                              side: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        """OG-based free-space sampling (works for both inside and outside).

        Dispatches to GPU or CPU path depending on ``self._use_gpu``.
        """
        if self.num_points == 0:
            return []

        logger.info("[ViewpointSampler] Sampling %d viewpoints from %s mesh "
                    "(OG free-space%s, clearance=%.2f m, range=%.2f–%.2f m) …",
                    num_candidates, side.upper(),
                    " GPU" if self._use_gpu else "",
                    min_distance, min_distance, max_distance)

        cache_attr = f"_feasible_{side}"
        cached = getattr(self, cache_attr)

        if self._use_gpu:
            if cached is None:
                t0 = time.perf_counter()
                cached = self._build_free_space_gpu(side, min_distance, max_distance)
                dt = time.perf_counter() - t0
                logger.info("[ViewpointSampler] GPU _build_free_space: %.3fs", dt)
                setattr(self, cache_attr, cached)

            centers_gpu, weights_gpu, coarse_res = cached

            logger.info("[ViewpointSampler] %s free space: %d feasible positions",
                        side.capitalize(), int(len(centers_gpu)))

            t0 = time.perf_counter()
            result = self._sample_from_free_space_gpu(
                centers_gpu, weights_gpu, coarse_res, num_candidates,
                side, min_distance)
            dt = time.perf_counter() - t0
            logger.info("[ViewpointSampler] GPU _sample_from_free_space: %.3fs", dt)
            return result
        else:
            if cached is None:
                cached = self._build_free_space(side, min_distance, max_distance)
                setattr(self, cache_attr, cached)

            centers, weights, coarse_res = cached

            logger.info("[ViewpointSampler] %s free space: %d feasible positions",
                        side.capitalize(), len(centers))

            return self._sample_from_free_space(
                centers, weights, coarse_res, num_candidates, side, min_distance)

    # ── SDF-based dispatch (fallback) ────────────────────────────────────

    def _sample_sdf_free_space(self, num_candidates: int,
                               min_distance: float, max_distance: float,
                               side: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        """SDF-based free-space sampling (legacy fallback, no OG)."""
        if self.num_points == 0:
            return []

        logger.info("[ViewpointSampler] Sampling %d viewpoints from %s mesh "
                    "(SDF free-space, clearance=%.2f m, range=%.2f–%.2f m) …",
                    num_candidates, side.upper(), min_distance,
                    min_distance, max_distance)

        cache_attr = f"_feasible_{side}"
        cached = getattr(self, cache_attr)
        if cached is None:
            cached = self._build_free_space_sdf(side, min_distance, max_distance)
            setattr(self, cache_attr, cached)

        feasible, res = cached

        logger.info("[ViewpointSampler] %s free space: %d feasible positions",
                    side.capitalize(), len(feasible))

        return self._sample_from_free_space_sdf(
            feasible, res, num_candidates, side, min_distance)

