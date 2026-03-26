"""Free-space construction and viewpoint sampling pipeline."""

import logging
from typing import List, Tuple

import cupy as cp
import numpy as np

from shared.grid_utils import downsample_occupancy_grid
from shared.occupancy_grid import OccupancyGrid

from ...core.constants import (
    NORM_EPS, DEFAULT_MAX_DIR_NOISE_RAD, CURVATURE_POSITION_WEIGHT,
)
from scipy.spatial.transform import Rotation

from .direction import knn_centroid_direction, apply_angular_noise
from shared.geometry import directions_rolls_to_rotmats

logger = logging.getLogger(__name__)


def build_sampling_space(occupancy_grid: OccupancyGrid, sdf_grid_gpu: cp.ndarray,
                         target_points_gpu: cp.ndarray, normals: np.ndarray, free_space_resolution: float,
                         side: str, min_dist: float, max_dist: float,
                         curvature_weighting: bool = False):
    """Build feasible sampling positions using EDT-SDF grid lookup on GPU.

    Downsamples the occupancy grid to *free_space_resolution*, filters by 
    signed distance, and returns weighted positions ready for sampling.

    Args:
        occupancy_grid:       OccupancyGrid object.
        sdf_grid_gpu:         CuPy array — precomputed SDF grid (fine res).
        target_points_gpu:    CuPy float32 (M, 3) — surface points.
        normals:              numpy (M, 3) — surface normals.
        free_space_resolution: float — coarse grid resolution.
        side:                 "outside" or "inside".
        min_dist:             minimum SDF distance.
        max_dist:             maximum SDF distance.
        curvature_weighting:  bias weights toward high-curvature regions.

    Returns:
        (feasible_centers_gpu, weights_gpu, coarse_resolution)
    """
    og = occupancy_grid
    coarse_res = free_space_resolution

    # 1. Downsample OG to coarse resolution (CPU — fast)
    coarse_grid, coarse_origin, actual_res = downsample_occupancy_grid(
        og.grid, og.origin, og.resolution, coarse_res
    )

    # 2. Get free voxel indices
    free_ijk = np.argwhere(~coarse_grid)
    if len(free_ijk) == 0:
        logger.warning("[build_sampling_space] No free voxels in coarse grid.")
        return cp.empty((0, 3), dtype=cp.float32), cp.empty(0, dtype=cp.float32), actual_res

    logger.info("[build_sampling_space] Coarse grid: %s, %d free voxels",
                coarse_grid.shape, len(free_ijk))

    # 3. Map coarse voxel indices -> fine SDF-grid indices (center of each coarse voxel)
    factor = round(actual_res / og.resolution)
    fine_ijk = free_ijk * factor + factor // 2

    # Bounds-check against SDF grid (padding in downsample can push edge voxels OOB)
    sdf_shape = np.array(sdf_grid_gpu.shape)
    in_bounds = np.all((fine_ijk >= 0) & (fine_ijk < sdf_shape), axis=1)
    free_ijk = free_ijk[in_bounds]
    fine_ijk = fine_ijk[in_bounds]

    if len(fine_ijk) == 0:
        logger.warning("[build_sampling_space] No in-bounds fine voxels after mapping.")
        return cp.empty((0, 3), dtype=cp.float32), cp.empty(0, dtype=cp.float32), actual_res

    # 4. Look up SDF via fine grid indices
    fine_ijk_gpu = cp.asarray(fine_ijk, dtype=cp.int32)
    sdf = sdf_grid_gpu[fine_ijk_gpu[:, 0], fine_ijk_gpu[:, 1], fine_ijk_gpu[:, 2]]

    # 5. Filter by side AND distance range
    if side == "outside":
        mask = (sdf >= min_dist) & (sdf <= max_dist)
    else:  # inside
        mask = (sdf <= -min_dist) & (sdf >= -max_dist)

    mask_cpu = cp.asnumpy(mask)
    feasible_ijk = free_ijk[mask_cpu]
    feasible_sdf = sdf[mask]
    n_feasible = len(feasible_ijk)
    logger.info("[build_sampling_space] After SDF filter (%s): %d / %d positions",
                side, n_feasible, len(free_ijk))

    if n_feasible == 0:
        return cp.empty((0, 3), dtype=cp.float32), cp.empty(0, dtype=cp.float32), actual_res

    # 6. Convert feasible coarse indices to world coordinates
    #    (inline of voxel_to_world: ijk * res + origin + 0.5 * res)
    feasible_centers = (feasible_ijk.astype(np.float32) * actual_res
                        + coarse_origin + 0.5 * actual_res)
    feasible_gpu = cp.asarray(feasible_centers, dtype=cp.float32)

    # 7. Compute sampling weights: w = sdf² (footprint area ∝ d²)
    weights = feasible_sdf ** 2

    if curvature_weighting:
        from .curvature import compute_local_curvature
        local_curv = compute_local_curvature(
            feasible_gpu, target_points_gpu,
            cp.asarray(normals, dtype=cp.float32))
        curv_norm = local_curv / (local_curv.max() + NORM_EPS)
        weights *= (1.0 + CURVATURE_POSITION_WEIGHT * curv_norm)

    weights = weights / weights.sum()

    logger.info("[build_sampling_space] %s free space: %d feasible positions "
                "(coarse res %.2fm, curvature_weighting=%s)",
                side.capitalize(), n_feasible, actual_res, curvature_weighting)

    return feasible_gpu, weights, actual_res


def sample_from_free_space(centers_gpu: cp.ndarray, weights_gpu: cp.ndarray,
                           coarse_res: float, num_candidates: int,
                           target_points_gpu: cp.ndarray, normals: np.ndarray,
                           max_dir_noise_rad: float = DEFAULT_MAX_DIR_NOISE_RAD,
                           direction_targets_gpu=None,
                           curvature_weighting: bool = False,
                           ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """GPU-accelerated viewpoint sampling from feasible positions.

    Args:
        centers_gpu:          (N, 3) CuPy array — feasible positions (centers of feasible voxels).
        weights_gpu:          (N,) CuPy array — normalized sampling weights.
        coarse_res:           coarse grid resolution (for jitter).
        num_candidates:       number of candidates to generate.
        target_points_gpu:    (M, 3) CuPy array — surface points.
        normals:              numpy (M, 3) — surface normals.
        max_dir_noise_rad:    maximum angular noise for viewing directions.
        direction_targets_gpu: optional (U, 3) CuPy — override direction targets (uncovered points).
        curvature_weighting:  use curvature-weighted KNN direction.

    Returns:
        List of (position, Rotation) tuples.
    """
    n_feasible = int(len(centers_gpu))
    if n_feasible == 0:
        logger.warning("[sample_from_free_space] No feasible positions — returning empty.")
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
    dir_targets = (direction_targets_gpu if direction_targets_gpu is not None
                   else target_points_gpu)
    normals_gpu = (cp.asarray(normals, dtype=cp.float32)
                   if curvature_weighting else None)
    base_dirs = knn_centroid_direction(
        sampled_gpu, dir_targets, normals_gpu=normals_gpu)
    norms = cp.linalg.norm(base_dirs, axis=1, keepdims=True)
    norms = cp.maximum(norms, NORM_EPS)
    base_dirs = base_dirs / norms

    # 4. Angular noise via Rodrigues rotation (GPU)
    directions_gpu = apply_angular_noise(base_dirs, max_dir_noise_rad)

    # 5. Random roll + GPU-vectorized rotation construction
    rolls_gpu = cp.random.uniform(0, 2 * cp.pi, size=num_candidates, dtype=cp.float32)
    rotmats_gpu = directions_rolls_to_rotmats(directions_gpu, rolls_gpu)

    sampled_cpu = cp.asnumpy(sampled_gpu)
    rotmats_cpu = cp.asnumpy(rotmats_gpu)
    orientations = Rotation.from_matrix(rotmats_cpu)  # batch — no loop

    candidates = list(zip(sampled_cpu, orientations))
    logger.info("[sample_from_free_space] Generated %d candidates from free space (GPU).",
                len(candidates))
    return candidates
