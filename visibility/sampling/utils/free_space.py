"""Free-space construction and viewpoint sampling pipeline."""

import logging
from typing import List, Tuple

import cupy as cp
import numpy as np

from ...core.constants import (
    NORM_EPS, DEFAULT_MAX_DIR_NOISE_RAD, CURVATURE_POSITION_WEIGHT,
)
from .direction import knn_centroid_direction, apply_angular_noise

logger = logging.getLogger(__name__)


def build_free_space(occupancy_grid, sdf_grid_gpu, og_origin_gpu, og_resolution,
                     target_points_gpu, normals, free_space_resolution,
                     side: str, min_dist: float, max_dist: float,
                     curvature_weighting: bool = False,
                     sdf_lookup_fn=None):
    """Build feasible positions using EDT-SDF grid lookup on GPU.

    Args:
        occupancy_grid:       OccupancyGrid object.
        sdf_grid_gpu:         CuPy array — precomputed SDF grid.
        og_origin_gpu:        CuPy float32 (3,) — grid origin.
        og_resolution:        float — fine grid resolution.
        target_points_gpu:    CuPy float32 (M, 3) — surface points.
        normals:              numpy (M, 3) — surface normals.
        free_space_resolution: float — coarse grid resolution.
        side:                 "outside" or "inside".
        min_dist:             minimum SDF distance.
        max_dist:             maximum SDF distance.
        curvature_weighting:  bias weights toward high-curvature regions.
        sdf_lookup_fn:        callable(positions_gpu) → CuPy SDF values.

    Returns:
        (feasible_centers_gpu, weights_gpu, coarse_resolution)
    """
    from shared.occupancy_grid import downsample_occupancy_grid, OccupancyGrid as OG

    og = occupancy_grid
    coarse_res = free_space_resolution

    # 1. Downsample OG to coarse resolution (CPU — fast)
    coarse_grid, coarse_origin, actual_res = downsample_occupancy_grid(
        og.grid, og.origin, og.resolution, coarse_res
    )
    coarse_og = OG(grid=coarse_grid, origin=coarse_origin, resolution=actual_res)

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
    sdf = sdf_lookup_fn(centers_gpu)

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

    if curvature_weighting:
        from .curvature import compute_local_curvature
        local_curv = compute_local_curvature(
            feasible_gpu, target_points_gpu,
            cp.asarray(normals, dtype=cp.float32))
        curv_norm = local_curv / (local_curv.max() + NORM_EPS)
        weights *= (1.0 + CURVATURE_POSITION_WEIGHT * curv_norm)

    weights = weights / weights.sum()

    logger.info("[ViewpointSampler] %s free space: %d feasible positions "
                "(coarse res %.2fm, curvature_weighting=%s)",
                side.capitalize(), n_feasible, actual_res, curvature_weighting)

    return feasible_gpu, weights, actual_res


def sample_from_free_space(centers_gpu, weights_gpu,
                           coarse_res: float, num_candidates: int,
                           target_points_gpu, normals,
                           max_dir_noise_rad: float = DEFAULT_MAX_DIR_NOISE_RAD,
                           direction_targets_gpu=None,
                           curvature_weighting: bool = False,
                           ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """GPU-accelerated viewpoint sampling from feasible positions.

    Args:
        centers_gpu:          (N, 3) CuPy array — feasible positions.
        weights_gpu:          (N,) CuPy array — normalized sampling weights.
        coarse_res:           coarse grid resolution (for jitter).
        num_candidates:       number of candidates to generate.
        target_points_gpu:    (M, 3) CuPy array — surface points.
        normals:              numpy (M, 3) — surface normals.
        max_dir_noise_rad:    angular noise for viewing directions.
        direction_targets_gpu: optional (U, 3) CuPy — override direction targets.
        curvature_weighting:  use curvature-weighted KNN direction.

    Returns:
        List of (position, quaternion) tuples.
    """
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

    # 5. Transfer back to CPU and convert directions to quaternions
    sampled_cpu = cp.asnumpy(sampled_gpu)
    directions_cpu = cp.asnumpy(directions_gpu)

    from shared.geometry import direction_roll_to_quaternion
    orientations = np.array([direction_roll_to_quaternion(d) for d in directions_cpu])

    candidates = list(zip(sampled_cpu, orientations))
    logger.info("[ViewpointSampler] Generated %d candidates from free space (GPU).",
                len(candidates))
    return candidates
