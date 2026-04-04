"""Sampling space construction from occupancy grid and SDF."""

import logging
from typing import Tuple

import cupy as cp

from shared.grid_utils import downsample_occupancy_grid
from shared.occupancy_grid import OccupancyGrid

from ...core.constants import NORM_EPS, CURVATURE_POSITION_WEIGHT
from ...core.types import Side

logger = logging.getLogger(__name__)


def build_sampling_space(occupancy_grid: OccupancyGrid, sdf_grid_gpu: cp.ndarray,
                         target_points_gpu: cp.ndarray, normals_gpu: cp.ndarray, free_space_resolution: float,
                         side: Side, min_dist: float, max_dist: float,
                         curvature_weighting: bool = False) -> Tuple[cp.ndarray, cp.ndarray, float]:
    """Build feasible sampling positions using EDT-SDF grid lookup on GPU.

    Downsamples the occupancy grid to *free_space_resolution*, filters by
    signed distance, and returns weighted positions ready for sampling.

    Args:
        occupancy_grid:       OccupancyGrid object (GPU-resident).
        sdf_grid_gpu:         CuPy array — precomputed SDF grid (fine res).
        target_points_gpu:    CuPy float32 (M, 3) — surface points.
        normals_gpu:          CuPy float32 (M, 3) — surface normals.
        free_space_resolution: float — coarse grid resolution.
        side:                 Side.OUTSIDE or Side.INSIDE.
        min_dist:             minimum SDF distance.
        max_dist:             maximum SDF distance.
        curvature_weighting:  bias weights toward high-curvature regions.

    Returns:
        (feasible_centers_gpu, weights_gpu, coarse_resolution)
    """
    og = occupancy_grid
    coarse_res = free_space_resolution

    # 1. Downsample OG to coarse resolution
    coarse_grid, coarse_origin, actual_res = downsample_occupancy_grid(
        og.grid, og.origin, og.resolution, coarse_res
    )

    # 2. Get free voxel indices
    free_ijk = cp.argwhere(~coarse_grid)
    if len(free_ijk) == 0:
        logger.warning("[build_sampling_space] No free voxels in coarse grid.")
        return cp.empty((0, 3), dtype=cp.float32), cp.empty(0, dtype=cp.float32), actual_res

    logger.info("[build_sampling_space] Coarse grid: %s, %d free voxels",
                coarse_grid.shape, len(free_ijk))

    # 3. Map coarse voxel indices -> fine SDF-grid indices (center of each coarse voxel)
    factor = round(actual_res / og.resolution)
    fine_ijk = free_ijk * factor + factor // 2

    # Bounds-check against SDF grid
    sdf_shape = cp.asarray(sdf_grid_gpu.shape, dtype=cp.int32)
    in_bounds = cp.all((fine_ijk >= 0) & (fine_ijk < sdf_shape), axis=1)
    free_ijk = free_ijk[in_bounds]
    fine_ijk = fine_ijk[in_bounds]

    if len(fine_ijk) == 0:
        logger.warning("[build_sampling_space] No in-bounds fine voxels after mapping.")
        return cp.empty((0, 3), dtype=cp.float32), cp.empty(0, dtype=cp.float32), actual_res

    # 4. Look up SDF via fine grid indices
    sdf = sdf_grid_gpu[fine_ijk[:, 0], fine_ijk[:, 1], fine_ijk[:, 2]]

    # 5. Filter by side AND distance range
    match side:
        case Side.OUTSIDE:
            mask = (sdf >= min_dist) & (sdf <= max_dist)
        case Side.INSIDE:
            mask = (sdf <= -min_dist) & (sdf >= -max_dist)

    feasible_ijk = free_ijk[mask]
    feasible_sdf = sdf[mask]
    n_feasible = len(feasible_ijk)
    logger.info("[build_sampling_space] After SDF filter (%s): %d / %d positions",
                side.value, n_feasible, len(free_ijk))

    if n_feasible == 0:
        return cp.empty((0, 3), dtype=cp.float32), cp.empty(0, dtype=cp.float32), actual_res

    # 6. Convert feasible coarse indices to world coordinates
    feasible_centers = (feasible_ijk.astype(cp.float32) * actual_res
                        + coarse_origin.astype(cp.float32) + 0.5 * actual_res)

    # 7. Compute sampling weights: w = sdf^2 (footprint area ~ d^2)
    weights = feasible_sdf ** 2

    if curvature_weighting:
        from .curvature import compute_local_curvature
        local_curv = compute_local_curvature(
            feasible_centers, target_points_gpu, normals_gpu)
        curv_norm = local_curv / (local_curv.max() + NORM_EPS)
        weights *= (1.0 + CURVATURE_POSITION_WEIGHT * curv_norm)

    weights = weights / weights.sum()

    logger.info("[build_sampling_space] %s free space: %d feasible positions "
                "(coarse res %.2fm, curvature_weighting=%s)",
                side.value.capitalize(), n_feasible, actual_res, curvature_weighting)

    return feasible_centers, weights, actual_res
