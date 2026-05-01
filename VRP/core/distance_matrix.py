"""Collision-free distance matrix via cuGraph SSSP.

The cuGraph Dijkstra runs in a one-shot subprocess because in-process
calls leak RMM ``DeviceBuffer`` allocations (~7 GB) that no amount of
``gc.collect`` or ``rmm.reinitialize`` reclaims; tearing down the CUDA
context on subprocess exit is the only reliable release.
"""

from __future__ import annotations

import concurrent.futures
import logging
import multiprocessing as mp
import os

import cupy as cp
import numpy as np

from .constants import OFFSETS_26, WEIGHTS_26

logger = logging.getLogger(__name__)


def _build_cugraph_distance_matrix(
    occupancy_grid,
    waypoints_xyz: cp.ndarray,
) -> cp.ndarray:
    """Build a grid graph with CuPy and run N Dijkstras with cuGraph.

    Args:
        occupancy_grid: OccupancyGrid instance (GPU-resident).
        waypoints_xyz: (N, 3) CuPy world-frame waypoint positions.

    Returns:
        (N, N) float32 CuPy distance matrix in metres.
    """
    import gc as _gc

    import cudf
    import cugraph

    grid = occupancy_grid.grid  # (Nx, Ny, Nz) bool CuPy
    origin = occupancy_grid.origin  # (3,) CuPy
    resolution = float(occupancy_grid.resolution)

    Nx, Ny, Nz = grid.shape
    N = len(waypoints_xyz)
    logger.info("[DistMatrix] Grid shape: %s, %d waypoints", grid.shape, N)

    # ── graph construction ────────────────────────────
    free_ijk_gpu = cp.argwhere(~grid)  # (F, 3) int64
    F = len(free_ijk_gpu)
    logger.info("[DistMatrix] Free voxels: %d", F)

    # Flat-index -> node-ID lookup table (GPU)
    free_flat_gpu = (
        free_ijk_gpu[:, 0].astype(cp.int64) * (Ny * Nz)
        + free_ijk_gpu[:, 1].astype(cp.int64) * Nz
        + free_ijk_gpu[:, 2].astype(cp.int64)
    )
    flat_lookup = cp.full(Nx * Ny * Nz, -1, dtype=cp.int32)
    flat_lookup[free_flat_gpu] = cp.arange(F, dtype=cp.int32)

    weights = WEIGHTS_26 * resolution

    src_all, dst_all, wt_all = [], [], []
    for oi in range(26):
        nbr = free_ijk_gpu + OFFSETS_26[oi]  # (F, 3) broadcast
        valid = (
            (nbr[:, 0] >= 0)
            & (nbr[:, 0] < Nx)
            & (nbr[:, 1] >= 0)
            & (nbr[:, 1] < Ny)
            & (nbr[:, 2] >= 0)
            & (nbr[:, 2] < Nz)
        )
        nbr_v = nbr[valid]
        # Check neighbors are free
        free_nbr = ~grid[nbr_v[:, 0], nbr_v[:, 1], nbr_v[:, 2]]
        nbr_v = nbr_v[free_nbr]
        src_v = free_ijk_gpu[valid][free_nbr]
        if len(src_v) == 0:
            continue
        src_flat = (
            src_v[:, 0].astype(cp.int64) * (Ny * Nz)
            + src_v[:, 1].astype(cp.int64) * Nz
            + src_v[:, 2].astype(cp.int64)
        )
        dst_flat = (
            nbr_v[:, 0].astype(cp.int64) * (Ny * Nz)
            + nbr_v[:, 1].astype(cp.int64) * Nz
            + nbr_v[:, 2].astype(cp.int64)
        )
        src_nodes = flat_lookup[src_flat]
        dst_nodes = flat_lookup[dst_flat]
        edge_mask = (src_nodes >= 0) & (dst_nodes >= 0)
        src_all.append(src_nodes[edge_mask])
        dst_all.append(dst_nodes[edge_mask])
        wt_all.append(cp.full(int(edge_mask.sum()), weights[oi], dtype=cp.float32))

    src_arr = cp.concatenate(src_all)
    dst_arr = cp.concatenate(dst_all)
    wt_arr = cp.concatenate(wt_all)

    # Build cuGraph graph directly from CuPy arrays
    gdf = cudf.DataFrame(
        {
            "src": cudf.core.column.as_column(src_arr),
            "dst": cudf.core.column.as_column(dst_arr),
            "weight": cudf.core.column.as_column(wt_arr),
        }
    )
    G = cugraph.Graph()
    G.from_cudf_edgelist(gdf, source="src", destination="dst", edge_attr="weight")
    logger.info("[DistMatrix] Graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    # ── Waypoint node IDs  ──────────────────────────────
    wp_gpu = waypoints_xyz.astype(cp.float64)
    origin_gpu = cp.asarray(origin, dtype=cp.float64)
    wp_ijk_gpu = cp.floor((wp_gpu - origin_gpu) / resolution).astype(cp.int64)
    wp_flat_gpu = wp_ijk_gpu[:, 0] * (Ny * Nz) + wp_ijk_gpu[:, 1] * Nz + wp_ijk_gpu[:, 2]
    wp_node_ids_gpu = flat_lookup[wp_flat_gpu]  # (N,) int32, GPU

    # ── Validate waypoints are in free space ───────────────────────
    occupied_mask = wp_node_ids_gpu < 0
    if bool(cp.any(occupied_mask)):
        occ_indices = cp.asnumpy(cp.where(occupied_mask)[0]).tolist()
        occ_voxels = cp.asnumpy(wp_ijk_gpu[occupied_mask]).tolist()
        raise ValueError(
            f"[DistMatrix] {len(occ_indices)} waypoint(s) map to occupied or "
            f"out-of-bounds voxels (node_id=-1): indices {occ_indices}, "
            f"voxel coords {occ_voxels}. Ensure waypoints are sampled with a "
            f"clearance >= OG inflation radius."
        )

    # ── SSSP for each waypoint ──────────────────────────────────────
    matrix = cp.zeros((N, N), dtype=cp.float32)
    try:
        for i in range(N):
            df = cugraph.shortest_path(G, int(wp_node_ids_gpu[i]))
            dists = df.sort_values("vertex")["distance"].values  # (F,) cupy
            matrix[i, :] = dists[wp_node_ids_gpu]
            matrix[i, i] = 0.0
            del df, dists
            if (i + 1) % max(1, N // 5) == 0 or i == N - 1:
                logger.info("[DistMatrix] Dijkstra %d/%d done", i + 1, N)
    finally:
        # Release cuDF/cuGraph objects so their UCX worker threads finish
        # before any in-process CPU solver (e.g. HiGHS) runs.  Without this,
        # Python's GC may collect these objects mid-solve and crash on UCX
        # spinlock teardown (signal 11, "ucs_recursive_spinlock_destroy() failed: busy").
        # Using try/finally ensures cleanup even when the Dijkstra loop raises.
        del G, gdf
        _gc.collect()
        cp.cuda.Device().synchronize()

    return matrix


def _dijkstra_subprocess_worker(
    grid_np: np.ndarray,
    origin_np: np.ndarray,
    resolution: float,
    waypoints_np: np.ndarray,
) -> np.ndarray:
    """Subprocess entrypoint: rebuild grid on GPU, run cuGraph, return numpy.

    Must be a top-level function to be picklable by ProcessPoolExecutor.
    Imports are inside the function so the parent process doesn't pay for
    them (and so the subprocess gets a fresh cuGraph/cuDF initialisation).
    """
    import cupy as _cp

    from shared.occupancy_grid import OccupancyGrid

    og = OccupancyGrid(
        grid=_cp.asarray(grid_np),
        origin=_cp.asarray(origin_np),
        resolution=resolution,
    )
    wp_gpu = _cp.asarray(waypoints_np)
    matrix_gpu = _build_cugraph_distance_matrix(og, wp_gpu)
    return _cp.asnumpy(matrix_gpu)


def compute_distance_matrix(
    occupancy_grid,
    waypoints_xyz: cp.ndarray,
    cache_path: str | None = None,
    force_rebuild: bool = False,
) -> cp.ndarray:
    """Compute the N x N collision-free distance matrix via cuGraph.

    Runs ``_build_cugraph_distance_matrix`` in a fresh subprocess so the
    ~7 GB RMM pool consumed by cuGraph/cuDF is guaranteed to be released
    when the subprocess exits.  See module docstring for rationale.

    Args:
        occupancy_grid: OccupancyGrid instance (GPU-resident).
        waypoints_xyz: (N, 3) CuPy world-frame waypoint positions.
        cache_path: optional path to cache the result as .npy.
        force_rebuild: ignore cache.

    Returns:
        (N, N) float32 CuPy distance matrix in metres.
    """
    if cache_path and not force_rebuild and os.path.exists(cache_path):
        logger.info("[DistMatrix] Loading cached matrix from %s", cache_path)
        return cp.asarray(np.load(cache_path))

    N = len(waypoints_xyz)
    logger.info("[DistMatrix] Computing %dx%d distance matrix via cuGraph (subprocess)...", N, N)

    grid_np = cp.asnumpy(occupancy_grid.grid)
    origin_np = cp.asnumpy(occupancy_grid.origin)
    resolution = float(occupancy_grid.resolution)
    waypoints_np = cp.asnumpy(waypoints_xyz)

    ctx = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
        future = pool.submit(
            _dijkstra_subprocess_worker,
            grid_np,
            origin_np,
            resolution,
            waypoints_np,
        )
        matrix_np = future.result()

    matrix = cp.asarray(matrix_np)

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        np.save(cache_path, matrix_np)
        logger.info("[DistMatrix] Saved to %s", cache_path)

    return matrix
