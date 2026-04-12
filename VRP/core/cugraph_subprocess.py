"""DEPRECATED: cuGraph is now called directly in-process in distance_matrix.py.

This file is no longer used and can be safely deleted.

---

Original docstring:

cuGraph all-pairs Dijkstra subprocess.

Runs inside the ``rapids_solver`` conda environment. Reads a JSON config
from argv[1], builds a cuGraph graph from the occupancy grid's free voxels,
runs all-pairs Dijkstra, and writes the resulting N x N distance matrix.

Graph construction is GPU-vectorized via CuPy: for each of the 26
neighbor offsets, all free voxels are shifted in parallel and valid edges
collected without Python-level loops over voxels.

References:
    Davidson, A. et al. (2014). Work-Efficient Parallel GPU Methods for
        Single-Source Shortest Paths. IPDPS.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from VRP.core.constants import OFFSETS_26, WEIGHTS_26

logger = logging.getLogger(__name__)


def cugraph_distance_matrix_main():
    """Entry point when this script is executed as a subprocess."""
    config_path = sys.argv[1]
    with open(config_path) as f:
        cfg = json.load(f)

    grid = np.load(cfg["grid_path"])
    origin = np.array(cfg["origin"], dtype=np.float64)
    resolution = float(cfg["resolution"])
    wp_flat = np.array(cfg["waypoints_flat"], dtype=np.float64)
    N = len(wp_flat) // 3
    waypoints = wp_flat.reshape(N, 3)
    out_path = cfg["out_matrix_path"]

    Nx, Ny, Nz = grid.shape
    logger.info("[cuGraph] Grid shape: %s, %s waypoints", grid.shape, N)

    import cudf
    import cugraph
    import cupy as cp

    # ── GPU-vectorized graph construction ──────────────────────────────
    grid_gpu = cp.asarray(grid)
    free_ijk_gpu = cp.argwhere(~grid_gpu)  # (F, 3) int64
    F = len(free_ijk_gpu)
    logger.info("[cuGraph] Free voxels: %s", F)

    # Flat-index -> node-ID lookup table (GPU)
    free_flat_gpu = (
        free_ijk_gpu[:, 0].astype(cp.int64) * (Ny * Nz)
        + free_ijk_gpu[:, 1].astype(cp.int64) * Nz
        + free_ijk_gpu[:, 2].astype(cp.int64)
    )
    flat_lookup = cp.full(Nx * Ny * Nz, -1, dtype=cp.int32)
    flat_lookup[free_flat_gpu] = cp.arange(F, dtype=cp.int32)

    weights = WEIGHTS_26 * float(resolution)

    src_all, dst_all, wt_all = [], [], []
    for oi in range(26):
        nbr = free_ijk_gpu + OFFSETS_26[oi]  # (F, 3) broadcast
        valid = (
            (nbr[:, 0] >= 0) & (nbr[:, 0] < Nx) &
            (nbr[:, 1] >= 0) & (nbr[:, 1] < Ny) &
            (nbr[:, 2] >= 0) & (nbr[:, 2] < Nz)
        )
        nbr_v = nbr[valid]
        # Check neighbors are free
        free_nbr = ~grid_gpu[nbr_v[:, 0], nbr_v[:, 1], nbr_v[:, 2]]
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

    # Build cuDF DataFrame directly from CuPy arrays
    gdf = cudf.DataFrame({
        "src": cudf.core.column.as_column(src_arr),
        "dst": cudf.core.column.as_column(dst_arr),
        "weight": cudf.core.column.as_column(wt_arr),
    })
    G = cugraph.Graph()
    G.from_cudf_edgelist(gdf, source="src", destination="dst", edge_attr="weight")
    logger.info("[cuGraph] Graph: %s nodes, %s edges",
                G.number_of_nodes(), G.number_of_edges())

    # ── Waypoint node IDs (vectorized) ────────────────────────────────
    wp_gpu = cp.asarray(waypoints)
    origin_gpu = cp.asarray(origin)
    wp_ijk_gpu = cp.floor((wp_gpu - origin_gpu) / resolution).astype(cp.int64)
    wp_flat_gpu = (
        wp_ijk_gpu[:, 0] * (Ny * Nz)
        + wp_ijk_gpu[:, 1] * Nz
        + wp_ijk_gpu[:, 2]
    )
    wp_node_ids_gpu = flat_lookup[wp_flat_gpu]  # (N,) int32, GPU

    # ── SSSP for each waypoint, extract via direct array indexing ─────
    matrix = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        df = cugraph.shortest_path(G, int(wp_node_ids_gpu[i]))
        dists = df.sort_values("vertex")["distance"].values  # (F,) cupy
        matrix[i, :] = cp.asnumpy(dists[wp_node_ids_gpu])
        matrix[i, i] = 0.0
        if (i + 1) % max(1, N // 5) == 0 or i == N - 1:
            logger.info("[cuGraph] Dijkstra %s/%s done", i + 1, N)

    np.save(out_path, matrix)
    logger.info("[cuGraph] Matrix saved to %s", out_path)


def rapids_env_has_cugraph(rapids_python: str) -> bool:
    """Quick check whether the RAPIDS env has cugraph importable."""
    try:
        result = subprocess.run(
            [rapids_python, "-c", "import cugraph; print('ok')"],
            capture_output=True, text=True, timeout=15
        )
        return result.returncode == 0 and "ok" in result.stdout
    except Exception:
        return False


def compute_via_subprocess(
    occupancy_grid,
    waypoints_world: np.ndarray,
    rapids_python: str,
) -> np.ndarray:
    """Call this script as a subprocess inside the RAPIDS env."""
    with tempfile.TemporaryDirectory() as tmpdir:
        grid_path = os.path.join(tmpdir, "grid.npy")
        config_path = os.path.join(tmpdir, "config.json")
        out_path = os.path.join(tmpdir, "matrix.npy")

        # serialize for subprocess
        import cupy as _cp
        grid_np = _cp.asnumpy(occupancy_grid.grid)
        np.save(grid_path, grid_np)
        cfg = {
            "grid_path": grid_path,
            "origin": (_cp.asnumpy(occupancy_grid.origin)).tolist(),
            "resolution": float(occupancy_grid.resolution),
            "waypoints_flat": waypoints_world.flatten().tolist(),
            "out_matrix_path": out_path,
        }
        with open(config_path, "w") as f:
            json.dump(cfg, f)

        script_path = os.path.abspath(__file__)
        logger.info("[DistMatrix] Launching cuGraph subprocess...")
        result = subprocess.run(
            [rapids_python, script_path, config_path],
            capture_output=False, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"cuGraph subprocess failed (exit {result.returncode})."
            )
        return np.load(out_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cugraph_subprocess.py <config.json>")
        sys.exit(1)
    cugraph_distance_matrix_main()
