"""cuGraph all-pairs Dijkstra subprocess.

Runs inside the ``rapids_solver`` conda environment. Reads a JSON config
from argv[1], builds a cuGraph graph from the occupancy grid's free voxels,
runs all-pairs Dijkstra, and writes the resulting N x N distance matrix.

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

from shared.grid_utils import OFFSETS_26 as _OFFSETS_26, WEIGHTS_26 as _WEIGHTS_26

logger = logging.getLogger(__name__)


def cugraph_distance_matrix_main():
    """Entry point when this script is executed as a subprocess.

    Reads JSON config from argv[1] with keys: grid_path, origin,
    resolution, waypoints_flat, out_matrix_path.
    """
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

    logger.info("[cuGraph] Grid shape: %s, %s waypoints", grid.shape, N)

    import cudf
    import cugraph
    import cupy as cp

    free_mask = ~grid
    Nx, Ny, Nz = grid.shape

    free_ijk = np.argwhere(free_mask)
    free_flat = np.ravel_multi_index(
        (free_ijk[:, 0], free_ijk[:, 1], free_ijk[:, 2]), grid.shape
    )

    flat_to_node = {int(f): i for i, f in enumerate(free_flat)}
    F = len(free_flat)
    logger.info("[cuGraph] Free voxels: %s", F)

    offsets = np.array(_OFFSETS_26, dtype=np.int32)
    weights_arr = np.array(_WEIGHTS_26, dtype=np.float32) * float(resolution)

    src_list, dst_list, wt_list = [], [], []
    BATCH = 50_000
    for start in range(0, len(free_ijk), BATCH):
        batch_ijk = free_ijk[start: start + BATCH]
        for (di, dj, dk), w in zip(offsets, weights_arr):
            nbr_ijk = batch_ijk + np.array([di, dj, dk])
            valid = (
                (nbr_ijk[:, 0] >= 0) & (nbr_ijk[:, 0] < Nx) &
                (nbr_ijk[:, 1] >= 0) & (nbr_ijk[:, 1] < Ny) &
                (nbr_ijk[:, 2] >= 0) & (nbr_ijk[:, 2] < Nz)
            )
            nbr_ijk = nbr_ijk[valid]
            src_ijk = batch_ijk[valid]
            free_nbr = ~grid[nbr_ijk[:, 0], nbr_ijk[:, 1], nbr_ijk[:, 2]]
            nbr_ijk = nbr_ijk[free_nbr]
            src_ijk = src_ijk[free_nbr]
            if len(src_ijk) == 0:
                continue
            src_flat = np.ravel_multi_index(
                (src_ijk[:, 0], src_ijk[:, 1], src_ijk[:, 2]), grid.shape
            )
            dst_flat = np.ravel_multi_index(
                (nbr_ijk[:, 0], nbr_ijk[:, 1], nbr_ijk[:, 2]), grid.shape
            )
            src_nodes = np.array([flat_to_node.get(int(s), -1) for s in src_flat])
            dst_nodes = np.array([flat_to_node.get(int(d), -1) for d in dst_flat])
            mask = (src_nodes >= 0) & (dst_nodes >= 0)
            src_list.append(src_nodes[mask])
            dst_list.append(dst_nodes[mask])
            wt_list.append(np.full(mask.sum(), w, dtype=np.float32))

    src_arr = np.concatenate(src_list).astype(np.int32)
    dst_arr = np.concatenate(dst_list).astype(np.int32)
    wt_arr = np.concatenate(wt_list).astype(np.float32)

    gdf = cudf.DataFrame({"src": src_arr, "dst": dst_arr, "weight": wt_arr})
    G = cugraph.Graph()
    G.from_cudf_edgelist(gdf, source="src", destination="dst", edge_attr="weight")
    logger.info("[cuGraph] Graph: %s nodes, %s edges",
                G.number_of_nodes(), G.number_of_edges())

    wp_ijk = np.floor((waypoints - origin) / resolution).astype(int)
    wp_flat_ids = np.ravel_multi_index(
        (wp_ijk[:, 0], wp_ijk[:, 1], wp_ijk[:, 2]), grid.shape
    )
    wp_node_ids = np.array([flat_to_node.get(int(f), 0) for f in wp_flat_ids])

    matrix = np.zeros((N, N), dtype=np.float32)
    for i, src_node in enumerate(wp_node_ids):
        df = cugraph.shortest_path(G, src_node)
        df = df.to_pandas().set_index("vertex")
        for j, dst_node in enumerate(wp_node_ids):
            if i == j:
                continue
            dist = df.at[int(dst_node), "distance"] if int(dst_node) in df.index else np.inf
            matrix[i, j] = float(dist)
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

        np.save(grid_path, occupancy_grid.grid)
        cfg = {
            "grid_path": grid_path,
            "origin": occupancy_grid.origin.tolist(),
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
