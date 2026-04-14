#!/usr/bin/env python3
"""E8: VRP Solver Comparison

Compares cuOpt vs HiGHS VRP solvers on problem instances of varying size:
  - (20 waypoints, 2 robots)
  - (50 waypoints, 3 robots)
  - (75 waypoints, 5 robots)
Each configuration is run across 3 seeds.

Measures solve_time, makespan, total_cost, and solver status.

Usage:
    conda run -n isaaclab python -m experiments.e08_vrp_solver_comparison
    conda run -n isaaclab python -m experiments.e08_vrp_solver_comparison --plots_only
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import cupy as cp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.common.runner import free_gpu_memory
from experiments.common.config import ModelConfig, SEEDS_3, RESULTS_DIR
from experiments.common.persistence import save_run_result, load_run_result
from experiments.common.plotting import (
    setup_thesis_style, save_figure, grouped_bar,
    THESIS_COL, DOUBLE_COL, CATEGORICAL_COLORS,
)

import open3d as o3d
from shared.mesh_loader import load_and_transform_mesh
from shared.surface_sampler import SurfacePointSampler
from shared.types import Side
from visibility.sampling import WeightedViewpointSampler
from VRP.core.distance_matrix import compute_distance_matrix
from VRP.vrp.vrp_solver import solve_vrp
from VRP.core.types import VRPBackend
from VRP.core.geometry import compute_start_grid
from VRP.vrp._helpers import per_vehicle_costs
from VRP.core.constants import (
    MESH_PATH, MESH_POSE, MESH_TARGET_LENGTH,
    ROBOT_RADIUS,
)

logger = logging.getLogger(__name__)

# Problem configurations: (n_waypoints, n_robots)
PROBLEM_SIZES = [(20, 2), (50, 3), (75, 5)]

SOLVERS = ["highs"]  # cuopt added dynamically if available


def _available_solvers() -> list[str]:
    """Return list of available solver names."""
    solvers = ["highs"]
    try:
        _ = VRPBackend("cuopt")
        solvers.append("cuopt")
    except Exception:
        logger.warning("cuOpt backend not available, skipping")
    return solvers


# ═══════════════════════════════════════════════════════════════════════════
# Single run logic
# ═══════════════════════════════════════════════════════════════════════════

def run_single(n_waypoints: int, n_robots: int, solver: str,
               seed: int, og, sampler, mesh_bounds_min, mesh_bounds_max,
               viz_path: str | None = None) -> dict:
    """Run one VRP solver on one problem instance."""
    np.random.seed(seed)
    cp.random.seed(seed)

    pos_gpu, rot_gpu = sampler.sample(n_waypoints, side=Side.OUTSIDE)
    insp_pos = cp.asnumpy(pos_gpu).astype(np.float32)
    insp_rot = cp.asnumpy(rot_gpu).astype(np.float32)
    K = n_robots
    robot_xyzs = compute_start_grid(K, mesh_bounds_min, mesh_bounds_max)
    home_pos = np.array(
        [[float(x[0]), float(x[1]), float(x[2])] for x in robot_xyzs],
        dtype=np.float32,
    )
    home_rot = np.tile(np.eye(3, dtype=np.float32), (K, 1, 1))
    all_pos = np.vstack([home_pos, insp_pos])
    home_indices = list(range(K))

    dist_matrix = compute_distance_matrix(og, cp.asarray(all_pos))

    t0 = time.perf_counter()
    vrp_result = solve_vrp(
        dist_matrix=dist_matrix, num_vehicles=K, depots=home_indices,
        backend=VRPBackend(solver), time_limit=120,
    )
    solve_time = time.perf_counter() - t0

    pv = per_vehicle_costs(vrp_result.routes, dist_matrix, home_indices)
    makespan = max(pv) if pv else 0.0

    if viz_path is not None:
        viz_data = {
            "all_pos": all_pos.astype(np.float32),
            "home_pos": home_pos.astype(np.float32),
            "n_waypoints": n_waypoints,
            "n_robots": n_robots,
            "solver": solver,
            "seed": seed,
            "makespan": makespan,
            "status": vrp_result.status,
        }
        for r_idx, route in enumerate(vrp_result.routes):
            full_route = [home_indices[r_idx]] + list(route) + [home_indices[r_idx]]
            viz_data[f"routes_r{r_idx}"] = np.array(full_route, dtype=np.int32)
        save_run_result(viz_data, viz_path)

    return {
        "n_waypoints": n_waypoints,
        "n_robots": n_robots,
        "solver": solver,
        "seed": seed,
        "solve_time": solve_time,
        "makespan": makespan,
        "total_cost": vrp_result.total_cost,
        "status": vrp_result.status,
        "per_vehicle_costs": pv,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Plot generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_plots(results: list[dict], output_dir: str):
    """Generate all E8 figures from collected results."""
    setup_thesis_style()
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    ok = [r for r in results if r["status"] == "success"]
    if not ok:
        logger.warning("No successful runs for plotting")
        return

    solvers = sorted(set(r["solver"] for r in ok))
    sizes = sorted(set((r["n_waypoints"], r["n_robots"]) for r in ok))
    size_labels = [f"{nw}wp/{nr}r" for nw, nr in sizes]

    # ── Fig 1: Grouped bar - solve time by solver ───────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3.5))
    data = {}
    errs = {}
    for s in solvers:
        vals = [np.mean([r["solve_time"] for r in ok
                         if r["solver"] == s and r["n_waypoints"] == nw and r["n_robots"] == nr])
                for nw, nr in sizes]
        stds = [np.std([r["solve_time"] for r in ok
                        if r["solver"] == s and r["n_waypoints"] == nw and r["n_robots"] == nr])
                for nw, nr in sizes]
        data[s] = vals
        errs[s] = stds
    grouped_bar(ax, data, size_labels, yerr=errs,
                ylabel="Solve time (s)",
                title="VRP Solve Time by Solver")
    ax.set_xlabel("Problem size")
    save_figure(fig, os.path.join(fig_dir, "e08_solve_time"))

    # ── Fig 2: Grouped bar - makespan by solver ─────────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3.5))
    data = {}
    errs = {}
    for s in solvers:
        vals = [np.mean([r["makespan"] for r in ok
                         if r["solver"] == s and r["n_waypoints"] == nw and r["n_robots"] == nr])
                for nw, nr in sizes]
        stds = [np.std([r["makespan"] for r in ok
                        if r["solver"] == s and r["n_waypoints"] == nw and r["n_robots"] == nr])
                for nw, nr in sizes]
        data[s] = vals
        errs[s] = stds
    grouped_bar(ax, data, size_labels, yerr=errs,
                ylabel="Makespan (m)",
                title="VRP Makespan by Solver")
    ax.set_xlabel("Problem size")
    save_figure(fig, os.path.join(fig_dir, "e08_makespan"))

    # ── Fig 3: Scatter - solve time vs problem size ─────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3.5))
    for i, s in enumerate(solvers):
        sr = [r for r in ok if r["solver"] == s]
        xs = [r["n_waypoints"] for r in sr]
        ys = [r["solve_time"] for r in sr]
        ax.scatter(xs, ys, alpha=0.6, label=s,
                   color=CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)])
    ax.set_xlabel("Number of waypoints")
    ax.set_ylabel("Solve time (s)")
    ax.set_title("Solve Time vs Problem Size")
    ax.legend()
    save_figure(fig, os.path.join(fig_dir, "e08_scatter_time_vs_size"))

    logger.info("E8 figures saved to %s", fig_dir)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="E8: VRP Solver Comparison")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_3)
    p.add_argument("--output_dir",
                   default=os.path.join(RESULTS_DIR, "e08_vrp_solver_comparison"))
    p.add_argument("--plots_only", action="store_true",
                   help="Only regenerate plots from existing results")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
    )

    raw_dir = os.path.join(args.output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    all_results = []

    if not args.plots_only:
        solvers = _available_solvers()
        logger.info("Available solvers: %s", solvers)

        # Build shared occupancy grid and sampler from Duke of Lancaster
        _RESOLUTION = 0.20
        logger.info("Building occupancy grid (res=%.2f) ...", _RESOLUTION)
        mesh = load_and_transform_mesh(MESH_PATH, MESH_TARGET_LENGTH, MESH_POSE)
        bmin = np.asarray(mesh.bounds[0], dtype=float)
        bmax = np.asarray(mesh.bounds[1], dtype=float)

        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
        o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
        o3d_mesh.compute_vertex_normals()

        _model_cfg = ModelConfig.duke_of_lancaster()
        from visibility.sampling.utils.sampling_grid_builder import build_sampling_occupancy_grid
        og = build_sampling_occupancy_grid(
            mesh=o3d_mesh,
            frustum_far=_model_cfg.frustum.far,
            min_clearance=2 * ROBOT_RADIUS,
            resolution=_RESOLUTION,
        )
        logger.info("  Grid: %s res=%.2f", og.grid.shape, og.resolution)

        _pts_np, _norms_np = SurfacePointSampler().sample(
            o3d_mesh, _model_cfg.num_surface_points, seed=42)
        sampler = WeightedViewpointSampler(
            o3d_mesh,
            cp.asarray(_pts_np, dtype=cp.float32),
            cp.asarray(_norms_np, dtype=cp.float32),
            _model_cfg.frustum.far,
            collision_radius=ROBOT_RADIUS,
            occupancy_grid=og,
        )

        total = len(PROBLEM_SIZES) * len(solvers) * len(args.seeds)
        run_idx = 0

        for n_waypoints, n_robots in PROBLEM_SIZES:
            for solver in solvers:
                for seed in args.seeds:
                    run_idx += 1
                    rpath = os.path.join(
                        raw_dir,
                        f"wp={n_waypoints}_robots={n_robots}_solver={solver}_seed={seed}",
                    )

                    logger.info("[%d/%d] wp=%d robots=%d solver=%s seed=%d",
                                run_idx, total, n_waypoints, n_robots, solver, seed)

                    try:
                        viz_path = None
                        if seed == SEEDS_3[0] and n_waypoints == 50:
                            viz_dir = os.path.join(args.output_dir, "viz")
                            os.makedirs(viz_dir, exist_ok=True)
                            viz_path = os.path.join(
                                viz_dir,
                                f"wp={n_waypoints}_robots={n_robots}"
                                f"_solver={solver}_seed={seed}")
                        result = run_single(
                            n_waypoints, n_robots, solver, seed,
                            og, sampler, bmin, bmax,
                            viz_path=viz_path,
                        )
                        all_results.append(result)
                        save_run_result(result, rpath)
                        logger.info("  time=%.2fs makespan=%.1f cost=%.1f status=%s",
                                    result["solve_time"], result["makespan"],
                                    result["total_cost"], result["status"])
                    except Exception as e:
                        logger.error("  FAILED: %s", e, exc_info=True)
                    finally:
                        free_gpu_memory()
    else:
        for fname in sorted(os.listdir(raw_dir)):
            if fname.endswith(".json"):
                all_results.append(
                    load_run_result(os.path.join(raw_dir, fname.replace(".json", ""))))

    if all_results:
        generate_plots(all_results, args.output_dir)


if __name__ == "__main__":
    main()
