#!/usr/bin/env python3
"""E9: MAPF Priority Ordering

Tests how priority-based Multi-Agent Path Finding (MAPF) performs across
seeds. The MultiAgentPathPlanner already tries multiple orderings internally;
this experiment measures the resulting variance in makespan, trajectory steps,
and failure counts.

Fixed configuration: 5 robots, 50 waypoints, Duke of Lancaster.

Usage:
    conda run -n isaaclab python -m experiments.e09_mapf_priority
    conda run -n isaaclab python -m experiments.e09_mapf_priority --plots_only
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
from experiments.common.config import ModelConfig, SEEDS_5, RESULTS_DIR
from experiments.common.persistence import save_run_result, load_run_result
from experiments.common.plotting import (
    setup_thesis_style, save_figure, grouped_bar, violin_with_swarm,
    THESIS_COL, CATEGORICAL_COLORS,
)

import open3d as o3d
from shared.mesh_loader import load_and_transform_mesh
from shared.surface_sampler import SurfacePointSampler
from shared.types import Side
from visibility.sampling import WeightedViewpointSampler
from VRP.core.distance_matrix import compute_distance_matrix
from VRP.vrp.vrp_solver import solve_vrp
from VRP.core.types import VRPBackend, VRPResult, ExecutionResult
from VRP.mapf.mapf_planner import MultiAgentPathPlanner
from VRP.core.geometry import compute_start_grid
from VRP.core.collision import find_trajectory_collisions
from VRP.vrp._helpers import per_vehicle_costs
from VRP.core.constants import (
    MESH_PATH, MESH_POSE, MESH_TARGET_LENGTH,
    ROBOT_RADIUS,
)

logger = logging.getLogger(__name__)

N_WAYPOINTS = 50
N_ROBOTS = 5


# ═══════════════════════════════════════════════════════════════════════════
# Single run logic
# ═══════════════════════════════════════════════════════════════════════════

def run_single(seed: int, og, sampler, mesh_bounds_min, mesh_bounds_max) -> dict:
    """Run full VRP + MAPF pipeline for one seed."""
    np.random.seed(seed)
    cp.random.seed(seed)

    K = N_ROBOTS
    pos_gpu, rot_gpu = sampler.sample(N_WAYPOINTS, side=Side.OUTSIDE)
    insp_pos = cp.asnumpy(pos_gpu).astype(np.float32)
    insp_rot = cp.asnumpy(rot_gpu).astype(np.float32)
    robot_xyzs = compute_start_grid(K, mesh_bounds_min, mesh_bounds_max)
    home_pos = np.array(
        [[float(x[0]), float(x[1]), float(x[2])] for x in robot_xyzs],
        dtype=np.float32,
    )
    home_rot = np.tile(np.eye(3, dtype=np.float32), (K, 1, 1))
    all_pos = np.vstack([home_pos, insp_pos])
    all_rot = np.concatenate([home_rot, insp_rot])
    home_indices = list(range(K))

    # VRP
    t0 = time.perf_counter()
    dist_matrix = compute_distance_matrix(og, cp.asarray(all_pos))
    t_dist = time.perf_counter() - t0

    t0 = time.perf_counter()
    vrp_result: VRPResult = solve_vrp(
        dist_matrix=dist_matrix, num_vehicles=K, depots=home_indices,
        backend=VRPBackend.CUOPT, time_limit=120,
    )
    t_vrp = time.perf_counter() - t0

    pv = per_vehicle_costs(vrp_result.routes, dist_matrix, home_indices)
    vrp_makespan = max(pv) if pv else 0.0

    routes = [
        [home_indices[i]] + list(r) + [home_indices[i]]
        for i, r in enumerate(vrp_result.routes)
    ]

    # MAPF
    t0 = time.perf_counter()
    start_positions = [np.array(xyz, dtype=np.float32) for xyz in robot_xyzs]
    wp_pos_gpu = cp.asarray(all_pos, dtype=cp.float32)
    wp_rot_gpu = cp.asarray(all_rot, dtype=cp.float32)

    executor = MultiAgentPathPlanner(start_positions=start_positions, og=og)
    exec_result: ExecutionResult = executor.execute(
        routes=routes, waypoint_positions=wp_pos_gpu,
        waypoint_rotmats=wp_rot_gpu, home_indices=set(home_indices),
        dist_matrix=dist_matrix,
    )
    t_mapf = time.perf_counter() - t0

    total_traj_steps = max(
        len(t) for t in exec_result.all_traj_positions
    ) if exec_result.all_traj_positions else 0
    fail_count = sum(exec_result.fail_counts)
    residual_collisions = len(find_trajectory_collisions(
        exec_result.all_traj_positions))

    return {
        "seed": seed,
        "n_waypoints": N_WAYPOINTS,
        "n_robots": N_ROBOTS,
        "vrp_makespan": vrp_makespan,
        "total_cost": vrp_result.total_cost,
        "total_traj_steps": total_traj_steps,
        "fail_count": fail_count,
        "residual_collisions": residual_collisions,
        "t_dist": t_dist,
        "t_vrp": t_vrp,
        "t_mapf": t_mapf,
        "t_total": t_dist + t_vrp + t_mapf,
        "status": vrp_result.status,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Plot generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_plots(results: list[dict], output_dir: str):
    """Generate all E9 figures."""
    setup_thesis_style()
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    ok = [r for r in results if r["status"] == "success"]
    if not ok:
        logger.warning("No successful runs for plotting")
        return

    seeds = sorted(set(r["seed"] for r in ok))
    seed_labels = [str(s) for s in seeds]

    # ── Fig 1: Bar chart of makespan per seed ───────────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3.5))
    makespans = [next((r["vrp_makespan"] for r in ok if r["seed"] == s), 0)
                 for s in seeds]
    ax.bar(range(len(seeds)), makespans, 0.6,
           color=CATEGORICAL_COLORS[0])
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels(seed_labels)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Makespan (m)")
    ax.set_title("MAPF Makespan per Seed")
    save_figure(fig, os.path.join(fig_dir, "e09_makespan_per_seed"))

    # ── Fig 2: Bar chart of fail counts per seed ────────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3.5))
    fails = [next((r["fail_count"] for r in ok if r["seed"] == s), 0)
             for s in seeds]
    ax.bar(range(len(seeds)), fails, 0.6,
           color=CATEGORICAL_COLORS[1])
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels(seed_labels)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Fail count")
    ax.set_title("MAPF Replanning Failures per Seed")
    save_figure(fig, os.path.join(fig_dir, "e09_fail_counts"))

    # ── Fig 3: Box plot of trajectory steps ─────────────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3.5))
    traj_data = {"Traj steps": [r["total_traj_steps"] for r in ok]}
    violin_with_swarm(ax, traj_data,
                      ylabel="Total trajectory steps",
                      title="Trajectory Steps Distribution")
    save_figure(fig, os.path.join(fig_dir, "e09_traj_steps_box"))

    logger.info("E9 figures saved to %s", fig_dir)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="E9: MAPF Priority Ordering")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_5)
    p.add_argument("--output_dir",
                   default=os.path.join(RESULTS_DIR, "e09_mapf_priority"))
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
        # Build shared occupancy grid and sampler
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

        total = len(args.seeds)
        for run_idx, seed in enumerate(args.seeds, 1):
            rpath = os.path.join(raw_dir, f"seed={seed}")
            logger.info("[%d/%d] Running seed=%d", run_idx, total, seed)

            try:
                result = run_single(seed, og, sampler, bmin, bmax)
                all_results.append(result)
                save_run_result(result, rpath)
                logger.info("  makespan=%.1f steps=%d fails=%d t=%.1fs",
                            result["vrp_makespan"], result["total_traj_steps"],
                            result["fail_count"], result["t_total"])
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
