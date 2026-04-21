#!/usr/bin/env python3
"""E10: Space-Time A* Resolution

Sweeps voxel resolution for Space-Time A* path planning:
  {0.25, 0.50, 0.75, 1.0, 1.5} m
and measures the impact on planning time and makespan.

Fixed: 3 robots, 30 waypoints, Duke of Lancaster.

Usage:
    conda run -n isaaclab python -m experiments.e10_sta_resolution
    conda run -n isaaclab python -m experiments.e10_sta_resolution --plots_only
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
from experiments.common.config import ModelConfig, SEEDS_3, E10_RESOLUTIONS, RESULTS_DIR
from experiments.common.persistence import save_run_result, load_run_result
from experiments.common.plotting import (
    setup_thesis_style, save_figure, dual_yaxis,
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

N_WAYPOINTS = 30
N_ROBOTS = 3


# ═══════════════════════════════════════════════════════════════════════════
# Single run logic
# ═══════════════════════════════════════════════════════════════════════════

def run_single(resolution: float, seed: int, og, sampler, mesh_bounds_min,
               mesh_bounds_max) -> dict:
    """Run VRP + MAPF with a given Space-Time A* resolution."""
    np.random.seed(seed)
    cp.random.seed(seed)

    import VRP.core.constants as vrp_constants
    original_res = getattr(vrp_constants, "SPACE_TIME_RESOLUTION", None)
    vrp_constants.SPACE_TIME_RESOLUTION = resolution

    try:
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

        dist_matrix = compute_distance_matrix(og, cp.asarray(all_pos))

        vrp_result: VRPResult = solve_vrp(
            dist_matrix=dist_matrix, num_vehicles=K, depots=home_indices,
            backend=VRPBackend.CUOPT, time_limit=120,
        )

        pv = per_vehicle_costs(vrp_result.routes, dist_matrix, home_indices)
        vrp_makespan = max(pv) if pv else 0.0

        routes = [
            [home_indices[i]] + list(r) + [home_indices[i]]
            for i, r in enumerate(vrp_result.routes)
        ]

        # MAPF with the patched resolution
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
        planning_time = time.perf_counter() - t0

        total_traj_steps = max(
            len(t) for t in exec_result.all_traj_positions
        ) if exec_result.all_traj_positions else 0
        fail_count = sum(exec_result.fail_counts)
        residual_collisions = len(find_trajectory_collisions(
            exec_result.all_traj_positions))

        return {
            "resolution": resolution,
            "seed": seed,
            "planning_time": planning_time,
            "makespan": vrp_makespan,
            "total_traj_steps": total_traj_steps,
            "fail_count": fail_count,
            "residual_collisions": residual_collisions,
            "status": vrp_result.status,
        }
    finally:
        # Restore original value
        if original_res is not None:
            vrp_constants.SPACE_TIME_RESOLUTION = original_res


# ═══════════════════════════════════════════════════════════════════════════
# Plot generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_plots(results: list[dict], output_dir: str):
    """Generate all E10 figures."""
    setup_thesis_style()
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    ok = [r for r in results if r["status"] == "success"]
    if not ok:
        logger.warning("No successful runs for plotting")
        return

    resolutions = sorted(set(r["resolution"] for r in ok))

    time_means = [np.mean([r["planning_time"] for r in ok if r["resolution"] == res])
                  for res in resolutions]
    time_stds = [np.std([r["planning_time"] for r in ok if r["resolution"] == res])
                 for res in resolutions]
    makespan_means = [np.mean([r["makespan"] for r in ok if r["resolution"] == res])
                      for res in resolutions]
    makespan_stds = [np.std([r["makespan"] for r in ok if r["resolution"] == res])
                     for res in resolutions]

    # ── Fig 1: Line - planning time vs resolution ───────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3))
    ax.errorbar(resolutions, time_means, yerr=time_stds,
                marker="o", capsize=3, color=CATEGORICAL_COLORS[0])
    ax.set_xlabel("Space-Time A* resolution (m)")
    ax.set_ylabel("Planning time (s)")
    ax.set_title("MAPF Planning Time vs Resolution")
    save_figure(fig, os.path.join(fig_dir, "e10_time_vs_resolution"))

    # ── Fig 2: Line - makespan vs resolution ────────────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3))
    ax.errorbar(resolutions, makespan_means, yerr=makespan_stds,
                marker="s", capsize=3, color=CATEGORICAL_COLORS[1])
    ax.set_xlabel("Space-Time A* resolution (m)")
    ax.set_ylabel("Makespan (m)")
    ax.set_title("Makespan vs Resolution")
    save_figure(fig, os.path.join(fig_dir, "e10_makespan_vs_resolution"))

    # ── Fig 3: Dual y-axis - time/makespan tradeoff ─────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3))
    dual_yaxis(ax, resolutions, time_means, makespan_means,
               "Planning time", "Makespan",
               ylabel1="Planning time (s)", ylabel2="Makespan (m)",
               title="Resolution Tradeoff")
    ax.set_xlabel("Resolution (m)")
    save_figure(fig, os.path.join(fig_dir, "e10_tradeoff"))

    logger.info("E10 figures saved to %s", fig_dir)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="E10: Space-Time A* Resolution")
    p.add_argument("--resolutions", type=float, nargs="+", default=E10_RESOLUTIONS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_3)
    p.add_argument("--output_dir",
                   default=os.path.join(RESULTS_DIR, "e10_sta_resolution"))
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

        total = len(args.resolutions) * len(args.seeds)
        run_idx = 0

        for resolution in args.resolutions:
            for seed in args.seeds:
                run_idx += 1
                rpath = os.path.join(raw_dir, f"res={resolution}_seed={seed}")
                logger.info("[%d/%d] resolution=%.2f seed=%d",
                            run_idx, total, resolution, seed)

                try:
                    result = run_single(resolution, seed, og, sampler, bmin, bmax)
                    all_results.append(result)
                    save_run_result(result, rpath)
                    logger.info("  plan_time=%.2fs makespan=%.1f fails=%d",
                                result["planning_time"], result["makespan"],
                                result["fail_count"])
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
