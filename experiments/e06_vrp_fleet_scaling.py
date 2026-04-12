#!/usr/bin/env python3
"""E6: VRP Fleet Scaling

Extends evaluate_vrp.py with more fleet sizes (up to 10), more seeds,
Held-Karp lower bounds, and bottleneck analysis.

Usage:
    conda run -n isaaclab python -m experiments.e06_vrp_fleet_scaling
    conda run -n isaaclab python -m experiments.e06_vrp_fleet_scaling --fleet_sizes 1 2 3 --waypoint_counts 10 20
"""

from __future__ import annotations

import argparse
import csv
import itertools
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict

import cupy as cp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.common.config import (
    ModelConfig, SEEDS_5, E06_FLEET_SIZES, E06_WAYPOINT_COUNTS, RESULTS_DIR,
)
from experiments.common.lower_bounds import held_karp_tsp_lb, fleet_tsp_lb
from experiments.common.plotting import (
    setup_thesis_style, save_figure, grouped_bar, violin_with_swarm,
    stacked_bar, heatmap_annotated, THESIS_COL, DOUBLE_COL, CATEGORICAL_COLORS,
)

from shared.mesh_loader import load_and_transform_mesh
from shared.grid_builder_utils import build_occupancy_grid
from VRP.core.waypoint_loader import load_waypoints
from VRP.core.distance_matrix import compute_distance_matrix
from VRP.vrp.vrp_solver import solve_vrp
from VRP.core.types import VRPBackend, VRPResult, ExecutionResult
from VRP.mapf.route_executor import MultiAgentPathPlanner
from VRP.core.geometry import compute_start_grid
from VRP.core.collision import find_trajectory_collisions
from VRP.core.constants import (
    INFLATION_VOXELS, MESH_PATH, MESH_POSE, MESH_TARGET_LENGTH,
    ROBOT_RADIUS, VOXEL_RESOLUTION,
)

logger = logging.getLogger(__name__)


@dataclass
class RunMetrics:
    fleet_size: int = 0
    n_waypoints: int = 0
    seed: int = 0
    status: str = ""
    total_cost: float = 0.0
    makespan: float = 0.0
    shortest_route: float = 0.0
    route_balance_ratio: float = 0.0
    route_cost_std: float = 0.0
    mean_route_cost: float = 0.0
    total_traj_steps: int = 0
    residual_collisions: int = 0
    fail_count_total: int = 0
    t_dist_matrix: float = 0.0
    t_vrp_solve: float = 0.0
    t_trajectory: float = 0.0
    t_total: float = 0.0


def run_single(fleet_size, n_waypoints, seed, og, mesh_bounds_min,
               mesh_bounds_max, solver_backend=VRPBackend.HIGHS) -> RunMetrics:
    m = RunMetrics(fleet_size=fleet_size, n_waypoints=n_waypoints, seed=seed)
    t_total_start = time.perf_counter()

    try:
        insp_positions, insp_rotmats = load_waypoints(
            source="random", n_random=n_waypoints, og=og, random_seed=seed,
        )
        K = fleet_size
        robot_start_xyzs = compute_start_grid(K, mesh_bounds_min, mesh_bounds_max)
        home_positions = np.array(
            [[float(xyz[0]), float(xyz[1]), float(xyz[2])] for xyz in robot_start_xyzs],
            dtype=np.float32,
        )
        home_rotmats = np.tile(np.eye(3, dtype=np.float32), (K, 1, 1))
        all_positions = np.vstack([home_positions, insp_positions])
        all_rotmats = np.concatenate([home_rotmats, insp_rotmats])
        home_indices = list(range(K))

        t0 = time.perf_counter()
        dist_matrix = compute_distance_matrix(og, all_positions)
        m.t_dist_matrix = time.perf_counter() - t0

        t0 = time.perf_counter()
        vrp_result: VRPResult = solve_vrp(
            dist_matrix=dist_matrix, num_vehicles=K,
            depots=home_indices, backend=solver_backend, time_limit=120,
        )
        m.t_vrp_solve = time.perf_counter() - t0
        m.status = vrp_result.status
        m.total_cost = vrp_result.total_cost

        if not any(vrp_result.routes):
            m.status = "empty_routes"
            m.t_total = time.perf_counter() - t_total_start
            return m

        routes = [
            [home_indices[i]] + list(r) + [home_indices[i]]
            for i, r in enumerate(vrp_result.routes)
        ]

        from VRP.vrp._helpers import per_vehicle_costs
        rc = np.array(per_vehicle_costs(vrp_result.routes, dist_matrix, home_indices))
        m.makespan = float(rc.max())
        m.shortest_route = float(rc[rc > 0].min()) if np.any(rc > 0) else 0.0
        m.mean_route_cost = float(rc.mean())
        m.route_cost_std = float(rc.std())
        m.route_balance_ratio = m.makespan / m.shortest_route if m.shortest_route > 0 else float("inf")

        t0 = time.perf_counter()
        start_positions = [np.array(xyz, dtype=np.float32) for xyz in robot_start_xyzs]
        wp_pos_gpu = cp.asarray(all_positions, dtype=cp.float32)
        wp_rot_gpu = cp.asarray(all_rotmats, dtype=cp.float32)

        executor = MultiAgentPathPlanner(start_positions=start_positions, og=og)
        exec_result: ExecutionResult = executor.execute(
            routes=routes, waypoint_positions=wp_pos_gpu,
            waypoint_rotmats=wp_rot_gpu, home_indices=set(home_indices),
            dist_matrix=dist_matrix,
        )
        m.t_trajectory = time.perf_counter() - t0

        m.total_traj_steps = max(
            len(t) for t in exec_result.all_traj_positions
        ) if exec_result.all_traj_positions else 0
        m.fail_count_total = sum(exec_result.fail_counts)
        m.residual_collisions = len(find_trajectory_collisions(
            exec_result.all_traj_positions))

    except Exception as e:
        logger.error("Run failed: %s", e, exc_info=True)
        m.status = f"error: {e}"

    m.t_total = time.perf_counter() - t_total_start
    return m


def generate_plots(all_metrics, fleet_sizes, waypoint_counts, held_karp_bounds,
                   output_dir):
    setup_thesis_style()
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    successful = [m for m in all_metrics if m.status == "success"]
    if not successful:
        logger.warning("No successful runs for plotting")
        return

    from collections import defaultdict
    groups = defaultdict(list)
    for r in successful:
        groups[(r.fleet_size, r.n_waypoints)].append(r)

    wp_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(waypoint_counts)))

    # ── Fig 1: Makespan vs fleet (with LB) ───────────────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3))
    for wi, nw in enumerate(waypoint_counts):
        xs, means, stds = [], [], []
        for k in fleet_sizes:
            vals = [r.makespan for r in groups.get((k, nw), [])]
            if vals:
                xs.append(k)
                means.append(np.mean(vals))
                stds.append(np.std(vals))
        if xs:
            ax.errorbar(xs, means, yerr=stds, marker="o", color=wp_colors[wi],
                        label=f"{nw} wps", capsize=2)
    # Add Held-Karp lower bound
    if held_karp_bounds:
        for nw, hk_lb in held_karp_bounds.items():
            lbs = [hk_lb / k for k in fleet_sizes if k > 0]
            ax.plot(fleet_sizes, lbs, "--", alpha=0.4, label=f"HK/{'{'}k{'}'} ({nw}wp)")
    ax.set_xlabel("Fleet size")
    ax.set_ylabel("Makespan (m)")
    ax.set_title("Makespan vs. Fleet Size")
    ax.legend(fontsize=6, ncol=2)
    save_figure(fig, os.path.join(fig_dir, "e06_makespan_vs_fleet"))

    # ── Fig 2: Speedup vs fleet ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3))
    for wi, nw in enumerate(waypoint_counts):
        baseline_vals = [r.makespan for r in groups.get((1, nw), [])]
        if not baseline_vals:
            continue
        baseline = np.mean(baseline_vals)
        xs, means = [], []
        for k in fleet_sizes:
            vals = [baseline / r.makespan for r in groups.get((k, nw), []) if r.makespan > 0]
            if vals:
                xs.append(k)
                means.append(np.mean(vals))
        if xs:
            ax.plot(xs, means, "o-", color=wp_colors[wi], label=f"{nw} wps")
    ax.plot(fleet_sizes, fleet_sizes, "k--", alpha=0.4, label="Ideal linear")
    ax.set_xlabel("Fleet size")
    ax.set_ylabel("Speedup")
    ax.set_title("Makespan Speedup vs. Fleet Size")
    ax.legend(fontsize=6, ncol=2)
    save_figure(fig, os.path.join(fig_dir, "e06_speedup"))

    # ── Fig 3: Stacked bar - timing breakdown ────────────────────────
    mid_wps = waypoint_counts[len(waypoint_counts) // 2]
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3))
    stage_data = {"Dist matrix": [], "VRP solve": [], "Trajectory": []}
    used_fleets = []
    for k in fleet_sizes:
        ok = groups.get((k, mid_wps), [])
        if ok:
            used_fleets.append(str(k))
            stage_data["Dist matrix"].append(np.mean([r.t_dist_matrix for r in ok]))
            stage_data["VRP solve"].append(np.mean([r.t_vrp_solve for r in ok]))
            stage_data["Trajectory"].append(np.mean([r.t_trajectory for r in ok]))
    if used_fleets:
        stacked_bar(ax, used_fleets, stage_data,
                    ylabel="Time (s)", title=f"Timing ({mid_wps} waypoints)")
        save_figure(fig, os.path.join(fig_dir, "e06_timing_breakdown"))

    # ── Fig 7: Heatmap - fleet x waypoints -> makespan ───────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 4))
    fleet_labels = [str(k) for k in fleet_sizes]
    wp_labels = [str(nw) for nw in waypoint_counts]
    hm_vals = np.zeros((len(fleet_sizes), len(waypoint_counts)))
    for i, k in enumerate(fleet_sizes):
        for j, nw in enumerate(waypoint_counts):
            vals = [r.makespan for r in groups.get((k, nw), [])]
            hm_vals[i, j] = np.mean(vals) if vals else 0
    heatmap_annotated(ax, fleet_labels, wp_labels, hm_vals, fmt=".0f",
                      title="Makespan (m)", xlabel="Waypoints", ylabel="Fleet size")
    save_figure(fig, os.path.join(fig_dir, "e06_heatmap"))

    logger.info("E6 figures saved to %s", fig_dir)


def main():
    p = argparse.ArgumentParser(description="E6: VRP Fleet Scaling")
    p.add_argument("--fleet_sizes", type=int, nargs="+", default=E06_FLEET_SIZES)
    p.add_argument("--waypoint_counts", type=int, nargs="+", default=E06_WAYPOINT_COUNTS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_5)
    p.add_argument("--solver", default="highs", choices=["cuopt", "highs"])
    p.add_argument("--output_dir", default=os.path.join(RESULTS_DIR, "e06_vrp_fleet_scaling"))
    p.add_argument("--plots_only", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    os.makedirs(args.output_dir, exist_ok=True)
    solver_backend = VRPBackend(args.solver)

    all_metrics = []

    if not args.plots_only:
        # Build shared OG
        logger.info("Building occupancy grid ...")
        mesh = load_and_transform_mesh(MESH_PATH, MESH_TARGET_LENGTH, MESH_POSE)
        mesh_bounds_min = np.asarray(mesh.bounds[0], dtype=float)
        mesh_bounds_max = np.asarray(mesh.bounds[1], dtype=float)

        max_fleet = max(args.fleet_sizes)
        max_depot_xyzs = compute_start_grid(max_fleet, mesh_bounds_min, mesh_bounds_max)
        extra_free = np.array(max_depot_xyzs, dtype=np.float32)
        extra_margin = max(3, int(np.ceil(ROBOT_RADIUS / VOXEL_RESOLUTION)))
        og = build_occupancy_grid(
            mesh=mesh, padding=1.0, inflation_voxels=INFLATION_VOXELS,
            resolution=VOXEL_RESOLUTION, fill_interior=True,
            extra_free_points=extra_free, extra_margin_voxels=extra_margin,
        )
        logger.info("  Grid: %s res=%.2f", og.grid.shape, og.resolution)

        # Compute Held-Karp bounds for representative waypoint counts
        held_karp_bounds = {}

        configs = list(itertools.product(args.fleet_sizes, args.waypoint_counts, args.seeds))
        total = len(configs)

        csv_path = os.path.join(args.output_dir, "results.csv")
        fieldnames = ["run_id"] + list(RunMetrics.__dataclass_fields__.keys())

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for run_id, (k, nw, seed) in enumerate(configs, 1):
                logger.info("=== Run %d/%d: fleet=%d wps=%d seed=%d ===",
                            run_id, total, k, nw, seed)
                m = run_single(k, nw, seed, og, mesh_bounds_min, mesh_bounds_max,
                               solver_backend)
                all_metrics.append(m)
                row = asdict(m)
                row["run_id"] = run_id
                writer.writerow(row)
                f.flush()
                logger.info("  status=%s makespan=%.1f t_total=%.1fs",
                            m.status, m.makespan, m.t_total)
    else:
        csv_path = os.path.join(args.output_dir, "results.csv")
        if os.path.exists(csv_path):
            import csv as csv_mod
            with open(csv_path) as f:
                reader = csv_mod.DictReader(f)
                for row in reader:
                    m = RunMetrics()
                    for k, v in row.items():
                        if k == "run_id":
                            continue
                        if hasattr(m, k):
                            field_type = type(getattr(m, k))
                            try:
                                setattr(m, k, field_type(v))
                            except (ValueError, TypeError):
                                setattr(m, k, v)
                    all_metrics.append(m)
        held_karp_bounds = {}

    if all_metrics:
        generate_plots(all_metrics, args.fleet_sizes, args.waypoint_counts,
                       held_karp_bounds, args.output_dir)


if __name__ == "__main__":
    main()
