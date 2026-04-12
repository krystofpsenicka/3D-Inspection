#!/usr/bin/env python3
"""E7: VRP Alpha Blending

Sweeps alpha parameter (makespan vs total-distance objective weighting)
and measures the tradeoff.

Usage:
    conda run -n isaaclab python -m experiments.e07_vrp_alpha_blending
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

from experiments.common.config import ModelConfig, SEEDS_3, E07_ALPHAS, RESULTS_DIR
from experiments.common.persistence import save_run_result, load_run_result
from experiments.common.plotting import (
    setup_thesis_style, save_figure, dual_yaxis, THESIS_COL, CATEGORICAL_COLORS,
)

from shared.mesh_loader import load_and_transform_mesh
from shared.grid_builder_utils import build_occupancy_grid
from VRP.core.waypoint_loader import load_waypoints
from VRP.core.distance_matrix import compute_distance_matrix
from VRP.vrp.vrp_solver import solve_vrp
from VRP.core.types import VRPBackend
from VRP.core.geometry import compute_start_grid
from VRP.core.constants import (
    INFLATION_VOXELS, MESH_PATH, MESH_POSE, MESH_TARGET_LENGTH,
    ROBOT_RADIUS, VOXEL_RESOLUTION,
)

logger = logging.getLogger(__name__)

N_WAYPOINTS = 50
N_ROBOTS = 3


def run_single(alpha: float, seed: int, og, mesh_bounds_min,
               mesh_bounds_max) -> dict:
    np.random.seed(seed)
    cp.random.seed(seed)

    insp_pos, insp_rot = load_waypoints(
        source="random", n_random=N_WAYPOINTS, og=og, random_seed=seed,
    )
    K = N_ROBOTS
    robot_xyzs = compute_start_grid(K, mesh_bounds_min, mesh_bounds_max)
    home_pos = np.array([[float(x[0]), float(x[1]), float(x[2])] for x in robot_xyzs],
                        dtype=np.float32)
    home_rot = np.tile(np.eye(3, dtype=np.float32), (K, 1, 1))
    all_pos = np.vstack([home_pos, insp_pos])
    all_rot = np.concatenate([home_rot, insp_rot])
    home_indices = list(range(K))

    dist_matrix = compute_distance_matrix(og, all_pos)

    t0 = time.perf_counter()
    vrp_result = solve_vrp(
        dist_matrix=dist_matrix, num_vehicles=K, depots=home_indices,
        alpha=alpha, backend=VRPBackend.HIGHS, time_limit=120,
    )
    solve_time = time.perf_counter() - t0

    from VRP.vrp._helpers import per_vehicle_costs
    per_v = per_vehicle_costs(vrp_result.routes, dist_matrix, home_indices)
    makespan = max(per_v) if per_v else 0.0

    return {
        "alpha": alpha,
        "seed": seed,
        "makespan": makespan,
        "total_cost": vrp_result.total_cost,
        "per_vehicle_costs": per_v,
        "route_balance_ratio": makespan / min(c for c in per_v if c > 0) if any(c > 0 for c in per_v) else 0,
        "solve_time": solve_time,
        "status": vrp_result.status,
    }


def generate_plots(results: list[dict], output_dir: str):
    setup_thesis_style()
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    alphas = sorted(set(r["alpha"] for r in results))
    ok = [r for r in results if r["status"] == "success"]

    makespan_means = [np.mean([r["makespan"] for r in ok if r["alpha"] == a]) for a in alphas]
    cost_means = [np.mean([r["total_cost"] for r in ok if r["alpha"] == a]) for a in alphas]
    balance_means = [np.mean([r["route_balance_ratio"] for r in ok if r["alpha"] == a]) for a in alphas]

    # ── Fig 1: Dual y-axis ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3))
    dual_yaxis(ax, alphas, makespan_means, cost_means,
               "Makespan", "Total cost",
               ylabel1="Makespan (m)", ylabel2="Total cost (m)",
               title="Alpha Blending Tradeoff")
    ax.set_xlabel("Alpha (1=makespan, 0=total cost)")
    save_figure(fig, os.path.join(fig_dir, "e07_alpha_tradeoff"))

    # ── Fig 2: Stacked area - per vehicle costs ──────────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3))
    for i, a in enumerate(alphas):
        runs = [r for r in ok if r["alpha"] == a]
        if runs:
            mean_pv = np.mean([r["per_vehicle_costs"] for r in runs], axis=0)
            for v_idx, cost in enumerate(mean_pv):
                ax.bar(i, cost, bottom=sum(mean_pv[:v_idx]),
                       color=CATEGORICAL_COLORS[v_idx % len(CATEGORICAL_COLORS)],
                       label=f"Robot {v_idx}" if i == 0 else "")
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"{a:.2f}" for a in alphas])
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Route cost (m)")
    ax.set_title("Per-Vehicle Cost by Alpha")
    ax.legend(fontsize=7)
    save_figure(fig, os.path.join(fig_dir, "e07_per_vehicle"))

    # ── Fig 3: Balance ratio vs alpha ────────────────────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3))
    ax.plot(alphas, balance_means, "o-", color=CATEGORICAL_COLORS[2])
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Route balance ratio")
    ax.set_title("Route Balance vs. Alpha")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Perfect balance")
    ax.legend()
    save_figure(fig, os.path.join(fig_dir, "e07_balance"))

    logger.info("E7 figures saved to %s", fig_dir)


def main():
    p = argparse.ArgumentParser(description="E7: VRP Alpha Blending")
    p.add_argument("--alphas", type=float, nargs="+", default=E07_ALPHAS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_3)
    p.add_argument("--output_dir", default=os.path.join(RESULTS_DIR, "e07_vrp_alpha_blending"))
    p.add_argument("--plots_only", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)-8s %(name)s: %(message)s")
    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []

    if not args.plots_only:
        mesh = load_and_transform_mesh(MESH_PATH, MESH_TARGET_LENGTH, MESH_POSE)
        bmin = np.asarray(mesh.bounds[0], dtype=float)
        bmax = np.asarray(mesh.bounds[1], dtype=float)
        max_xyzs = compute_start_grid(N_ROBOTS, bmin, bmax)
        og = build_occupancy_grid(
            mesh=mesh, padding=1.0, inflation_voxels=INFLATION_VOXELS,
            resolution=VOXEL_RESOLUTION, fill_interior=True,
            extra_free_points=np.array(max_xyzs, dtype=np.float32),
            extra_margin_voxels=max(3, int(np.ceil(ROBOT_RADIUS / VOXEL_RESOLUTION))),
        )

        raw_dir = os.path.join(args.output_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        for alpha in args.alphas:
            for seed in args.seeds:
                logger.info("Running alpha=%.2f seed=%d", alpha, seed)
                result = run_single(alpha, seed, og, bmin, bmax)
                all_results.append(result)
                save_run_result(result, os.path.join(raw_dir, f"alpha={alpha}_seed={seed}"))
                logger.info("  makespan=%.1f total_cost=%.1f", result["makespan"], result["total_cost"])
    else:
        raw_dir = os.path.join(args.output_dir, "raw")
        for fname in sorted(os.listdir(raw_dir)):
            if fname.endswith(".json"):
                all_results.append(load_run_result(os.path.join(raw_dir, fname.replace(".json", ""))))

    if all_results:
        generate_plots(all_results, args.output_dir)


if __name__ == "__main__":
    main()
