#!/usr/bin/env python3
"""E11: End-to-End Pipeline

Runs the full inspection pipeline (stages 1-8) at different coverage targets
and measures per-stage timing, viewpoint count, achieved coverage, and
makespan.

Parameters: target_coverage {0.85, 0.90, 0.925, 0.95, 0.97, 0.99} x 3 seeds.
Fixed: 5 robots, targeted_50 strategy, 1500 candidates, Duke of Lancaster.

Usage:
    conda run -n isaaclab python -m experiments.e11_end_to_end
    conda run -n isaaclab python -m experiments.e11_end_to_end --plots_only
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

from experiments.common.config import ModelConfig, SEEDS_3, RESULTS_DIR
from experiments.common.runner import set_seed, timed, free_gpu_memory
from experiments.common.pipeline_setup import PipelineContext, DegenerateNormalsError
from experiments.common.persistence import save_run_result, load_run_result
from experiments.common.plotting import (
    setup_thesis_style, save_figure, stacked_bar, pareto_front,
    THESIS_COL, DOUBLE_COL, CATEGORICAL_COLORS,
)

from shared.types import Side
from visibility.set_cover import LazyGreedySetCover

from VRP.core.distance_matrix import compute_distance_matrix
from VRP.vrp.vrp_solver import solve_vrp
from VRP.core.types import VRPBackend, VRPResult, ExecutionResult
from VRP.mapf.route_executor import MultiAgentPathPlanner
from VRP.core.geometry import compute_start_grid
from VRP.vrp._helpers import per_vehicle_costs
from VRP.core.constants import ROBOT_RADIUS

logger = logging.getLogger(__name__)

COVERAGE_TARGETS = [0.85, 0.90, 0.925, 0.95, 0.97, 0.99]
N_ROBOTS = 5
N_CANDIDATES = 1500


# ═══════════════════════════════════════════════════════════════════════════
# Single run logic
# ═══════════════════════════════════════════════════════════════════════════

def _save_viz_e11(opt_result, exec_result, target_points, normals,
                  all_pos, home_pos, vrp_result, home_indices,
                  target_coverage, seed, viz_path):
    """Save full pipeline viz data for e11 replay."""
    data = {
        # Set cover
        "positions": opt_result.positions,
        "rotations": opt_result.rotations,
        "visibility_map": opt_result.visibility_map,
        "target_points": np.asarray(target_points, dtype=np.float32),
        "normals": np.asarray(normals, dtype=np.float32),
        # VRP geometry
        "all_pos": all_pos.astype(np.float32),
        "home_pos": home_pos.astype(np.float32),
        # Metadata
        "target_coverage": target_coverage,
        "seed": seed,
        "n_robots": len(home_indices),
        "vrp_status": vrp_result.status,
        "num_viewpoints": opt_result.num_viewpoints,
        "coverage": float(opt_result.total_coverage),
    }
    # VRP routes (full loops: depot → waypoints → depot)
    for r_idx, route in enumerate(vrp_result.routes):
        full_route = [home_indices[r_idx]] + list(route) + [home_indices[r_idx]]
        data[f"routes_r{r_idx}"] = np.array(full_route, dtype=np.int32)
    # MAPF trajectories (concatenated per robot)
    for r_idx, robot_legs in enumerate(exec_result.all_traj_positions):
        if len(robot_legs) > 0:
            data[f"traj_r{r_idx}"] = np.asarray(robot_legs, dtype=np.float32)
    # Waypoints per robot
    for r_idx, robot_wps in enumerate(exec_result.all_waypoints):
        if robot_wps:
            data[f"waypoints_r{r_idx}"] = np.array(robot_wps, dtype=np.float32)
    save_run_result(data, viz_path)


def run_single(target_coverage: float, seed: int,
               output_dir: str | None = None) -> dict:
    """Run full pipeline at one coverage target."""
    set_seed(seed)
    model_cfg = ModelConfig.duke_of_lancaster()
    ctx = PipelineContext(model_cfg)

    # Stage 1: Mesh loading
    with timed() as t_mesh:
        ctx.load_mesh()
    tm, o3d_mesh = ctx.load_mesh()
    bmin = np.asarray(tm.bounds[0], dtype=float)
    bmax = np.asarray(tm.bounds[1], dtype=float)

    # Stage 2: Surface sampling
    with timed() as t_surface:
        target_points, normals = ctx.sample_surface()  # fixed seed, disk-cached

    # Stage 3: Occupancy grid for sampling
    with timed() as t_og:
        ctx.build_sampling_og()

    # Stage 4: Viewpoint sampling
    with timed() as t_sample:
        sampler = ctx.build_sampler("weighted_curvature")
        n_uniform = int(N_CANDIDATES * 0.50)
        n_targeted = N_CANDIDATES - n_uniform
        pos_gpu, rot_gpu = sampler.sample(
            cp.arange(len(target_points)), n_uniform,
            side=Side.OUTSIDE, curvature_weighting=False,
        )
        vis_query = ctx.build_visibility_query("raycast")
        V_init, _ = vis_query.compute_visibility_batch(pos_gpu, rot_gpu)
        coverage_count = V_init.astype(cp.int32).sum(axis=0)
        uncovered = cp.where(coverage_count < 1)[0]
        if len(uncovered) > 0 and n_targeted > 0:
            t_pos, t_rot = sampler.sample(
                uncovered, n_targeted, side=Side.OUTSIDE,
                curvature_weighting=False,
            )
            pos_gpu = cp.concatenate([pos_gpu, t_pos])
            rot_gpu = cp.concatenate([rot_gpu, t_rot])

    # Stage 5: Visibility computation
    with timed() as t_vis:
        V, _ = vis_query.compute_visibility_batch(pos_gpu, rot_gpu)

    # Stage 6: Set cover optimization
    with timed() as t_opt:
        optimizer = LazyGreedySetCover(
            len(target_points), pos_gpu, rot_gpu, V,
        )
        opt_result = optimizer.optimize(
            target_coverage=target_coverage, max_viewpoints=1000,
        )

    num_viewpoints = opt_result.num_viewpoints
    coverage = float(opt_result.total_coverage)

    # Stage 7: VRP
    with timed() as t_vrp:
        K = N_ROBOTS
        robot_xyzs = compute_start_grid(K, bmin, bmax)
        home_pos = np.array(
            [[float(x[0]), float(x[1]), float(x[2])] for x in robot_xyzs],
            dtype=np.float32,
        )
        home_rot = np.tile(np.eye(3, dtype=np.float32), (K, 1, 1))
        vp_pos_np = opt_result.positions.get()
        vp_rot_np = opt_result.rotations.get()
        all_pos = np.vstack([home_pos, vp_pos_np])
        all_rot = np.concatenate([home_rot, vp_rot_np])
        home_indices = list(range(K))

        from visibility.sampling.utils.sampling_grid_builder import build_sampling_occupancy_grid
        _model_cfg = ModelConfig.duke_of_lancaster()
        og_vrp = build_sampling_occupancy_grid(
            mesh=o3d_mesh,
            frustum_far=_model_cfg.frustum.far,
            min_clearance=2 * ROBOT_RADIUS,
            resolution=0.20,
        )
        dist_matrix = compute_distance_matrix(og_vrp, cp.asarray(all_pos))
        vrp_result: VRPResult = solve_vrp(
            dist_matrix=dist_matrix, num_vehicles=K, depots=home_indices,
            backend=VRPBackend.CUOPT, time_limit=120,
        )

    pv = per_vehicle_costs(vrp_result.routes, dist_matrix, home_indices)
    makespan = max(pv) if pv else 0.0
    total_cost = vrp_result.total_cost

    # Stage 8: MAPF
    with timed() as t_mapf:
        routes = [
            [home_indices[i]] + list(r) + [home_indices[i]]
            for i, r in enumerate(vrp_result.routes)
        ]
        start_positions = [np.array(xyz, dtype=np.float32) for xyz in robot_xyzs]
        wp_pos_gpu = cp.asarray(all_pos, dtype=cp.float32)
        wp_rot_gpu = cp.asarray(all_rot, dtype=cp.float32)
        executor = MultiAgentPathPlanner(start_positions=start_positions, og=og_vrp)
        exec_result: ExecutionResult = executor.execute(
            routes=routes, waypoint_positions=wp_pos_gpu,
            waypoint_rotmats=wp_rot_gpu, home_indices=set(home_indices),
            dist_matrix=dist_matrix,
        )

    if output_dir is not None:
        viz_dir = os.path.join(output_dir, "viz")
        os.makedirs(viz_dir, exist_ok=True)
        viz_path = os.path.join(viz_dir, f"target={target_coverage}_seed={seed}")
        _save_viz_e11(opt_result, exec_result, target_points, normals,
                      all_pos, home_pos, vrp_result, home_indices,
                      target_coverage, seed, viz_path)

    return {
        "target_coverage": target_coverage,
        "seed": seed,
        "num_viewpoints": num_viewpoints,
        "coverage": coverage,
        "makespan": makespan,
        "total_cost": total_cost,
        "t_mesh": t_mesh.elapsed,
        "t_surface": t_surface.elapsed,
        "t_og": t_og.elapsed,
        "t_sample": t_sample.elapsed,
        "t_vis": t_vis.elapsed,
        "t_opt": t_opt.elapsed,
        "t_vrp": t_vrp.elapsed,
        "t_mapf": t_mapf.elapsed,
        "t_total": (t_mesh.elapsed + t_surface.elapsed + t_og.elapsed +
                    t_sample.elapsed + t_vis.elapsed + t_opt.elapsed +
                    t_vrp.elapsed + t_mapf.elapsed),
        "vrp_status": vrp_result.status,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Plot generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_plots(results: list[dict], output_dir: str):
    """Generate all E11 figures."""
    setup_thesis_style()
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    ok = [r for r in results if r.get("vrp_status") == "success"]
    if not ok:
        logger.warning("No successful runs for plotting")
        return

    targets = sorted(set(r["target_coverage"] for r in ok))
    target_labels = [f"{t*100:.1f}%" for t in targets]

    stage_names = ["Mesh", "Surface", "OG", "Sampling", "Visibility",
                   "Set Cover", "VRP", "MAPF"]
    stage_keys = ["t_mesh", "t_surface", "t_og", "t_sample", "t_vis",
                  "t_opt", "t_vrp", "t_mapf"]

    # ── Fig 1: Stacked bar - time per stage ─────────────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 4))
    stage_data = {}
    for sname, skey in zip(stage_names, stage_keys):
        stage_data[sname] = [
            np.mean([r[skey] for r in ok if r["target_coverage"] == t])
            for t in targets
        ]
    stacked_bar(ax, target_labels, stage_data,
                ylabel="Time (s)", title="Pipeline Time Breakdown")
    ax.set_xlabel("Target coverage")
    ax.tick_params(axis="x", rotation=25)
    save_figure(fig, os.path.join(fig_dir, "e11_time_breakdown"))

    # ── Fig 2: Pareto - coverage vs makespan ────────────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3.5))
    cov_vals = [r["coverage"] * 100 for r in ok]
    mk_vals = [r["makespan"] for r in ok]
    labels = [f"{r['target_coverage']*100:.0f}%/s{r['seed']}" for r in ok]
    pareto_front(ax, cov_vals, mk_vals, labels=labels,
                 xlabel="Coverage (%)", ylabel="Makespan (m)",
                 title="Coverage vs Makespan Pareto")
    save_figure(fig, os.path.join(fig_dir, "e11_pareto"))

    # ── Fig 3: Line - viewpoints vs target ──────────────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3))
    vp_means = [np.mean([r["num_viewpoints"] for r in ok if r["target_coverage"] == t])
                for t in targets]
    vp_stds = [np.std([r["num_viewpoints"] for r in ok if r["target_coverage"] == t])
               for t in targets]
    ax.errorbar([t * 100 for t in targets], vp_means, yerr=vp_stds,
                marker="o", capsize=3, color=CATEGORICAL_COLORS[0])
    ax.set_xlabel("Target coverage (%)")
    ax.set_ylabel("Selected viewpoints")
    ax.set_title("Viewpoints vs Coverage Target")
    save_figure(fig, os.path.join(fig_dir, "e11_viewpoints_vs_target"))

    # ── Fig 4: Line - makespan vs target ────────────────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3))
    mk_means = [np.mean([r["makespan"] for r in ok if r["target_coverage"] == t])
                for t in targets]
    mk_stds = [np.std([r["makespan"] for r in ok if r["target_coverage"] == t])
               for t in targets]
    ax.errorbar([t * 100 for t in targets], mk_means, yerr=mk_stds,
                marker="s", capsize=3, color=CATEGORICAL_COLORS[1])
    ax.set_xlabel("Target coverage (%)")
    ax.set_ylabel("Makespan (m)")
    ax.set_title("Makespan vs Coverage Target")
    save_figure(fig, os.path.join(fig_dir, "e11_makespan_vs_target"))

    # ── Fig 5: Pie chart - time breakdown at 95% ────────────────────
    target_95 = [r for r in ok if abs(r["target_coverage"] - 0.95) < 0.01]
    if target_95:
        fig, ax = plt.subplots(figsize=(THESIS_COL, THESIS_COL))
        mean_times = [np.mean([r[sk] for r in target_95]) for sk in stage_keys]
        nonzero = [(n, t) for n, t in zip(stage_names, mean_times) if t > 0.01]
        if nonzero:
            pie_names, pie_vals = zip(*nonzero)
            colors = [CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)]
                      for i in range(len(pie_names))]
            ax.pie(pie_vals, labels=pie_names, colors=colors, autopct="%.1f%%",
                   textprops={"fontsize": 8})
            ax.set_title("Time Breakdown at 95% Coverage")
            save_figure(fig, os.path.join(fig_dir, "e11_pie_95"))
        else:
            plt.close(fig)

    logger.info("E11 figures saved to %s", fig_dir)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="E11: End-to-End Pipeline")
    p.add_argument("--targets", type=float, nargs="+", default=COVERAGE_TARGETS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_3)
    p.add_argument("--output_dir",
                   default=os.path.join(RESULTS_DIR, "e11_end_to_end"))
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
        total = len(args.targets) * len(args.seeds)
        run_idx = 0

        for target in args.targets:
            for seed in args.seeds:
                run_idx += 1
                rpath = os.path.join(raw_dir, f"target={target}_seed={seed}")
                logger.info("[%d/%d] target=%.3f seed=%d",
                            run_idx, total, target, seed)

                try:
                    result = run_single(
                        target, seed,
                        output_dir=args.output_dir if seed == SEEDS_3[0] else None,
                    )
                    all_results.append(result)
                    save_run_result(result, rpath)
                    logger.info("  vps=%d cov=%.2f%% makespan=%.1f t=%.1fs",
                                result["num_viewpoints"],
                                result["coverage"] * 100,
                                result["makespan"],
                                result["t_total"])
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
