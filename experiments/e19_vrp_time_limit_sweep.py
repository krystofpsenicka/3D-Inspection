#!/usr/bin/env python3
"""E19: VRP Time-Limit Sweep (full VRP + MAPF pipeline)

Holds everything constant except the VRP solver ``time_limit`` and observes
how more VRP search budget propagates into the VRP objective (meters) and
— after MAPF planning — into the final makespan / total time in seconds.

Fixed configuration (overridable via CLI):
  - Model: Duke of Lancaster
  - Fleet: 5 robots (``E19_FLEET_SIZE``)
  - Waypoints: 75 (``E19_N_WAYPOINTS``)
  - Sampler: weighted_curvature (SDF² + curvature bias)
  - Alpha: 0.5 (matches e06)
  - Backend: cuOpt
  - MAPF: Space-Time A* with default resolution

Per (seed, time_limit) we record:
  - VRP ``objective_value`` / ``makespan`` / ``total_cost`` (meters)
  - MAPF makespan and total time in seconds (derived via SPACE_TIME_DT)
  - Stage timings: ``t_vrp_solve``, ``t_mapf``
  - Lower bounds: VRP (meters) + post-MAPF (seconds). LBs depend only on
    the sampled instance, so they are computed once per seed and copied
    to each time_limit row.

Usage:
    conda run -n isaaclab python -m experiments.e19_vrp_time_limit_sweep
    conda run -n isaaclab python -m experiments.e19_vrp_time_limit_sweep --plots_only
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import cupy as cp
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.common.config import (
    ModelConfig, SEEDS_3, RESULTS_DIR,
    E19_TIME_LIMITS, E19_FLEET_SIZE, E19_N_WAYPOINTS,
)
from experiments.common.runner import free_gpu_memory, handle_row_exception, is_oom
from experiments.common.persistence import save_run_result, load_run_result
from experiments.common.lb_sidecar import (
    compute_all_lbs, recompute_lbs, save_lb_json, load_raw_lb_dir,
)
from experiments.common.plotting import (
    setup_thesis_style, save_figure, stacked_bar,
    THESIS_COL, DOUBLE_COL, CATEGORICAL_COLORS,
)

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
    ROBOT_RADIUS, AUV_CRUISE_SPEED, SPACE_TIME_DWELL_S, SPACE_TIME_DT,
)

logger = logging.getLogger(__name__)

_ALPHA = 0.5


# ═══════════════════════════════════════════════════════════════════════════
# Setup helpers
# ═══════════════════════════════════════════════════════════════════════════

def _build_setup(resolution: float = 0.20):
    """Load the Duke mesh and build OG + curvature-weighted sampler once."""
    logger.info("Loading mesh and building occupancy grid (res=%.2f) ...", resolution)
    mesh = load_and_transform_mesh(MESH_PATH, MESH_TARGET_LENGTH, MESH_POSE)
    bmin = np.asarray(mesh.bounds[0], dtype=float)
    bmax = np.asarray(mesh.bounds[1], dtype=float)

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    o3d_mesh.compute_vertex_normals()

    model_cfg = ModelConfig.duke_of_lancaster()
    from visibility.sampling.utils.sampling_grid_builder import build_sampling_occupancy_grid
    og = build_sampling_occupancy_grid(
        mesh=o3d_mesh,
        frustum_far=model_cfg.frustum.far,
        min_clearance=2 * ROBOT_RADIUS,
        resolution=resolution,
    )
    logger.info("  Grid: %s res=%.2f", og.grid.shape, og.resolution)

    pts_np, norms_np = SurfacePointSampler().sample(
        o3d_mesh, model_cfg.num_surface_points, seed=42)
    sampler = WeightedViewpointSampler(
        o3d_mesh,
        cp.asarray(pts_np, dtype=cp.float32),
        cp.asarray(norms_np, dtype=cp.float32),
        model_cfg.frustum.far,
        collision_radius=ROBOT_RADIUS,
        occupancy_grid=og,
    )
    return og, sampler, bmin, bmax, model_cfg


def _sample_instance(seed: int, og, sampler, bmin, bmax):
    """Sample the frozen instance for a seed. Shared across all time_limits
    so every row of this seed sees identical waypoints + distance matrix."""
    cp.random.seed(seed)
    pos_gpu, rot_gpu = sampler.sample(
        E19_N_WAYPOINTS, side=Side.OUTSIDE, curvature_weighting=True,
    )
    insp_pos = cp.asnumpy(pos_gpu).astype(np.float32)
    insp_rot = cp.asnumpy(rot_gpu).astype(np.float32)

    K = E19_FLEET_SIZE
    robot_xyzs = compute_start_grid(K, bmin, bmax)
    home_pos = np.array(
        [[float(x[0]), float(x[1]), float(x[2])] for x in robot_xyzs],
        dtype=np.float32,
    )
    home_rot = np.tile(np.eye(3, dtype=np.float32), (K, 1, 1))
    all_pos = np.vstack([home_pos, insp_pos])
    all_rot = np.concatenate([home_rot, insp_rot])
    home_indices = list(range(K))
    start_positions = [np.array(xyz, dtype=np.float32) for xyz in robot_xyzs]

    dist_matrix = compute_distance_matrix(og, cp.asarray(all_pos))
    return {
        "K": K,
        "all_pos": all_pos,
        "all_rot": all_rot,
        "home_indices": home_indices,
        "start_positions": start_positions,
        "dist_matrix": dist_matrix,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Single run (reuses the frozen instance)
# ═══════════════════════════════════════════════════════════════════════════

def run_single(time_limit: int, seed: int, instance: dict, og):
    """Run VRP (with given time_limit) + MAPF on the pre-sampled instance.

    Returns ``(main, lb)``. ``main`` is the user-facing main result (no LB
    fields). ``lb`` is the sidecar dict with analytical LBs + the cuOpt
    dual bound captured from this specific solve.
    """
    K = instance["K"]
    dist_matrix = instance["dist_matrix"]
    home_indices = instance["home_indices"]
    all_pos = instance["all_pos"]
    all_rot = instance["all_rot"]
    start_positions = instance["start_positions"]

    # Stage 7: VRP
    t0 = time.perf_counter()
    vrp_result: VRPResult = solve_vrp(
        dist_matrix=dist_matrix, num_vehicles=K, depots=home_indices,
        alpha=_ALPHA, backend=VRPBackend.CUOPT, time_limit=time_limit,
    )
    t_vrp = time.perf_counter() - t0

    vrp_makespan_m = vrp_result.makespan
    vrp_total_cost_m = vrp_result.total_cost
    vrp_objective_m = vrp_result.objective_value
    vrp_status = vrp_result.status

    # LB sidecar — analytical LBs + cuOpt dual bound from *this* solve.
    lb: dict | None = None
    try:
        lb = compute_all_lbs(
            dist_matrix, home_indices, K, E19_N_WAYPOINTS, _ALPHA,
            include_mapf=True,
            vrp_best_bound_m=vrp_result.best_bound,
            vrp_objective_value_m=vrp_result.objective_value,
        )
    except Exception as e:
        logger.warning("LB computation failed: %s", e)

    out = {
        "time_limit": time_limit,
        "seed": seed,
        "alpha": _ALPHA,
        "vrp_status": vrp_status,
        "vrp_makespan_m": float(vrp_makespan_m),
        "vrp_total_cost_m": float(vrp_total_cost_m),
        "vrp_objective_m": float(vrp_objective_m),
        "t_vrp_solve": t_vrp,
        # Defaults for MAPF (filled below if VRP succeeded)
        "mapf_status": "skipped",
        "mapf_makespan_s": float("nan"),
        "mapf_total_time_s": float("nan"),
        "mapf_objective_s": float("nan"),
        "mapf_fail_count": 0,
        "mapf_residual_collisions": 0,
        "t_mapf": 0.0,
    }

    if not any(vrp_result.routes):
        out["vrp_status"] = "empty_routes"
        return out, lb

    # Stage 8: MAPF
    try:
        routes = [
            [home_indices[i]] + list(r) + [home_indices[i]]
            for i, r in enumerate(vrp_result.routes)
        ]
        wp_pos_gpu = cp.asarray(all_pos, dtype=cp.float32)
        wp_rot_gpu = cp.asarray(all_rot, dtype=cp.float32)

        t0 = time.perf_counter()
        executor = MultiAgentPathPlanner(start_positions=start_positions, og=og)
        exec_result: ExecutionResult = executor.execute(
            routes=routes, waypoint_positions=wp_pos_gpu,
            waypoint_rotmats=wp_rot_gpu, home_indices=set(home_indices),
            dist_matrix=dist_matrix, alpha=_ALPHA,
        )
        out["t_mapf"] = time.perf_counter() - t0

        # Convert step counts to seconds via the Space-Time A* timestep.
        if exec_result.all_traj_positions:
            per_robot_steps = [len(t) for t in exec_result.all_traj_positions]
            makespan_steps = max(per_robot_steps)
            total_steps = sum(per_robot_steps)
            out["mapf_makespan_s"] = float(makespan_steps * SPACE_TIME_DT)
            out["mapf_total_time_s"] = float(total_steps * SPACE_TIME_DT)
            out["mapf_objective_s"] = float(
                _ALPHA * out["mapf_makespan_s"]
                + (1.0 - _ALPHA) * out["mapf_total_time_s"]
            )

        out["mapf_fail_count"] = int(sum(exec_result.fail_counts))
        out["mapf_residual_collisions"] = int(
            len(find_trajectory_collisions(exec_result.all_traj_positions))
        )
        out["mapf_status"] = "success"
    except Exception as e:
        if is_oom(e):
            # Propagate so the outer per-row handler can exit under --resume.
            raise
        logger.error("MAPF failed: %s", e, exc_info=True)
        out["mapf_status"] = f"error: {e}"

    return out, lb


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def _per_tl(results, time_limits, metric):
    means, stds = [], []
    for tl in time_limits:
        vals = [r[metric] for r in results if r["time_limit"] == tl
                and not (isinstance(r.get(metric), float) and np.isnan(r[metric]))]
        means.append(float(np.mean(vals)) if vals else float("nan"))
        stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
    return np.array(means), np.array(stds)


def _lb_stem_for(r: dict) -> str:
    return f"time_limit={r['time_limit']}_seed={r['seed']}"


def _lb_band(results, time_limits, lb_by_stem: dict, lb_key: str):
    """Per-seed LB min/max across time_limits. Each seed is a different
    instance, so draw the spread rather than a single scalar. Reads from
    the sidecar dict loaded by ``load_raw_lb_dir``."""
    lb_min, lb_max = [], []
    for tl in time_limits:
        vals = []
        for r in results:
            if r["time_limit"] != tl:
                continue
            lb = lb_by_stem.get(_lb_stem_for(r), {})
            v = float(lb.get(lb_key, 0.0))
            if v > 0.0:
                vals.append(v)
        if vals:
            lb_min.append(float(np.min(vals)))
            lb_max.append(float(np.max(vals)))
        else:
            lb_min.append(float("nan"))
            lb_max.append(float("nan"))
    return lb_min, lb_max


def generate_plots(results: list[dict], output_dir: str,
                   lb_by_stem: dict | None = None):
    setup_thesis_style()
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    lb_by_stem = lb_by_stem or {}

    ok = [r for r in results if r.get("vrp_status") in ("success", "time_limit")]
    if not ok:
        logger.warning("No runs for plotting")
        return

    time_limits = sorted(set(r["time_limit"] for r in ok))
    labels = [str(t) for t in time_limits]

    def _line_plot(metric, ylabel, title, fname,
                   lb_key=None, cuopt_key=None):
        fig, ax = plt.subplots(figsize=(THESIS_COL, 3.2))
        m, s = _per_tl(ok, time_limits, metric)
        ax.errorbar(time_limits, m, yerr=s, marker="o", capsize=3,
                    color=CATEGORICAL_COLORS[0], linewidth=1.5,
                    label="Incumbent")
        has_legend = False
        if lb_key is not None:
            lb_min, lb_max = _lb_band(ok, time_limits, lb_by_stem, lb_key)
            if any(not np.isnan(v) for v in lb_min):
                ax.fill_between(time_limits, lb_min, lb_max, alpha=0.15,
                                color=CATEGORICAL_COLORS[3], linewidth=0,
                                label="Analytical LB band")
                has_legend = True
        if cuopt_key is not None:
            lb_min, lb_max = _lb_band(ok, time_limits, lb_by_stem, cuopt_key)
            if any(not np.isnan(v) for v in lb_min):
                ax.fill_between(time_limits, lb_min, lb_max, alpha=0.25,
                                color=CATEGORICAL_COLORS[2], linewidth=0,
                                label="cuOpt bound band")
                has_legend = True
        if has_legend:
            ax.legend(fontsize=7)
        ax.set_xlabel("VRP time_limit (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        save_figure(fig, os.path.join(fig_dir, fname))

    # ── VRP metrics (meters) with analytical LB + cuOpt bound bands ────
    _line_plot("vrp_objective_m",
               "VRP objective (m)",
               "VRP Objective vs Time Limit",
               "e19_vrp_objective_vs_timelimit",
               lb_key="vrp_objective_lb_m",
               cuopt_key="vrp_objective_best_bound_m")
    _line_plot("vrp_makespan_m",
               "VRP makespan (m)",
               "VRP Makespan vs Time Limit",
               "e19_vrp_makespan_vs_timelimit",
               lb_key="vrp_makespan_lb_m")
    _line_plot("vrp_total_cost_m",
               "VRP total cost (m)",
               "VRP Total Cost vs Time Limit",
               "e19_vrp_total_cost_vs_timelimit",
               lb_key="vrp_total_cost_lb_m")

    # ── Post-MAPF metrics (seconds) with LB bands ──────────────────────
    _line_plot("mapf_makespan_s",
               "MAPF makespan (s)",
               "MAPF Makespan vs VRP Time Limit",
               "e19_mapf_makespan_s_vs_timelimit",
               lb_key="mapf_makespan_time_lb_s")
    _line_plot("mapf_total_time_s",
               "MAPF total time (s)",
               "MAPF Total Time vs VRP Time Limit",
               "e19_mapf_total_time_s_vs_timelimit",
               lb_key="mapf_total_time_lb_s")
    _line_plot("mapf_objective_s",
               "MAPF objective (s)",
               "MAPF Objective vs VRP Time Limit",
               "e19_mapf_objective_s_vs_timelimit",
               lb_key="mapf_objective_time_lb_s")

    # ── Stage timing (stacked bar: VRP solve + MAPF plan) ──────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3.2))
    vrp_m, _ = _per_tl(ok, time_limits, "t_vrp_solve")
    mapf_m, _ = _per_tl(ok, time_limits, "t_mapf")
    stacked_bar(ax, labels,
                {"VRP solve": list(vrp_m), "MAPF plan": list(mapf_m)},
                ylabel="Time (s)",
                title="Stage Timing by VRP Time Limit")
    ax.set_xlabel("VRP time_limit (s)")
    save_figure(fig, os.path.join(fig_dir, "e19_stage_timing"))

    logger.info("E19 figures saved to %s", fig_dir)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="E19: VRP Time-Limit Sweep")
    p.add_argument("--time_limits", type=int, nargs="+", default=E19_TIME_LIMITS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_3)
    p.add_argument("--output_dir",
                   default=os.path.join(RESULTS_DIR, "e19_vrp_time_limit_sweep"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--plots_only", action="store_true")
    p.add_argument("--compute_lbs_only", action="store_true",
                   help="Skip full VRP+MAPF pipeline; recompute LBs per "
                        "unique seed from existing main JSONs and write "
                        "sidecar JSONs into raw_lb/ (one per (time_limit, "
                        "seed) row — analytical LBs are instance-level and "
                        "so identical across time_limits).")
    p.add_argument("--include_cuopt_bound", action="store_true",
                   help="In --compute_lbs_only mode, also run a short cuOpt "
                        "solve per seed to extract the MIP dual bound. "
                        "Expensive; off by default.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    raw_dir = os.path.join(args.output_dir, "raw")
    raw_lb_dir = os.path.join(args.output_dir, "raw_lb")
    os.makedirs(raw_dir, exist_ok=True)
    all_results: list[dict] = []

    if args.compute_lbs_only:
        if not os.path.isdir(raw_dir):
            logger.error("No raw/ directory at %s; nothing to augment.", raw_dir)
            return
        os.makedirs(raw_lb_dir, exist_ok=True)
        og, sampler, bmin, bmax, _model_cfg = _build_setup()

        # Load all existing main JSONs; group them by seed so we only
        # rebuild each instance (and run the optional cuOpt bound solve)
        # once per seed.
        stems = sorted(
            f[:-5] for f in os.listdir(raw_dir) if f.endswith(".json")
        )
        by_seed: dict[int, list[str]] = {}
        for stem in stems:
            main = load_run_result(os.path.join(raw_dir, stem))
            all_results.append(main)
            by_seed.setdefault(int(main["seed"]), []).append(stem)

        logger.info("Recomputing LBs for %d unique seeds (from %d rows)%s ...",
                    len(by_seed), len(stems),
                    " including cuOpt bound" if args.include_cuopt_bound else "")

        for seed, stem_list in by_seed.items():
            try:
                instance = _sample_instance(seed, og, sampler, bmin, bmax)
                lb = recompute_lbs(
                    instance["dist_matrix"], instance["home_indices"],
                    instance["K"], E19_N_WAYPOINTS, _ALPHA,
                    include_mapf=True,
                    include_cuopt=args.include_cuopt_bound,
                )
            except Exception as e:
                logger.error("LB recompute failed for seed=%d: %s", seed, e)
                continue
            # One sidecar JSON per (time_limit, seed) so plotting works
            # row-wise; the LB is identical across time_limits for a seed.
            for stem in stem_list:
                save_lb_json(os.path.join(raw_lb_dir, stem), lb)
            free_gpu_memory()
            logger.info("  seed=%d: LB written for %d rows", seed, len(stem_list))

        lb_by_stem = load_raw_lb_dir(raw_lb_dir, raw_dir)
        if all_results:
            generate_plots(all_results, args.output_dir, lb_by_stem=lb_by_stem)
        return

    if not args.plots_only:
        os.makedirs(raw_lb_dir, exist_ok=True)
        og, sampler, bmin, bmax, _model_cfg = _build_setup()

        for seed in args.seeds:
            logger.info("=" * 60)
            logger.info("Seed %d: sampling frozen instance (%d waypoints, K=%d)",
                        seed, E19_N_WAYPOINTS, E19_FLEET_SIZE)
            try:
                instance = _sample_instance(seed, og, sampler, bmin, bmax)
            except Exception as e:
                if args.resume and is_oom(e):
                    free_gpu_memory()
                    logger.error(
                        "Instance setup OOM (seed=%d) under --resume; exiting 1 "
                        "so a restart can reclaim GPU memory.", seed,
                    )
                    raise SystemExit(1)
                logger.error("Instance setup FAILED for seed %d: %s", seed, e,
                             exc_info=True)
                continue

            for tl in args.time_limits:
                stem = f"time_limit={tl}_seed={seed}"
                rpath = os.path.join(raw_dir, stem)
                lb_path = os.path.join(raw_lb_dir, stem)
                if args.resume and os.path.exists(rpath + ".json"):
                    logger.info("[SKIP] time_limit=%d seed=%d", tl, seed)
                    all_results.append(load_run_result(rpath))
                    continue

                logger.info("[RUN] time_limit=%d seed=%d", tl, seed)
                try:
                    result, lb = run_single(tl, seed, instance, og)
                    all_results.append(result)
                    save_run_result(result, rpath)
                    if lb is not None:
                        save_lb_json(lb_path, lb)
                    logger.info(
                        "  vrp_obj=%.1fm mapf_mks=%.2fs t_vrp=%.1fs t_mapf=%.1fs",
                        result.get("vrp_objective_m", float("nan")),
                        result.get("mapf_makespan_s", float("nan")),
                        result.get("t_vrp_solve", 0.0),
                        result.get("t_mapf", 0.0),
                    )
                except Exception as e:
                    handle_row_exception(
                        e,
                        f"time_limit={tl} seed={seed}",
                        resume=args.resume,
                    )
                finally:
                    free_gpu_memory()
    else:
        for fname in sorted(os.listdir(raw_dir)):
            if fname.endswith(".json"):
                all_results.append(load_run_result(
                    os.path.join(raw_dir, fname.replace(".json", ""))))

    lb_by_stem = load_raw_lb_dir(raw_lb_dir, raw_dir)
    if all_results:
        generate_plots(all_results, args.output_dir, lb_by_stem=lb_by_stem)

        # Summary table
        ok = [r for r in all_results if r.get("vrp_status") in ("success", "time_limit")]
        if ok:
            time_limits = sorted(set(r["time_limit"] for r in ok))
            logger.info("\n%s\nE19 SUMMARY\n%s", "=" * 80, "=" * 80)
            logger.info("%-10s %12s %12s %12s %12s",
                        "TimeLim", "VRP obj (m)", "VRP mks (m)",
                        "MAPF mks (s)", "MAPF tot (s)")
            logger.info("-" * 65)
            for tl in time_limits:
                rows = [r for r in ok if r["time_limit"] == tl]
                logger.info(
                    "%-10d %12.1f %12.1f %12.2f %12.2f",
                    tl,
                    np.mean([r["vrp_objective_m"] for r in rows]),
                    np.mean([r["vrp_makespan_m"] for r in rows]),
                    float(np.nanmean([r["mapf_makespan_s"] for r in rows])),
                    float(np.nanmean([r["mapf_total_time_s"] for r in rows])),
                )


if __name__ == "__main__":
    main()
