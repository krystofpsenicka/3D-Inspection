#!/usr/bin/env python3
"""E15: Cross-Model Generalization

Runs the full 8-stage inspection pipeline on Duke of Lancaster + TOSCA_ALL to
evaluate how well the pipeline generalizes across shape complexity.

Pipeline stages timed:
  1. Mesh loading
  2. Surface sampling
  3. Occupancy grid
  4. Viewpoint sampling  (weighted_curvature — chosen based on e01 results)
  5. Visibility          (GPU raycast — ground truth, from e03 results)
  6. Set cover           (LazyGreedy CPU — fastest solver, from e04 results)
  7. VRP routing         (cuOpt, K=5 robots — from e08 results)
  8. MAPF trajectory     (resolution 0.5 m — from e10 results)

Implementation choice rationale (printed in summary and figure annotations):
  - Sampler: weighted_curvature achieves the best coverage/viewpoints ratio (e01)
  - Visibility: GPU raycast is exact and fastest for N≤5K candidates (e03)
  - Set cover: LazyGreedy (CPU) beats LazyGreedy (GPU) for N~1500 due to heap (e04)
  - VRP: cuOpt achieves near-optimal within time limit (e08)
  - MAPF: 0.5 m voxel resolution balances path quality vs planning cost (e10)

Usage:
    conda run -n isaaclab python -m experiments.e10_cross_model
    conda run -n isaaclab python -m experiments.e10_cross_model --plots_only
    conda run -n isaaclab python -m experiments.e10_cross_model --models duke_of_lancaster wolf0
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.common.config import (
    ModelConfig, SEEDS_3, TOSCA_ALL, RESULTS_DIR,
)
from experiments.common.persistence import save_run_result, load_run_result
from experiments.common.lower_bounds import joint_problem_lb, JOINT_LB_FIELDS
from experiments.common.lb_sidecar import (
    compute_all_lbs, save_lb_json, load_raw_lb_dir, ALL_LB_FIELDS,
)
from experiments.common.plotting import (
    setup_thesis_style, save_figure, grouped_bar, stacked_bar,
    THESIS_COL, DOUBLE_COL, CATEGORICAL_COLORS,
)

# ── Runtime imports (need isaaclab/CUDA). Plot-only mode skips these. ──────
_RUNTIME_IMPORT_ERROR: ImportError | None = None
try:
    import cupy as cp
    from experiments.common.runner import (
        set_seed, timed, free_gpu_memory, handle_row_exception, is_oom,
    )
    from experiments.common.pipeline_setup import (
        PipelineContext, DegenerateNormalsError,
    )
    from experiments.common.sampling_dispatch import sample_strategy
    from visibility.set_cover import LazyGreedySetCover
    _RUNTIME_AVAILABLE = True
except ImportError as _e:
    cp = None
    set_seed = timed = free_gpu_memory = handle_row_exception = is_oom = None
    PipelineContext = DegenerateNormalsError = None
    sample_strategy = None
    LazyGreedySetCover = None
    _RUNTIME_AVAILABLE = False
    _RUNTIME_IMPORT_ERROR = _e

logger = logging.getLogger(__name__)

ALL_MODELS = ["duke_of_lancaster"] + TOSCA_ALL
TARGET_COVERAGE = 0.95
FLEET_SIZE = 5   # robots for VRP/MAPF
VRP_ALPHA = 0.5  # blend β in eq. (1.1): 0.5 · makespan + 0.5 · total_cost

# Fields persisted as an LB sidecar JSON next to each main result.
LB_SIDECAR_FIELDS = tuple(JOINT_LB_FIELDS) + tuple(ALL_LB_FIELDS)

_IMPLEMENTATION_CHOICES = (
    "Sampler: weighted_curvature (SDF² + curvature bias, e01)  |  "
    "Visibility: GPU raycast (exact, e03)  |  "
    "Set cover: LazyGreedy CPU (O(log N) heap, fastest, e04)  |  "
    "VRP: cuOpt (near-optimal, e08)  |  "
    "MAPF: 0.5 m resolution (e10)"
)

# Graceful import of VRP/MAPF stack
try:
    from VRP.core.distance_matrix import compute_distance_matrix
    from VRP.vrp.vrp_solver import solve_vrp
    from VRP.core.types import VRPBackend, ExecutionResult
    from VRP.mapf.mapf_planner import MultiAgentPathPlanner
    from VRP.core.geometry import compute_start_grid
    from VRP.core.constants import AUV_CRUISE_SPEED, SPACE_TIME_DWELL_S
    _VRP_AVAILABLE = True
except ImportError as _vrp_err:
    logger.warning("VRP/MAPF stack not available (%s) — stages 7/8 will be skipped.", _vrp_err)
    _VRP_AVAILABLE = False
    AUV_CRUISE_SPEED = 2.0
    SPACE_TIME_DWELL_S = 2.0


# ═══════════════════════════════════════════════════════════════════════════
# Single run logic
# ═══════════════════════════════════════════════════════════════════════════

def run_single(ctx: PipelineContext, model_cfg: ModelConfig, seed: int) -> dict:
    """Run full 8-stage pipeline on one model with one seed."""
    result: dict = {
        "model": model_cfg.name,
        "seed": seed,
        # Stage timings
        "t_mesh": 0.0, "t_surface": 0.0, "t_og": 0.0,
        "t_sample": 0.0, "t_vis": 0.0, "t_opt": 0.0,
        "t_vrp": 0.0, "t_mapf": 0.0, "t_joint_lb": 0.0,
        # Metrics
        "mesh_vertices": 0, "mesh_faces": 0,
        "num_viewpoints": 0, "coverage": 0.0, "redundancy": 0.0,
        "vrp_makespan": float("nan"), "vrp_total_cost": float("nan"),
        "vrp_status": "skipped",
        "mapf_steps": 0, "mapf_collisions": 0,
        "mapf_makespan_s": float("nan"), "mapf_total_time_s": float("nan"),
    }
    # Joint-problem + stage LB fields (all 0.0 if not filled in).
    for k in JOINT_LB_FIELDS:
        result[k] = 0 if k.endswith("poses_lb") else 0.0
    for k in ALL_LB_FIELDS:
        result[k] = 0.0

    # 1. Mesh loading
    with timed() as t_mesh:
        tm, _ = ctx.load_mesh()
    result["t_mesh"] = t_mesh.elapsed
    result["mesh_vertices"] = len(tm.vertices)
    result["mesh_faces"] = len(tm.faces)

    # 2. Surface sampling
    with timed() as t_surface:
        target_points, normals = ctx.sample_surface()
    result["t_surface"] = t_surface.elapsed

    # 3. Occupancy grid
    with timed() as t_og:
        og = ctx.build_sampling_og()
    result["t_og"] = t_og.elapsed

    # 4. Viewpoint sampling (weighted_curvature)
    set_seed(seed)
    vis_query = ctx.build_visibility_query("raycast")
    with timed() as t_sample:
        pos_gpu, rot_gpu, _, _, _, _ = sample_strategy(
            ctx, "weighted_curvature", model_cfg.num_candidates,
            target_points, normals, vis_query, model_cfg,
        )
    result["t_sample"] = t_sample.elapsed

    # 5. Visibility (GPU raycast)
    with timed() as t_vis:
        V, _ = vis_query.compute_visibility_batch(pos_gpu, rot_gpu)
    result["t_vis"] = t_vis.elapsed

    # 6. Set cover (LazyGreedy CPU)
    V_np = cp.asnumpy(V)
    pos_np = cp.asnumpy(pos_gpu)
    rot_np = cp.asnumpy(rot_gpu)
    num_points = int(len(target_points))

    # Joint-problem lower bound — computed here (after visibility) so
    # we have V_np for Component 1. Depot layout matches the VRP stage.
    if _VRP_AVAILABLE:
        try:
            bounds_min, bounds_max = ctx.mesh_bounds
            _robot_starts_lb = compute_start_grid(
                FLEET_SIZE, bounds_min, bounds_max)
            _home_lb = np.array(
                [[float(x[0]), float(x[1]), float(x[2])] for x in _robot_starts_lb],
                dtype=np.float32)
            _tp_np = (cp.asnumpy(target_points) if hasattr(target_points, "get")
                      else np.asarray(target_points))
            with timed() as t_joint_lb:
                joint_lb = joint_problem_lb(
                    V_np, _tp_np, _home_lb,
                    alpha_coverage=TARGET_COVERAGE,
                    beta_blend=VRP_ALPHA,
                    frustum_far=float(model_cfg.frustum.far),
                    cruise_speed=AUV_CRUISE_SPEED,
                    dwell_s=SPACE_TIME_DWELL_S,
                    fleet_size=FLEET_SIZE,
                )
            result["t_joint_lb"] = t_joint_lb.elapsed
            for k in JOINT_LB_FIELDS:
                result[k] = joint_lb[k]
        except Exception as e:
            if is_oom(e):
                raise
            logger.warning("Joint LB failed: %s", e, exc_info=True)

    with timed() as t_opt:
        optimizer = LazyGreedySetCover(num_points, pos_np, rot_np, V_np)
        opt_result = optimizer.optimize(
            target_coverage=TARGET_COVERAGE, max_viewpoints=1000)
    result["t_opt"] = t_opt.elapsed

    result["num_viewpoints"] = opt_result.num_viewpoints
    result["coverage"] = float(opt_result.total_coverage)
    result["redundancy"] = float(opt_result.redundancy)

    # Selected viewpoint positions / rotations
    sel_idx = opt_result.selected_indices
    if hasattr(sel_idx, "get"):
        sel_idx = sel_idx.get()
    sel_idx = np.asarray(sel_idx)
    insp_positions = pos_np[sel_idx]   # (N_vp, 3)
    insp_rotmats = rot_np[sel_idx]     # (N_vp, 3, 3)

    if not _VRP_AVAILABLE or opt_result.num_viewpoints == 0:
        result["total_time"] = sum(
            result[k] for k in ("t_mesh", "t_surface", "t_og",
                                 "t_sample", "t_vis", "t_opt"))
        return result

    # 7. VRP routing (cuOpt, K=FLEET_SIZE robots)
    try:
        bounds_min, bounds_max = ctx.mesh_bounds
        robot_starts = compute_start_grid(FLEET_SIZE, bounds_min, bounds_max)
        home_positions = np.array(
            [[float(xyz[0]), float(xyz[1]), float(xyz[2])] for xyz in robot_starts],
            dtype=np.float32)
        home_rotmats = np.tile(np.eye(3, dtype=np.float32), (FLEET_SIZE, 1, 1))
        all_positions = np.vstack([home_positions, insp_positions])
        all_rotmats = np.concatenate([home_rotmats, insp_rotmats])
        home_indices = list(range(FLEET_SIZE))

        with timed() as t_vrp:
            dist_matrix = compute_distance_matrix(og, cp.asarray(all_positions))
            vrp_result = solve_vrp(
                dist_matrix=dist_matrix,
                num_vehicles=FLEET_SIZE,
                depots=home_indices,
                backend=VRPBackend.CUOPT,
                time_limit=60,
                alpha=VRP_ALPHA,
            )
        result["t_vrp"] = t_vrp.elapsed
        result["vrp_status"] = vrp_result.status
        result["vrp_total_cost"] = float(vrp_result.total_cost)

        # Stage LBs for VRP / MAPF in meters/seconds, using the same
        # dist_matrix and the cuOpt dual bound.
        try:
            stage_lb = compute_all_lbs(
                dist_matrix, home_indices, FLEET_SIZE,
                opt_result.num_viewpoints, VRP_ALPHA,
                include_mapf=True,
                vrp_best_bound_m=vrp_result.best_bound,
                vrp_objective_value_m=vrp_result.objective_value,
            )
            for k in ALL_LB_FIELDS:
                result[k] = float(stage_lb.get(k, 0.0))
        except Exception as e:
            logger.warning("Stage LB (compute_all_lbs) failed: %s", e)

        if any(vrp_result.routes):
            from VRP.vrp._helpers import per_vehicle_costs
            rc = np.array(per_vehicle_costs(
                vrp_result.routes, dist_matrix, home_indices))
            result["vrp_makespan"] = float(rc.max())

            # 8. MAPF trajectory planning
            routes = [
                [home_indices[i]] + list(r) + [home_indices[i]]
                for i, r in enumerate(vrp_result.routes)
            ]
            start_positions = [
                np.array(xyz, dtype=np.float32) for xyz in robot_starts
            ]
            wp_pos_gpu = cp.asarray(all_positions, dtype=cp.float32)
            wp_rot_gpu = cp.asarray(all_rotmats, dtype=cp.float32)

            with timed() as t_mapf:
                executor = MultiAgentPathPlanner(
                    start_positions=start_positions, og=og)
                exec_result = executor.execute(
                    routes=routes,
                    waypoint_positions=wp_pos_gpu,
                    waypoint_rotmats=wp_rot_gpu,
                    home_indices=set(home_indices),
                    dist_matrix=dist_matrix,
                    alpha=VRP_ALPHA,
                )
            result["t_mapf"] = t_mapf.elapsed

            if exec_result.all_traj_positions:
                result["mapf_steps"] = max(
                    len(t) for t in exec_result.all_traj_positions)
            result["mapf_collisions"] = sum(exec_result.fail_counts)
            result["mapf_makespan_s"] = float(exec_result.actual_makespan)
            if exec_result.actual_per_vehicle_times:
                result["mapf_total_time_s"] = float(
                    sum(exec_result.actual_per_vehicle_times))
        else:
            result["vrp_status"] = "empty_routes"

    except Exception as e:
        if is_oom(e):
            # Propagate so the outer per-row handler can exit under --resume.
            raise
        logger.error("VRP/MAPF stage failed: %s", e, exc_info=True)
        result["vrp_status"] = f"error: {e}"

    result["total_time"] = sum(
        result[k] for k in ("t_mesh", "t_surface", "t_og",
                             "t_sample", "t_vis", "t_opt",
                             "t_vrp", "t_mapf"))
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Plot generation
# ═══════════════════════════════════════════════════════════════════════════

_STAGE_NAMES = ["Mesh", "Surface", "OG", "Sampling", "Visibility",
                "Set Cover", "VRP", "MAPF"]
_STAGE_KEYS = ["t_mesh", "t_surface", "t_og", "t_sample", "t_vis",
               "t_opt", "t_vrp", "t_mapf"]


def generate_plots(results: list[dict], output_dir: str):
    """Generate all E15 figures."""
    setup_thesis_style()
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    if not results:
        logger.warning("No results for plotting")
        return

    models = sorted(
        set(r["model"] for r in results),
        key=lambda m: np.mean([r["mesh_faces"] for r in results if r["model"] == m]),
    )

    def _mm(model, metric):
        vals = []
        for r in results:
            if r.get("model") != model or metric not in r:
                continue
            v = r[metric]
            if isinstance(v, float) and np.isnan(v):
                continue
            vals.append(v)
        return float(np.mean(vals)) if vals else float("nan")

    def _ms(model, metric):
        vals = []
        for r in results:
            if r.get("model") != model or metric not in r:
                continue
            v = r[metric]
            if isinstance(v, float) and np.isnan(v):
                continue
            vals.append(v)
        return float(np.std(vals)) if vals else 0.0

    subtitle = f"\n{_IMPLEMENTATION_CHOICES}"

    # ── Fig 1: Viewpoints by model (with N_poses LB overlay) ──────────
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4))
    vp_m = [_mm(m, "num_viewpoints") for m in models]
    vp_s = [_ms(m, "num_viewpoints") for m in models]
    x = np.arange(len(models))
    bars = ax.bar(x, vp_m, 0.6, yerr=vp_s, capsize=3,
                  color=CATEGORICAL_COLORS[0], alpha=0.85,
                  label="Selected")
    ax.bar_label(bars, fmt="%.0f", fontsize=7, padding=2)
    match_lb = [_mm(m, "joint_n_poses_lb") for m in models]
    info_lb = [_mm(m, "joint_info_n_poses_lb") for m in models]
    for xi, (mlb, ilb) in enumerate(zip(match_lb, info_lb)):
        if mlb > 0:
            ax.hlines(mlb, xi - 0.3, xi + 0.3, colors="black",
                      linestyles=":", linewidth=1.2,
                      label="Matching LB" if xi == 0 else None)
        if ilb > 0:
            ax.hlines(ilb, xi - 0.3, xi + 0.3, colors="gray",
                      linestyles="--", linewidth=1.0,
                      label="Info-theoretic LB" if xi == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha="right")
    ax.set_ylabel("Selected viewpoints")
    ax.set_title(f"Viewpoints by Model (95% coverage){subtitle}", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    save_figure(fig, os.path.join(fig_dir, "e10_viewpoints"))

    # ── Fig 2: Coverage by model ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4))
    cov_m = [_mm(m, "coverage") * 100 for m in models]
    cov_s = [_ms(m, "coverage") * 100 for m in models]
    bars = ax.bar(x, cov_m, 0.6, yerr=cov_s, capsize=3,
                  color=CATEGORICAL_COLORS[1], alpha=0.85)
    ax.bar_label(bars, fmt="%.1f%%", fontsize=7, padding=2)
    ax.axhline(95, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha="right")
    ax.set_ylabel("Coverage (%)")
    ax.set_title(f"Coverage by Model{subtitle}", fontsize=9)
    save_figure(fig, os.path.join(fig_dir, "e10_coverage"))

    # ── Fig 3: Full 8-stage timing breakdown (stacked bar) ───────────────
    # Include only stages that have non-zero values (VRP/MAPF may be skipped)
    active_stages = []
    active_keys = []
    for name, key in zip(_STAGE_NAMES, _STAGE_KEYS):
        if any(r.get(key, 0) > 0 for r in results):
            active_stages.append(name)
            active_keys.append(key)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4.5))
    stage_data = {
        sname: [_mm(m, skey) for m in models]
        for sname, skey in zip(active_stages, active_keys)
    }
    stacked_bar(ax, models, stage_data,
                ylabel="Time (s)",
                title=f"Pipeline Stage Timing by Model (K={FLEET_SIZE} robots){subtitle}")
    ax.tick_params(axis="x", rotation=35)
    save_figure(fig, os.path.join(fig_dir, "e10_timing_breakdown"))

    # Same data on a log y-axis so the small stages (sampling, visibility,
    # set cover) remain readable next to the dominant VRP/MAPF stages.
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4.5))
    width = 0.8 / max(1, len(active_stages))
    xv = np.arange(len(models))
    for i, (sname, skey) in enumerate(zip(active_stages, active_keys)):
        vals = [_mm(m, skey) for m in models]
        # Replace zero / NaN with NaN so log axis doesn't blow up.
        vals = [v if (v is not None and v > 0 and not np.isnan(v)) else float("nan")
                for v in vals]
        ax.bar(xv + (i - (len(active_stages) - 1) / 2) * width, vals,
               width, label=sname,
               color=CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)],
               alpha=0.85)
    ax.set_yscale("log")
    ax.set_xticks(xv)
    ax.set_xticklabels(models, rotation=35, ha="right")
    ax.set_ylabel("Time (s, log scale)")
    ax.set_title(
        f"Pipeline Stage Timing — log scale (K={FLEET_SIZE} robots){subtitle}")
    ax.legend(fontsize=7, ncol=2)
    save_figure(fig, os.path.join(fig_dir, "e10_timing_breakdown_log"))

    # ── Fig 4: VRP makespan by model (with stage LB band) ──────────────
    def _vrp_bar_with_lb(metric: str, lb_field: str, title: str,
                         ylabel: str, stem: str, color_idx: int):
        vrp_rows = [r for r in results
                    if not np.isnan(r.get(metric, float("nan")))]
        if not vrp_rows:
            return
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4))
        vm = [_mm(m, metric) for m in models]
        vs = [_ms(m, metric) for m in models]
        valid_models = [m for m, v in zip(models, vm) if not np.isnan(v)]
        valid_m = [v for v in vm if not np.isnan(v)]
        valid_s = [s for v, s in zip(vm, vs) if not np.isnan(v)]
        if not valid_models:
            return
        xv = np.arange(len(valid_models))
        ax.bar(xv, valid_m, 0.6, yerr=valid_s, capsize=3,
               color=CATEGORICAL_COLORS[color_idx], alpha=0.85,
               label="Observed")
        # LB overlay (prefer cuOpt dual bound when present).
        lb_vals = []
        for m in valid_models:
            cu = _mm(m, "vrp_objective_best_bound_m")
            an = _mm(m, lb_field)
            lb_vals.append(max(cu, an) if cu > 0 else an)
        if any(v > 0 for v in lb_vals):
            ax.scatter(xv, lb_vals, marker="_", s=200, color="black",
                       linewidths=1.5, zorder=5, label="LB")
            for xi, (obs, lb) in enumerate(zip(valid_m, lb_vals)):
                if lb > 0 and obs > 0:
                    gap = (obs - lb) / lb * 100.0
                    ax.text(xi, obs, f" +{gap:.0f}%",
                            fontsize=6, ha="center", va="bottom")
        ax.set_xticks(xv)
        ax.set_xticklabels(valid_models, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
        save_figure(fig, os.path.join(fig_dir, stem))

    _vrp_bar_with_lb(
        "vrp_makespan", "vrp_makespan_lb_m",
        f"VRP Makespan by Model (K={FLEET_SIZE}, α={VRP_ALPHA})",
        "VRP makespan (m)", "e10_vrp_makespan", 3,
    )
    _vrp_bar_with_lb(
        "vrp_total_cost", "vrp_total_cost_lb_m",
        f"VRP Total Cost by Model (K={FLEET_SIZE}, α={VRP_ALPHA})",
        "VRP total cost (m)", "e10_vrp_total_cost", 2,
    )

    # ── Fig 5: Viewpoints vs mesh complexity (scatter, per-mesh colors) ──
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4))
    for i, m in enumerate(models):
        mr = [r for r in results if r["model"] == m]
        if not mr:
            continue
        color = CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)]
        ax.scatter([r["mesh_faces"] for r in mr],
                   [r["num_viewpoints"] for r in mr],
                   color=color, s=40, alpha=0.85, label=m)
    ax.set_xlabel("Mesh faces")
    ax.set_ylabel("Selected viewpoints")
    ax.set_title("Viewpoints vs Mesh Complexity")
    ax.legend(fontsize=6, ncol=2, loc="best")
    save_figure(fig, os.path.join(fig_dir, "e10_scatter_faces_vs_vps"))

    # ── Fig 6: Joint-problem makespan gap (seconds) ────────────────────
    def _paired_gap_fig(value_fn, lb_field: str, title: str, ylabel: str,
                        stem: str):
        rows = [(m, [value_fn(r) for r in results if r["model"] == m])
                for m in models]
        rows = [(m, [v for v in vs if v is not None]) for m, vs in rows]
        rows = [(m, vs) for m, vs in rows if vs]
        if not rows:
            return
        lbs = []
        for m, _ in rows:
            lb = _mm(m, lb_field)
            lbs.append(lb if lb > 0 else float("nan"))
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4))
        xv = np.arange(len(rows))
        obs_mean = [float(np.mean(vs)) for _, vs in rows]
        obs_std = [float(np.std(vs)) for _, vs in rows]
        w = 0.38
        ax.bar(xv - w / 2, lbs, w, color=CATEGORICAL_COLORS[2], alpha=0.85,
               label="Joint LB")
        bars = ax.bar(xv + w / 2, obs_mean, w, yerr=obs_std, capsize=3,
                      color=CATEGORICAL_COLORS[3], alpha=0.85,
                      label="Observed")
        for xi, (obs, lb) in enumerate(zip(obs_mean, lbs)):
            if lb and not np.isnan(lb) and lb > 0:
                gap = (obs - lb) / lb * 100.0
                ax.text(xi + w / 2, obs, f" +{gap:.0f}%",
                        fontsize=6, ha="center", va="bottom")
        ax.set_xticks(xv)
        ax.set_xticklabels([m for m, _ in rows], rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
        save_figure(fig, os.path.join(fig_dir, stem))

    _paired_gap_fig(
        lambda r: r.get("mapf_makespan_s") if not np.isnan(
            r.get("mapf_makespan_s", float("nan"))) else None,
        "joint_makespan_lb_s",
        f"Joint Makespan Gap (K={FLEET_SIZE}, α={VRP_ALPHA})",
        "Makespan (s)",
        "e10_joint_makespan_gap",
    )
    _paired_gap_fig(
        _observed_objective_s,
        "joint_objective_lb_s",
        f"Joint Objective Gap (β={VRP_ALPHA})",
        f"β·makespan + (1-β)·total (s)",
        "e10_joint_objective_gap",
    )

    logger.info("E10 figures saved to %s", fig_dir)


# ═══════════════════════════════════════════════════════════════════════════
# Joint-LB gap helpers
# ═══════════════════════════════════════════════════════════════════════════

def _observed_objective_s(r: dict) -> float | None:
    """β · observed_makespan + (1-β) · observed_total_time, or None if
    MAPF did not run / times are missing."""
    mks = r.get("mapf_makespan_s", float("nan"))
    tot = r.get("mapf_total_time_s", float("nan"))
    if np.isnan(mks) or np.isnan(tot):
        return None
    return VRP_ALPHA * mks + (1.0 - VRP_ALPHA) * tot


def _objective_gap_pct(r: dict) -> float | None:
    """(observed − joint_objective_LB) / joint_objective_LB × 100, or None."""
    obs = _observed_objective_s(r)
    lb = r.get("joint_objective_lb_s", 0.0)
    if obs is None or lb is None or lb <= 0:
        return None
    return (obs - lb) / lb * 100.0


def _makespan_gap_pct(r: dict) -> float | None:
    mks = r.get("mapf_makespan_s", float("nan"))
    lb = r.get("joint_makespan_lb_s", 0.0)
    if np.isnan(mks) or lb <= 0:
        return None
    return (mks - lb) / lb * 100.0


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="E15: Cross-Model Generalization")
    p.add_argument("--models", nargs="+", default=ALL_MODELS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_3)
    p.add_argument("--output_dir",
                   default=os.path.join(RESULTS_DIR, "e10_cross_model"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--plots_only", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
    )

    if not args.plots_only and not _RUNTIME_AVAILABLE:
        raise SystemExit(
            f"Runtime imports unavailable ({_RUNTIME_IMPORT_ERROR}). "
            "Activate the isaaclab env or pass --plots_only."
        )

    logger.info("Implementation choices: %s", _IMPLEMENTATION_CHOICES)
    if not _VRP_AVAILABLE:
        logger.warning("VRP/MAPF stack not available — stages 7/8 will be skipped")

    raw_dir = os.path.join(args.output_dir, "raw")
    raw_lb_dir = os.path.join(args.output_dir, "raw_lb")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(raw_lb_dir, exist_ok=True)
    all_results: list[dict] = []

    if not args.plots_only:
        model_configs = []
        for name in args.models:
            if name == "duke_of_lancaster":
                model_configs.append(ModelConfig.duke_of_lancaster())
            else:
                try:
                    model_configs.append(ModelConfig.tosca(name))
                except FileNotFoundError:
                    logger.warning("Model %s not found — skipping", name)

        total = len(model_configs) * len(args.seeds)
        run_idx = 0

        for model_cfg in model_configs:
            logger.info("=" * 60)
            logger.info("Model: %s", model_cfg.name)
            try:
                ctx = PipelineContext(model_cfg)
                ctx.load_mesh()
                ctx.sample_surface(seed=42)
                ctx.build_sampling_og()
            except DegenerateNormalsError as e:
                logger.warning("Skipping %s: %s", model_cfg.name, e)
                continue

            for seed in args.seeds:
                run_idx += 1
                stem = f"model={model_cfg.name}_seed={seed}"
                rpath = os.path.join(raw_dir, stem)
                lb_stem = os.path.join(raw_lb_dir, stem)

                if args.resume and os.path.exists(rpath + ".json"):
                    logger.info("[%d/%d] SKIP %s seed=%d",
                                run_idx, total, model_cfg.name, seed)
                    all_results.append(load_run_result(rpath))
                    continue

                logger.info("[%d/%d] Running: %s seed=%d",
                            run_idx, total, model_cfg.name, seed)
                try:
                    result = run_single(ctx, model_cfg, seed)
                    all_results.append(result)
                    save_run_result(result, rpath)
                    save_lb_json(lb_stem,
                                 {k: result[k] for k in LB_SIDECAR_FIELDS
                                  if k in result})
                    _gap = _objective_gap_pct(result)
                    logger.info("  vps=%d cov=%.2f%% t=%.1fs vrp=%s "
                                "mks=%.1fs jmks_lb=%.1fs jobj_gap=%s",
                                result["num_viewpoints"],
                                result["coverage"] * 100,
                                result["total_time"],
                                result["vrp_status"],
                                result["mapf_makespan_s"]
                                if not np.isnan(result.get("mapf_makespan_s", float("nan")))
                                else float("nan"),
                                result.get("joint_makespan_lb_s", 0.0),
                                f"{_gap:.1f}%" if _gap is not None else "n/a")
                except Exception as e:
                    handle_row_exception(
                        e,
                        f"model={model_cfg.name} seed={seed}",
                        resume=args.resume,
                    )
                finally:
                    free_gpu_memory()
    else:
        for fname in sorted(os.listdir(raw_dir)):
            if fname.endswith(".json"):
                all_results.append(load_run_result(
                    os.path.join(raw_dir, fname.replace(".json", ""))))
        # Merge LB sidecars by stem so old main JSONs that predate the LB
        # fields still get the bound values for plotting.
        lb_map = load_raw_lb_dir(raw_lb_dir, raw_dir)
        for r in all_results:
            stem = f"model={r['model']}_seed={r['seed']}"
            r.update(lb_map.get(stem, {}))

    if all_results:
        generate_plots(all_results, args.output_dir)

        logger.info("\n%s\nE15 SUMMARY — %s\n%s",
                    "=" * 80, _IMPLEMENTATION_CHOICES, "=" * 80)
        logger.info("%-20s %7s %6s %6s %7s %9s %9s %8s %8s",
                    "Model", "Faces", "VPs", "Cov%",
                    "VP_LB", "Mks(s)", "Jmks_LB", "Jmks%", "Jobj%")
        logger.info("-" * 100)
        models = sorted(set(r["model"] for r in all_results))
        for m in models:
            mr = [r for r in all_results if r["model"] == m]
            if mr:
                def _m(k):
                    vals = []
                    for r in mr:
                        if k not in r:
                            continue
                        v = r[k]
                        if isinstance(v, float) and np.isnan(v):
                            continue
                        vals.append(v)
                    return float(np.mean(vals)) if vals else float("nan")
                mks_gaps = [_makespan_gap_pct(r) for r in mr]
                mks_gaps = [g for g in mks_gaps if g is not None]
                obj_gaps = [_objective_gap_pct(r) for r in mr]
                obj_gaps = [g for g in obj_gaps if g is not None]
                logger.info(
                    "%-20s %7.0f %6.1f %6.2f %7.1f %9.1f %9.1f %7s %7s",
                    m, _m("mesh_faces"), _m("num_viewpoints"),
                    _m("coverage") * 100,
                    _m("joint_n_poses_lb"),
                    _m("mapf_makespan_s"),
                    _m("joint_makespan_lb_s"),
                    f"{np.mean(mks_gaps):.0f}%" if mks_gaps else "n/a",
                    f"{np.mean(obj_gaps):.0f}%" if obj_gaps else "n/a",
                )


if __name__ == "__main__":
    main()
