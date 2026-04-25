#!/usr/bin/env python3
"""E17: Sampler Routing Impact

Tests whether the sampler strategy affects downstream VRP routing objectives
(makespan and total_cost).  Runs stages 1-7 of the pipeline (skips MAPF) with
varying sampler strategies and measures how viewpoint spatial distribution
propagates to routing cost.

Hypothesis: CMA-ES's travel_weight parameter penalises point-to-point distance
during sampling, producing viewpoint sets that are cheaper to route through.

Strategies: weighted, weighted_curvature, targeted_100 (spi=100, k=3),
            cmaes_100 (popsize=40, maxiter=40, k=3)
CMA-ES travel_weight sweep: per-model-group — Duke uses E00_C_TRAVEL_WEIGHTS_DUKE
            and TOSCA uses E00_C_TRAVEL_WEIGHTS_TOSCA (same as e00 Section 2B).
Models: Duke of Lancaster + TOSCA_REPRESENTATIVE (wolf0, cat0, david0).
Fixed: 5 robots, 1500 candidates, 0.95 target coverage. 3 seeds.

Usage:
    conda run -n isaaclab python -m experiments.e17_sampler_routing_impact
    conda run -n isaaclab python -m experiments.e17_sampler_routing_impact --plots_only
    conda run -n isaaclab python -m experiments.e17_sampler_routing_impact --resume
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import cupy as cp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.common.config import (
    ModelConfig, SEEDS_3, RESULTS_DIR, TOSCA_REPRESENTATIVE,
    E17_N_ROBOTS, E17_N_CANDIDATES,
    E00_C_TRAVEL_WEIGHTS_TOSCA, E00_C_TRAVEL_WEIGHTS_DUKE,
)

# Strategy set for e17 (thesis-final): weighted + weighted_curvature baselines,
# targeted_100 (iterative targeted at k=3 with spi=100), cmaes_100 (CMA-ES
# with k=3, popsize=40, maxiter=40 and a per-model-group travel_weight sweep).
_E17_STRATEGIES = ["weighted", "weighted_curvature", "targeted_100", "cmaes_100"]
_E17_MODELS = ["duke_of_lancaster"] + list(TOSCA_REPRESENTATIVE)


def _is_duke(model_name: str) -> bool:
    return model_name == "duke_of_lancaster"


def _cmaes_travel_weights(model_name: str) -> list[float]:
    """TW sweep list matching e00 Section 2B's per-model-group choice."""
    return (list(E00_C_TRAVEL_WEIGHTS_DUKE) if _is_duke(model_name)
            else list(E00_C_TRAVEL_WEIGHTS_TOSCA))


def _strategy_kwargs_e17(strategy: str, travel_weight: float | None) -> dict:
    """Per-strategy kwargs forwarded to sample_strategy() for e17."""
    if strategy == "targeted_100":
        return {"k_coverage": 3, "samples_per_iteration": 100}
    if strategy == "cmaes_100":
        return {
            "k_coverage": 3,
            "popsize": 40,
            "maxiter": 40,
            "travel_weight": travel_weight if travel_weight is not None else 0.0,
        }
    return {}
from experiments.common.runner import set_seed, timed, free_gpu_memory, handle_row_exception
from experiments.common.pipeline_setup import PipelineContext
from experiments.common.persistence import save_run_result, load_run_result
from experiments.common.sampling_dispatch import sample_strategy
from experiments.common.lb_sidecar import (
    compute_all_lbs, recompute_lbs, save_lb_json, load_raw_lb_dir,
)
from experiments.common.plotting import (
    setup_thesis_style, save_figure, stacked_bar,
    THESIS_COL, DOUBLE_COL, CATEGORICAL_COLORS,
)

# e17's solve_vrp currently uses the default alpha=1.0 (pure makespan).
# Centralised so the LB sidecar uses the same value.
_E17_ALPHA = 1.0

from visibility.set_cover import LazyGreedySetCover

from VRP.core.distance_matrix import compute_distance_matrix
from VRP.vrp.vrp_solver import solve_vrp
from VRP.core.types import VRPBackend, VRPResult
from VRP.core.geometry import compute_start_grid
from VRP.vrp._helpers import per_vehicle_costs

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

_GROUP_COLORS = {
    "weighted": CATEGORICAL_COLORS[0],
    "weighted_curvature": CATEGORICAL_COLORS[1],
    "targeted": CATEGORICAL_COLORS[2],
    "cmaes": CATEGORICAL_COLORS[3],
}


def _bar_color(strategy: str) -> str:
    if strategy == "weighted":
        return _GROUP_COLORS["weighted"]
    if strategy == "weighted_curvature":
        return _GROUP_COLORS["weighted_curvature"]
    if strategy.startswith("targeted_"):
        return _GROUP_COLORS["targeted"]
    if strategy.startswith("cmaes_"):
        return _GROUP_COLORS["cmaes"]
    return "grey"


def _display_label(strategy: str, travel_weight) -> str:
    if travel_weight is not None:
        return f"{strategy}\ntw={travel_weight}"
    return strategy.replace("weighted_curvature", "w_curv")


def _build_run_configs(strategies, cmaes_travel_weights):
    """Expand ``strategies`` into (strategy, travel_weight) tuples.

    For CMA-ES strategies, one entry per travel-weight value. For other
    strategies, ``travel_weight`` is ``None``.
    """
    configs = []
    for s in strategies:
        if s.startswith("cmaes_"):
            for tw in cmaes_travel_weights:
                configs.append((s, tw))
        else:
            configs.append((s, None))
    return configs


def _result_path(raw_dir, model_name, strategy, travel_weight, seed):
    tw_str = f"tw={travel_weight}" if travel_weight is not None else "tw=none"
    return os.path.join(
        raw_dir,
        f"model={model_name}_strategy={strategy}_{tw_str}_seed={seed}",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Single run logic
# ═══════════════════════════════════════════════════════════════════════════

def _rebuild_instance_for_lb(
    ctx: PipelineContext, strategy: str, travel_weight, seed: int,
    target_coverage: float,
) -> tuple:
    """Re-run sampling + visibility + set-cover + distance-matrix construction
    for an existing (model, strategy, travel_weight, seed) row. Mirrors
    stages 1-6 of ``run_single`` without running VRP. Returns
    ``(K, home_indices, dist_matrix, num_viewpoints)``.

    Because the sampling output depends on the full seeded sequence of calls,
    this is the cheapest way to reconstruct the exact distance matrix the
    original solve saw. The duplicated logic with ``run_single`` is
    deliberate — changing ``run_single`` risks behaviour drift for the main
    experiment.
    """
    set_seed(seed)
    model_cfg = ctx.model
    target_points, normals = ctx.sample_surface()
    vis_query = ctx.build_visibility_query("raycast")

    pos_gpu, rot_gpu, *_ = sample_strategy(
        ctx, strategy, E17_N_CANDIDATES,
        target_points, normals, vis_query, model_cfg,
        **_strategy_kwargs_e17(strategy, travel_weight),
    )
    V, _ = vis_query.compute_visibility_batch(pos_gpu, rot_gpu)
    V_np = cp.asnumpy(V)
    pos_np = cp.asnumpy(pos_gpu)
    rot_np = cp.asnumpy(rot_gpu)
    optimizer = LazyGreedySetCover(len(target_points), pos_np, rot_np, V_np)
    opt_result = optimizer.optimize(
        target_coverage=target_coverage, max_viewpoints=1000,
    )

    K = E17_N_ROBOTS
    _, o3d_mesh = ctx.load_mesh()
    bmin, bmax = ctx.mesh_bounds
    robot_xyzs = compute_start_grid(K, bmin, bmax)
    home_pos = np.array(
        [[float(x[0]), float(x[1]), float(x[2])] for x in robot_xyzs],
        dtype=np.float32,
    )
    vp_pos_np = opt_result.positions.get()
    all_pos = np.vstack([home_pos, vp_pos_np])
    home_indices = list(range(K))

    import trimesh
    from shared.grid_builder_utils import compute_grid_bounds, voxelize_mesh
    from shared.grid_utils import inflate_grid
    from visibility.sampling.utils.sampling_occupancy_grid import SamplingOccupancyGrid
    _verts = np.asarray(o3d_mesh.vertices)
    _tm = trimesh.Trimesh(vertices=_verts, faces=np.asarray(o3d_mesh.triangles))
    _padding = model_cfg.frustum.far + 2 * model_cfg.collision_radius
    _vrp_origin, _vrp_shape = compute_grid_bounds(
        _verts.min(axis=0), _verts.max(axis=0), _padding, 0.20,
    )
    _raw_vrp, _ = voxelize_mesh(_tm, _vrp_shape, _vrp_origin, 0.20,
                                 fill_interior=False)
    og_vrp = SamplingOccupancyGrid(
        grid=inflate_grid(_raw_vrp, 1),
        origin=_vrp_origin,
        resolution=0.20,
        raw_grid=_raw_vrp,
        filled_raw_grid=_raw_vrp,
    )
    dist_matrix = compute_distance_matrix(og_vrp, cp.asarray(all_pos))
    return K, home_indices, dist_matrix, opt_result.num_viewpoints


def run_single(ctx: PipelineContext, strategy: str, travel_weight,
               seed: int, target_coverage: float = 0.95) -> dict:
    """Run stages 1-7 for one (strategy, travel_weight, seed) combo."""
    set_seed(seed)
    model_cfg = ctx.model
    target_points, normals = ctx.sample_surface()
    vis_query = ctx.build_visibility_query("raycast")

    # Stage 4: Sampling — kwargs depend on the strategy (see _strategy_kwargs_e17)
    with timed() as t_sample:
        pos_gpu, rot_gpu, n_base, n_iter, base_name, n_ws_fb = sample_strategy(
            ctx, strategy, E17_N_CANDIDATES,
            target_points, normals, vis_query, model_cfg,
            **_strategy_kwargs_e17(strategy, travel_weight),
        )

    # Stage 5: Visibility
    with timed() as t_vis:
        V, _ = vis_query.compute_visibility_batch(pos_gpu, rot_gpu)

    # Stage 6: Set cover
    with timed() as t_opt:
        V_np = cp.asnumpy(V)
        pos_np = cp.asnumpy(pos_gpu)
        rot_np = cp.asnumpy(rot_gpu)
        optimizer = LazyGreedySetCover(len(target_points), pos_np, rot_np, V_np)
        opt_result = optimizer.optimize(
            target_coverage=target_coverage, max_viewpoints=1000,
        )

    num_viewpoints = opt_result.num_viewpoints
    coverage = float(opt_result.total_coverage)
    redundancy = float(opt_result.redundancy)

    # Stage 7: VRP
    with timed() as t_vrp:
        K = E17_N_ROBOTS
        _, o3d_mesh = ctx.load_mesh()
        bmin, bmax = ctx.mesh_bounds

        robot_xyzs = compute_start_grid(K, bmin, bmax)
        home_pos = np.array(
            [[float(x[0]), float(x[1]), float(x[2])] for x in robot_xyzs],
            dtype=np.float32,
        )
        home_rot = np.tile(np.eye(3, dtype=np.float32), (K, 1, 1))

        vp_pos_np = opt_result.positions.get()
        vp_rot_np = opt_result.rotations.get()
        all_pos = np.vstack([home_pos, vp_pos_np])
        home_indices = list(range(K))

        # og_vrp must accept every viewpoint the sampling OG accepts. At 0.20 m
        # resolution (kept coarse because cuGraph OOMs on the 0.10 m sampling
        # grid), worst-case voxel alignment mismatch is ~0.67 m, so any
        # inflation > 1 voxel can reject a viewpoint the sampling OG considers
        # feasible. Inflate by 1 voxel only (0.20 m clearance) — OK for VRP
        # distance estimation (no MAPF in e17; paths need only pair waypoints
        # for route ordering). Also match sampling OG's padding so grid origins
        # align and `compute_grid_bounds` agrees on voxel centres.
        import trimesh
        from shared.grid_builder_utils import compute_grid_bounds, voxelize_mesh
        from shared.grid_utils import inflate_grid
        from visibility.sampling.utils.sampling_occupancy_grid import SamplingOccupancyGrid

        _verts = np.asarray(o3d_mesh.vertices)
        _tm = trimesh.Trimesh(vertices=_verts, faces=np.asarray(o3d_mesh.triangles))
        _padding = model_cfg.frustum.far + 2 * model_cfg.collision_radius
        _vrp_origin, _vrp_shape = compute_grid_bounds(
            _verts.min(axis=0), _verts.max(axis=0), _padding, 0.20,
        )
        _raw_vrp, _ = voxelize_mesh(_tm, _vrp_shape, _vrp_origin, 0.20,
                                     fill_interior=False)
        og_vrp = SamplingOccupancyGrid(
            grid=inflate_grid(_raw_vrp, 1),
            origin=_vrp_origin,
            resolution=0.20,
            raw_grid=_raw_vrp,
            filled_raw_grid=_raw_vrp,
        )
        dist_matrix = compute_distance_matrix(og_vrp, cp.asarray(all_pos))
        vrp_result: VRPResult = solve_vrp(
            dist_matrix=dist_matrix, num_vehicles=K, depots=home_indices,
            alpha=_E17_ALPHA, backend=VRPBackend.CUOPT, time_limit=120,
        )

    pv = per_vehicle_costs(vrp_result.routes, dist_matrix, home_indices)
    makespan = max(pv) if pv else 0.0
    total_cost = vrp_result.total_cost

    # LB sidecar (VRP-only; this experiment does not run MAPF).
    lb: dict | None = None
    try:
        lb = compute_all_lbs(
            dist_matrix, home_indices, K, num_viewpoints, _E17_ALPHA,
            include_mapf=False,
            vrp_best_bound_m=vrp_result.best_bound,
            vrp_objective_value_m=vrp_result.objective_value,
        )
    except Exception as e:
        logger.warning("LB computation failed: %s", e)

    result = {
        "model": model_cfg.name,
        "strategy": strategy,
        "travel_weight": travel_weight,
        "seed": seed,
        "n_base_candidates": n_base,
        "n_iterative_candidates": n_iter,
        "n_warmstart_fallbacks": int(n_ws_fb),
        "num_candidates": int(len(pos_gpu)),
        "num_viewpoints": num_viewpoints,
        "coverage": coverage,
        "redundancy": redundancy,
        "makespan": makespan,
        "total_cost": total_cost,
        "vrp_status": vrp_result.status,
        "t_sample": t_sample.elapsed,
        "t_vis": t_vis.elapsed,
        "t_opt": t_opt.elapsed,
        "t_vrp": t_vrp.elapsed,
        "t_total": t_sample.elapsed + t_vis.elapsed + t_opt.elapsed + t_vrp.elapsed,
    }
    return result, lb


# ═══════════════════════════════════════════════════════════════════════════
# Plot generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_plots(results: list[dict], output_dir: str,
                   lb_by_stem: dict | None = None,
                   target_coverage: float = 0.95):
    """Generate all E17 figures, one set per model."""
    setup_thesis_style()
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    lb_by_stem = lb_by_stem or {}

    all_ok = [r for r in results if r.get("vrp_status") == "success"]
    if not all_ok:
        logger.warning("No successful runs for plotting")
        return

    models = sorted(set(r.get("model", "unknown") for r in all_ok))
    for model_name in models:
        ok = [r for r in all_ok if r.get("model") == model_name]
        if not ok:
            continue
        _generate_plots_for_model(ok, fig_dir, model_name, lb_by_stem,
                                  target_coverage)


def _e17_lb_stem(r: dict) -> str:
    tw = r.get("travel_weight")
    tw_str = f"tw={tw}" if tw is not None else "tw=none"
    return (f"model={r['model']}_strategy={r['strategy']}_{tw_str}"
            f"_seed={r['seed']}")


def _generate_plots_for_model(ok: list[dict], fig_dir: str, model_name: str,
                              lb_by_stem: dict, target_coverage: float):
    # Ordered (strategy, travel_weight) combos for x-axis
    seen = set()
    run_cfgs = []
    for r in ok:
        key = (r["strategy"], r["travel_weight"])
        if key not in seen:
            run_cfgs.append(key)
            seen.add(key)
    non_cmaes = sorted([c for c in run_cfgs if c[1] is None], key=lambda x: x[0])
    cmaes = sorted([c for c in run_cfgs if c[1] is not None], key=lambda x: x[1])
    run_cfgs = non_cmaes + cmaes

    labels = [_display_label(s, tw) for s, tw in run_cfgs]
    colors = [_bar_color(s) for s, _ in run_cfgs]
    x = np.arange(len(run_cfgs))

    def _vals(metric):
        means, stds = [], []
        for s, tw in run_cfgs:
            vals = [r[metric] for r in ok
                    if r["strategy"] == s and r["travel_weight"] == tw]
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if vals else 0)
        return means, stds

    # Coverage is a confounder on makespan/total_cost — a strategy that finishes
    # early with missed points will look cheap. Flag any (strategy, tw) whose
    # mean coverage is below target, and hatch those bars on the main plots.
    cov_m, cov_s = _vals("coverage")  # mean in [0, 1]
    below_target = [c < target_coverage for c in cov_m]

    def _mark_below_target(bars):
        for bar, below in zip(bars, below_target):
            if below:
                bar.set_hatch("///")
                bar.set_edgecolor("0.25")
                bar.set_linewidth(0.8)

    # ── Fig 1: Makespan by strategy (main result) ──────────────────────
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4))
    mk_m, mk_s = _vals("makespan")
    bars = ax.bar(x, mk_m, yerr=mk_s, color=colors, capsize=3, alpha=0.85)
    _mark_below_target(bars)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("Makespan (m)")
    ax.set_title("VRP Makespan by Sampler Strategy")
    if any(below_target):
        hatch_proxy = plt.Rectangle(
            (0, 0), 1, 1, facecolor="white", edgecolor="0.25",
            hatch="///", linewidth=0.8)
        ax.legend([hatch_proxy],
                  [f"Coverage < {target_coverage*100:.0f}% target"],
                  fontsize=7, loc="best")
    fig.tight_layout()
    save_figure(fig, os.path.join(fig_dir, f"{model_name}_e17_makespan_by_strategy"))

    # ── Fig 2: Total cost by strategy ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4))
    tc_m, tc_s = _vals("total_cost")
    bars = ax.bar(x, tc_m, yerr=tc_s, color=colors, capsize=3, alpha=0.85)
    _mark_below_target(bars)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("Total route cost (m)")
    ax.set_title("VRP Total Cost by Sampler Strategy")
    if any(below_target):
        hatch_proxy = plt.Rectangle(
            (0, 0), 1, 1, facecolor="white", edgecolor="0.25",
            hatch="///", linewidth=0.8)
        ax.legend([hatch_proxy],
                  [f"Coverage < {target_coverage*100:.0f}% target"],
                  fontsize=7, loc="best")
    fig.tight_layout()
    save_figure(fig, os.path.join(fig_dir, f"{model_name}_e17_total_cost_by_strategy"))

    # ── Fig 2b: Coverage by strategy (interprets Fig 1 / Fig 2) ────────
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.5))
    cov_m_pct = [c * 100 for c in cov_m]
    cov_s_pct = [c * 100 for c in cov_s]
    ax.bar(x, cov_m_pct, yerr=cov_s_pct, color=colors, capsize=3, alpha=0.85)
    ax.axhline(target_coverage * 100, color="red", linestyle="--",
               linewidth=1.0, label=f"Target {target_coverage*100:.0f}%")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("Coverage (%)")
    ax.set_title("Coverage by Sampler Strategy")
    # Zoom y-axis to the interesting band (don't waste space on 0-100 when all
    # strategies are near the target).
    y_min = max(0.0, min(min(cov_m_pct), target_coverage * 100) - 5)
    ax.set_ylim(bottom=y_min, top=102)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    save_figure(fig, os.path.join(fig_dir, f"{model_name}_e17_coverage_by_strategy"))

    # ── Fig 3: Num viewpoints by strategy ──────────────────────────────
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4))
    vp_m, vp_s = _vals("num_viewpoints")
    ax.bar(x, vp_m, yerr=vp_s, color=colors, capsize=3, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("Selected viewpoints")
    ax.set_title("Viewpoints After Set Cover by Strategy")
    fig.tight_layout()
    save_figure(fig, os.path.join(fig_dir, f"{model_name}_e17_viewpoints_by_strategy"))

    # ── Fig 4: Scatter — viewpoints vs makespan ───────────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 4))
    strategy_groups: dict[str, dict] = {}
    for r in ok:
        key = r["strategy"]
        if key not in strategy_groups:
            strategy_groups[key] = {"vp": [], "mk": []}
        strategy_groups[key]["vp"].append(r["num_viewpoints"])
        strategy_groups[key]["mk"].append(r["makespan"])
    for i, (strat, vals) in enumerate(sorted(strategy_groups.items())):
        ax.scatter(vals["vp"], vals["mk"],
                   color=CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)],
                   label=strat, s=40, alpha=0.8)
    ax.set_xlabel("Selected viewpoints")
    ax.set_ylabel("Makespan (m)")
    ax.set_title("Viewpoints vs Makespan")
    ax.legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, os.path.join(fig_dir, f"{model_name}_e17_scatter_vp_vs_makespan"))

    # ── Fig 5: Stacked bar — timing breakdown ─────────────────────────
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4))
    stage_data = {
        "Sampling": _vals("t_sample")[0],
        "Visibility": _vals("t_vis")[0],
        "Set Cover": _vals("t_opt")[0],
        "VRP": _vals("t_vrp")[0],
    }
    stacked_bar(ax, labels, stage_data,
                ylabel="Time (s)",
                title="Pipeline Time Breakdown by Strategy")
    ax.tick_params(axis="x", rotation=40)
    fig.tight_layout()
    save_figure(fig, os.path.join(fig_dir, f"{model_name}_e17_timing_breakdown"))

    # ── Fig 6: CMA-ES makespan vs travel_weight ──────────────────────
    cmaes_runs = [r for r in ok if r["strategy"] == "cmaes_100"
                  and r["travel_weight"] is not None]
    if cmaes_runs:
        fig, ax = plt.subplots(figsize=(THESIS_COL, 3.5))
        tws = sorted(set(r["travel_weight"] for r in cmaes_runs))
        mk_means = [np.mean([r["makespan"] for r in cmaes_runs
                             if r["travel_weight"] == tw]) for tw in tws]
        mk_stds = [np.std([r["makespan"] for r in cmaes_runs
                           if r["travel_weight"] == tw]) for tw in tws]
        ax.errorbar(tws, mk_means, yerr=mk_stds, marker="o", capsize=3,
                    color=CATEGORICAL_COLORS[3], linewidth=1.5)
        ax.set_xlabel("Travel weight")
        ax.set_ylabel("Makespan (m)")
        ax.set_title("CMA-ES: Makespan vs Travel Weight")
        fig.tight_layout()
        save_figure(fig, os.path.join(fig_dir, f"{model_name}_e17_cmaes_tw_vs_makespan"))

    # ── Fig 7: Objective + cuOpt bound by strategy (paired bars) ──────
    # Only drawn when at least one matching LB sidecar was loaded.
    lb_pairs = []  # (incumbent_obj, cuopt_lb) per (strategy, tw) combo
    for s, tw in run_cfgs:
        rows = [r for r in ok if r["strategy"] == s and r["travel_weight"] == tw]
        if not rows:
            lb_pairs.append((float("nan"), float("nan")))
            continue
        incumbent = float(np.mean(
            [_E17_ALPHA * r["makespan"] + (1 - _E17_ALPHA) * r["total_cost"]
             for r in rows]))
        lbs = [lb_by_stem.get(_e17_lb_stem(r), {}).get(
                   "vrp_objective_best_bound_m", 0.0) for r in rows]
        lbs = [v for v in lbs if v > 0]
        lb_m = float(np.mean(lbs)) if lbs else float("nan")
        lb_pairs.append((incumbent, lb_m))
    if any(not np.isnan(p[1]) for p in lb_pairs):
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4))
        width = 0.38
        inc = [p[0] for p in lb_pairs]
        lb_vals = [p[1] if not np.isnan(p[1]) else 0.0 for p in lb_pairs]
        ax.bar(x - width / 2, inc, width, color=colors, alpha=0.85,
               label="Incumbent objective")
        ax.bar(x + width / 2, lb_vals, width,
               color=[c for c in colors], alpha=0.4, hatch="//",
               label="cuOpt dual bound (mean)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right")
        ax.set_ylabel(f"VRP objective (α={_E17_ALPHA}) (m)")
        ax.set_title(f"Objective vs cuOpt Bound ({model_name})")
        ax.legend(fontsize=7)
        fig.tight_layout()
        save_figure(fig, os.path.join(
            fig_dir, f"{model_name}_e17_objective_by_strategy"))

    logger.info("E17 figures saved to %s", fig_dir)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="E17: Sampler Routing Impact")
    p.add_argument("--models", nargs="+", default=_E17_MODELS,
                   help="Models to evaluate (default: Duke + TOSCA_REPRESENTATIVE)")
    p.add_argument("--strategies", nargs="+", default=_E17_STRATEGIES)
    p.add_argument("--cmaes_travel_weights", type=float, nargs="+", default=None,
                   help="Override per-model travel-weight sweep; if unset, uses "
                        "E00_C_TRAVEL_WEIGHTS_{DUKE,TOSCA} depending on the model.")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_3)
    p.add_argument("--target_coverage", type=float, default=0.95)
    p.add_argument("--output_dir",
                   default=os.path.join(RESULTS_DIR, "e17_sampler_routing_impact"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--plots_only", action="store_true",
                   help="Only regenerate plots from existing results")
    p.add_argument("--compute_lbs_only", action="store_true",
                   help="Skip VRP solves; rebuild each existing instance "
                        "(sampling + visibility + set cover + distance "
                        "matrix), compute analytical LBs, write sidecar "
                        "JSONs into raw_lb/.")
    p.add_argument("--include_cuopt_bound", action="store_true",
                   help="In --compute_lbs_only mode, also run a short cuOpt "
                        "solve per instance to extract the MIP dual bound. "
                        "Expensive; off by default.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
    )

    raw_dir = os.path.join(args.output_dir, "raw")
    raw_lb_dir = os.path.join(args.output_dir, "raw_lb")
    os.makedirs(raw_dir, exist_ok=True)
    all_results: list[dict] = []

    def _make_cfg(name: str):
        if name == "duke_of_lancaster":
            return ModelConfig.duke_of_lancaster()
        try:
            return ModelConfig.tosca(name)
        except FileNotFoundError:
            logger.warning("Model not found: %s", name)
            return None

    def _lb_path_for(result_path: str) -> str:
        """Mirror path in raw_lb/ for a given raw/ result path."""
        stem = os.path.basename(result_path)
        return os.path.join(raw_lb_dir, stem)

    if args.compute_lbs_only:
        if not os.path.isdir(raw_dir):
            logger.error("No raw/ directory at %s; nothing to augment.", raw_dir)
            return
        os.makedirs(raw_lb_dir, exist_ok=True)
        # Group existing main JSONs by model so we only build one
        # PipelineContext per model.
        stems = sorted(f[:-5] for f in os.listdir(raw_dir) if f.endswith(".json"))
        by_model: dict[str, list[dict]] = {}
        for stem in stems:
            main = load_run_result(os.path.join(raw_dir, stem))
            all_results.append(main)
            by_model.setdefault(main["model"], []).append({**main, "_stem": stem})

        logger.info(
            "Recomputing LBs for %d rows across %d models%s ...",
            len(stems), len(by_model),
            " including cuOpt bound" if args.include_cuopt_bound else "",
        )

        for model_name, rows in by_model.items():
            cfg = _make_cfg(model_name)
            if cfg is None:
                logger.warning("Skipping %d rows for missing model %s",
                               len(rows), model_name)
                continue
            ctx = PipelineContext(cfg)
            ctx.load_mesh()
            ctx.sample_surface(seed=42)
            ctx.build_sampling_og()

            logger.info("[%s] %d rows to recompute", model_name, len(rows))
            for idx, row in enumerate(rows, 1):
                stem = row["_stem"]
                try:
                    K, home_indices, dist_matrix, n_vp = _rebuild_instance_for_lb(
                        ctx, row["strategy"], row["travel_weight"],
                        int(row["seed"]), args.target_coverage,
                    )
                    lb = recompute_lbs(
                        dist_matrix, home_indices, K, n_vp, _E17_ALPHA,
                        include_mapf=False,
                        include_cuopt=args.include_cuopt_bound,
                    )
                    save_lb_json(os.path.join(raw_lb_dir, stem), lb)
                except Exception as e:
                    logger.error("LB recompute failed for %s: %s", stem, e)
                finally:
                    free_gpu_memory()
                if idx % 5 == 0:
                    logger.info("  [%s] ... %d/%d done", model_name, idx, len(rows))

        lb_by_stem = load_raw_lb_dir(raw_lb_dir, raw_dir)
        if all_results:
            generate_plots(all_results, args.output_dir, lb_by_stem=lb_by_stem,
                           target_coverage=args.target_coverage)
        return

    if not args.plots_only:
        os.makedirs(raw_lb_dir, exist_ok=True)
        for model_name in args.models:
            cfg = _make_cfg(model_name)
            if cfg is None:
                continue
            tws = (list(args.cmaes_travel_weights)
                   if args.cmaes_travel_weights is not None
                   else _cmaes_travel_weights(model_name))
            run_cfgs = _build_run_configs(args.strategies, tws)
            combos = [(s, tw, seed) for s, tw in run_cfgs for seed in args.seeds]
            total = len(combos)

            logger.info("=" * 60)
            logger.info("E17 — Model: %s — %d runs (%d configs x %d seeds)",
                        model_name, total, len(run_cfgs), len(args.seeds))

            ctx = PipelineContext(cfg)
            ctx.load_mesh()
            ctx.sample_surface(seed=42)
            ctx.build_sampling_og()

            for idx, (strategy, tw, seed) in enumerate(combos, 1):
                rpath = _result_path(raw_dir, model_name, strategy, tw, seed)

                if args.resume and os.path.exists(rpath + ".json"):
                    logger.info("[%s %d/%d] SKIP %s tw=%s seed=%d",
                                model_name, idx, total, strategy, tw, seed)
                    all_results.append(load_run_result(rpath))
                    continue

                logger.info("[%s %d/%d] strategy=%s travel_weight=%s seed=%d",
                            model_name, idx, total, strategy, tw, seed)
                try:
                    result, lb = run_single(ctx, strategy, tw, seed,
                                            args.target_coverage)
                    all_results.append(result)
                    save_run_result(result, rpath)
                    if lb is not None:
                        save_lb_json(_lb_path_for(rpath), lb)
                    logger.info(
                        "  vps=%d cov=%.2f%% makespan=%.1f total_cost=%.1f t=%.1fs",
                        result["num_viewpoints"],
                        result["coverage"] * 100,
                        result["makespan"],
                        result["total_cost"],
                        result["t_total"])
                except Exception as e:
                    handle_row_exception(
                        e,
                        f"model={model_name} strategy={strategy} tw={tw} seed={seed}",
                        resume=args.resume,
                    )
                finally:
                    free_gpu_memory()
    else:
        for fname in sorted(os.listdir(raw_dir)):
            if fname.endswith(".json"):
                all_results.append(
                    load_run_result(os.path.join(raw_dir, fname.replace(".json", ""))))

    lb_by_stem = load_raw_lb_dir(raw_lb_dir, raw_dir)
    if all_results:
        generate_plots(all_results, args.output_dir, lb_by_stem=lb_by_stem,
                       target_coverage=args.target_coverage)

        # Summary table (per model)
        models = sorted(set(r.get("model", "unknown") for r in all_results))
        for model_name in models:
            logger.info("\n%s\nE17 SUMMARY — %s\n%s", "=" * 80, model_name, "=" * 80)
            logger.info("%-22s %6s %8s %10s %10s %10s %8s",
                        "Strategy", "TW", "VPs", "Coverage%", "Makespan", "TotalCost", "Time(s)")
            logger.info("-" * 80)
            tws = (list(args.cmaes_travel_weights)
                   if args.cmaes_travel_weights is not None
                   else _cmaes_travel_weights(model_name))
            run_cfgs = _build_run_configs(args.strategies, tws)
            for s, tw in run_cfgs:
                sr = [r for r in all_results
                      if r.get("model") == model_name
                      and r["strategy"] == s and r["travel_weight"] == tw
                      and r.get("vrp_status") == "success"]
                if sr:
                    logger.info(
                        "%-22s %6s %8.0f %10.2f %10.1f %10.1f %8.1f",
                        s,
                        f"{tw}" if tw is not None else "—",
                        np.mean([r["num_viewpoints"] for r in sr]),
                        np.mean([r["coverage"] * 100 for r in sr]),
                        np.mean([r["makespan"] for r in sr]),
                        np.mean([r["total_cost"] for r in sr]),
                        np.mean([r["t_total"] for r in sr]),
                    )


if __name__ == "__main__":
    main()
