#!/usr/bin/env python3
"""E17: Sampler Routing Impact

Tests whether the sampler strategy affects downstream VRP routing objectives
(makespan and total_cost).  Runs stages 1-7 of the pipeline (skips MAPF) with
varying sampler strategies and measures how viewpoint spatial distribution
propagates to routing cost.

Hypothesis: CMA-ES's travel_weight parameter penalises point-to-point distance
during sampling, producing viewpoint sets that are cheaper to route through.

Strategies: weighted, weighted_curvature, targeted_25, targeted_50, cmaes_100
CMA-ES travel_weight sweep: [0.0, 0.1, 0.3]
Fixed: 5 robots, 1500 candidates, 0.95 target coverage, Duke of Lancaster.
3 seeds.

Usage:
    conda run -n isaaclab python -m experiments.e17_sampler_routing_impact
    conda run -n isaaclab python -m experiments.e17_sampler_routing_impact --plots_only
    conda run -n isaaclab python -m experiments.e17_sampler_routing_impact --skip_existing
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
    ModelConfig, SEEDS_3, RESULTS_DIR,
    E17_STRATEGIES, E17_CMAES_TRAVEL_WEIGHTS, E17_N_ROBOTS, E17_N_CANDIDATES,
)
from experiments.common.runner import set_seed, timed, free_gpu_memory
from experiments.common.pipeline_setup import PipelineContext
from experiments.common.persistence import save_run_result, load_run_result
from experiments.common.sampling_dispatch import sample_strategy
from experiments.common.plotting import (
    setup_thesis_style, save_figure, stacked_bar,
    THESIS_COL, DOUBLE_COL, CATEGORICAL_COLORS,
)

from visibility.set_cover import LazyGreedySetCover

from VRP.core.distance_matrix import compute_distance_matrix
from VRP.vrp.vrp_solver import solve_vrp
from VRP.core.types import VRPBackend, VRPResult
from VRP.core.geometry import compute_start_grid
from VRP.vrp._helpers import per_vehicle_costs
from VRP.core.constants import ROBOT_RADIUS

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
    configs = []
    for s in strategies:
        if s.startswith("cmaes_"):
            for tw in cmaes_travel_weights:
                configs.append((s, tw))
        else:
            configs.append((s, None))
    return configs


def _result_path(raw_dir, strategy, travel_weight, seed):
    tw_str = f"tw={travel_weight}" if travel_weight is not None else "tw=none"
    return os.path.join(raw_dir, f"strategy={strategy}_{tw_str}_seed={seed}")


# ═══════════════════════════════════════════════════════════════════════════
# Single run logic
# ═══════════════════════════════════════════════════════════════════════════

def run_single(ctx: PipelineContext, strategy: str, travel_weight,
               seed: int, target_coverage: float = 0.95) -> dict:
    """Run stages 1-7 for one (strategy, travel_weight, seed) combo."""
    set_seed(seed)
    model_cfg = ctx.model
    target_points, normals = ctx.sample_surface()
    vis_query = ctx.build_visibility_query("raycast")

    # Stage 4: Sampling
    with timed() as t_sample:
        pos_gpu, rot_gpu, n_base, n_iter, base_name, n_ws_fb = sample_strategy(
            ctx, strategy, E17_N_CANDIDATES,
            target_points, normals, vis_query, model_cfg,
            k_coverage=4,
            samples_per_iteration=25,
            travel_weight=travel_weight,
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

        from visibility.sampling.utils.sampling_grid_builder import build_sampling_occupancy_grid
        og_vrp = build_sampling_occupancy_grid(
            mesh=o3d_mesh,
            frustum_far=model_cfg.frustum.far,
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

    return {
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


# ═══════════════════════════════════════════════════════════════════════════
# Plot generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_plots(results: list[dict], output_dir: str):
    """Generate all E17 figures."""
    setup_thesis_style()
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    ok = [r for r in results if r.get("vrp_status") == "success"]
    if not ok:
        logger.warning("No successful runs for plotting")
        return

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

    # ── Fig 1: Makespan by strategy (main result) ──────────────────────
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4))
    mk_m, mk_s = _vals("makespan")
    ax.bar(x, mk_m, yerr=mk_s, color=colors, capsize=3, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("Makespan (m)")
    ax.set_title("VRP Makespan by Sampler Strategy")
    fig.tight_layout()
    save_figure(fig, os.path.join(fig_dir, "e17_makespan_by_strategy"))

    # ── Fig 2: Total cost by strategy ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4))
    tc_m, tc_s = _vals("total_cost")
    ax.bar(x, tc_m, yerr=tc_s, color=colors, capsize=3, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("Total route cost (m)")
    ax.set_title("VRP Total Cost by Sampler Strategy")
    fig.tight_layout()
    save_figure(fig, os.path.join(fig_dir, "e17_total_cost_by_strategy"))

    # ── Fig 3: Num viewpoints by strategy ──────────────────────────────
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4))
    vp_m, vp_s = _vals("num_viewpoints")
    ax.bar(x, vp_m, yerr=vp_s, color=colors, capsize=3, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("Selected viewpoints")
    ax.set_title("Viewpoints After Set Cover by Strategy")
    fig.tight_layout()
    save_figure(fig, os.path.join(fig_dir, "e17_viewpoints_by_strategy"))

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
    save_figure(fig, os.path.join(fig_dir, "e17_scatter_vp_vs_makespan"))

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
    save_figure(fig, os.path.join(fig_dir, "e17_timing_breakdown"))

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
        save_figure(fig, os.path.join(fig_dir, "e17_cmaes_tw_vs_makespan"))

    logger.info("E17 figures saved to %s", fig_dir)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="E17: Sampler Routing Impact")
    p.add_argument("--strategies", nargs="+", default=E17_STRATEGIES)
    p.add_argument("--cmaes_travel_weights", type=float, nargs="+",
                   default=E17_CMAES_TRAVEL_WEIGHTS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_3)
    p.add_argument("--target_coverage", type=float, default=0.95)
    p.add_argument("--output_dir",
                   default=os.path.join(RESULTS_DIR, "e17_sampler_routing_impact"))
    p.add_argument("--skip_existing", action="store_true")
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
    all_results: list[dict] = []

    if not args.plots_only:
        run_cfgs = _build_run_configs(args.strategies, args.cmaes_travel_weights)
        combos = [(s, tw, seed) for s, tw in run_cfgs for seed in args.seeds]
        total = len(combos)

        logger.info("E17: %d runs (%d configs x %d seeds), Duke of Lancaster",
                    total, len(run_cfgs), len(args.seeds))

        model_cfg = ModelConfig.duke_of_lancaster()
        ctx = PipelineContext(model_cfg)
        ctx.load_mesh()
        ctx.sample_surface(seed=42)
        ctx.build_sampling_og()

        for idx, (strategy, tw, seed) in enumerate(combos, 1):
            rpath = _result_path(raw_dir, strategy, tw, seed)

            if args.skip_existing and os.path.exists(rpath + ".json"):
                logger.info("[%d/%d] SKIP %s tw=%s seed=%d",
                            idx, total, strategy, tw, seed)
                all_results.append(load_run_result(rpath))
                continue

            logger.info("[%d/%d] strategy=%s travel_weight=%s seed=%d",
                        idx, total, strategy, tw, seed)
            try:
                result = run_single(ctx, strategy, tw, seed, args.target_coverage)
                all_results.append(result)
                save_run_result(result, rpath)
                logger.info("  vps=%d cov=%.2f%% makespan=%.1f total_cost=%.1f t=%.1fs",
                            result["num_viewpoints"],
                            result["coverage"] * 100,
                            result["makespan"],
                            result["total_cost"],
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

        # Summary table
        logger.info("\n%s\nE17 SUMMARY\n%s", "=" * 80, "=" * 80)
        logger.info("%-22s %6s %8s %10s %10s %10s %8s",
                    "Strategy", "TW", "VPs", "Coverage%", "Makespan", "TotalCost", "Time(s)")
        logger.info("-" * 80)
        run_cfgs = _build_run_configs(args.strategies, args.cmaes_travel_weights)
        for s, tw in run_cfgs:
            sr = [r for r in all_results
                  if r["strategy"] == s and r["travel_weight"] == tw
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
