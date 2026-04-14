#!/usr/bin/env python3
"""E1: Sampling Strategy Comparison

Section A (k=1):
  10 strategies × 3 seeds × (Duke of Lancaster + TOSCA representative).
  Fixed candidate budget: model.num_candidates (1500 Duke, 500 TOSCA).
  Records n_base/n_iterative split, actual candidates generated, viewpoints
  selected, coverage, pre-set-cover pool redundancy, and per-stage timing.
  Base sampler for all hybrid strategies: weighted.

Section B (k>1):
  targeted_50 and cmaes_50 × k∈{1,2,3} × TOSCA representative only.
  Budget scales: N = 1500·k — gives the sampler room to reach k-fold coverage.
  Duke is excluded (too slow for the k-sweep).

Strategies (Section A):
  weighted            — SDF² uniform, all N from the weighted base sampler
  weighted_curvature  — SDF² + curvature bias, all N from base sampler
  targeted_X          — X% targeted toward uncovered, (100-X)% weighted
  cmaes_X             — X% CMA-ES optimised, (100-X)% weighted

Set-cover: LazyGreedySetCover (CPU, O(log N) heap) — fastest per e04 results.

Usage:
    conda run -n isaaclab python -m experiments.e01_sampling_strategy
    conda run -n isaaclab python -m experiments.e01_sampling_strategy --section A
    conda run -n isaaclab python -m experiments.e01_sampling_strategy --section B
    conda run -n isaaclab python -m experiments.e01_sampling_strategy --plots_only
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
    ModelConfig, SEEDS_3,
    E01_STRATEGIES_A, E01_STRATEGIES_B, E01_K_VALUES,
    TOSCA_REPRESENTATIVE, RESULTS_DIR,
)
from experiments.common.runner import set_seed, timed, free_gpu_memory
from experiments.common.pipeline_setup import PipelineContext, DegenerateNormalsError
from experiments.common.persistence import save_run_result, load_run_result
from experiments.common.sampling_dispatch import sample_strategy
from experiments.common.plotting import (
    setup_thesis_style, save_figure, grouped_bar, stacked_bar,
    THESIS_COL, DOUBLE_COL, CATEGORICAL_COLORS,
)
from visibility.set_cover import LazyGreedySetCover

logger = logging.getLogger(__name__)

# Candidate budget for Section B (scales with k)
_SECTION_B_BASE_N = 1500


# ═══════════════════════════════════════════════════════════════════════════
# Per-run helpers
# ═══════════════════════════════════════════════════════════════════════════

def _set_cover(target_points, pos_gpu, rot_gpu, V, target_coverage):
    """Run LazyGreedySetCover (CPU) and return opt_result."""
    V_np = cp.asnumpy(V)
    pos_np = cp.asnumpy(pos_gpu)
    rot_np = cp.asnumpy(rot_gpu)
    optimizer = LazyGreedySetCover(len(target_points), pos_np, rot_np, V_np)
    return optimizer.optimize(target_coverage=target_coverage, max_viewpoints=1000), V_np


def run_single_A(ctx: PipelineContext, strategy: str, seed: int,
                 target_coverage: float = 0.95) -> dict:
    """Section A: k=1, fixed N = model.num_candidates."""
    target_points, normals = ctx.sample_surface()
    set_seed(seed)
    vis_query = ctx.build_visibility_query("raycast")
    model = ctx.model

    with timed() as t_sample:
        pos_gpu, rot_gpu, n_base, n_iter, base_name = sample_strategy(
            ctx, strategy, model.num_candidates,
            target_points, normals, vis_query, model,
        )

    with timed() as t_vis:
        V, _ = vis_query.compute_visibility_batch(pos_gpu, rot_gpu)

    V_np = cp.asnumpy(V)
    pool_redundancy = float(V_np.sum() / len(target_points))

    with timed() as t_opt:
        opt_result, _ = _set_cover(target_points, pos_gpu, rot_gpu, V, target_coverage)

    return {
        "section": "A",
        "model": model.name,
        "strategy": strategy,
        "seed": seed,
        "k_coverage": 1,
        "n_base_candidates": n_base,
        "n_iterative_candidates": n_iter,
        "base_sampler": base_name,
        "num_candidates_requested": model.num_candidates,
        "num_candidates": int(len(pos_gpu)),
        "num_viewpoints": opt_result.num_viewpoints,
        "coverage": float(opt_result.total_coverage),
        "pool_redundancy": pool_redundancy,
        "redundancy": float(opt_result.redundancy),
        "sampling_time": t_sample.elapsed,
        "visibility_time": t_vis.elapsed,
        "optimization_time": t_opt.elapsed,
        "total_time": t_sample.elapsed + t_vis.elapsed + t_opt.elapsed,
    }


def run_single_B(ctx: PipelineContext, strategy: str, k_coverage: int,
                 seed: int, target_coverage: float = 0.95) -> dict:
    """Section B: variable k, N = _SECTION_B_BASE_N * k."""
    target_points, normals = ctx.sample_surface()
    set_seed(seed)
    vis_query = ctx.build_visibility_query("raycast")
    model = ctx.model
    num_candidates = _SECTION_B_BASE_N * k_coverage

    with timed() as t_sample:
        pos_gpu, rot_gpu, n_base, n_iter, base_name = sample_strategy(
            ctx, strategy, num_candidates,
            target_points, normals, vis_query, model,
            k_coverage=k_coverage,
        )

    with timed() as t_vis:
        V, _ = vis_query.compute_visibility_batch(pos_gpu, rot_gpu)

    V_np = cp.asnumpy(V)
    pool_redundancy = float(V_np.sum() / len(target_points))

    with timed() as t_opt:
        opt_result, _ = _set_cover(target_points, pos_gpu, rot_gpu, V, target_coverage)

    return {
        "section": "B",
        "model": model.name,
        "strategy": strategy,
        "k_coverage": k_coverage,
        "seed": seed,
        "n_base_candidates": n_base,
        "n_iterative_candidates": n_iter,
        "base_sampler": base_name,
        "num_candidates_requested": num_candidates,
        "num_candidates": int(len(pos_gpu)),
        "num_viewpoints": opt_result.num_viewpoints,
        "coverage": float(opt_result.total_coverage),
        "pool_redundancy": pool_redundancy,
        "redundancy": float(opt_result.redundancy),
        "sampling_time": t_sample.elapsed,
        "visibility_time": t_vis.elapsed,
        "optimization_time": t_opt.elapsed,
        "total_time": t_sample.elapsed + t_vis.elapsed + t_opt.elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ═══════════════════════════════════════════════════════════════════════════

# Consistent color/style per strategy group (matches e02)
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


def _agg(results, strategy, metric, model=None):
    """Return (mean, std) for a metric filtered by strategy (and optionally model)."""
    rows = [r for r in results if r["strategy"] == strategy]
    if model is not None:
        rows = [r for r in rows if r["model"] == model]
    vals = [r[metric] for r in rows]
    if not vals:
        return float("nan"), 0.0
    return float(np.mean(vals)), float(np.std(vals))


# ═══════════════════════════════════════════════════════════════════════════
# Section A plots
# ═══════════════════════════════════════════════════════════════════════════

def generate_plots_A(results: list[dict], strategies: list[str],
                     fig_dir: str, model_name: str):
    """Generate all Section A figures for one model."""
    mr = [r for r in results if r["model"] == model_name and r["section"] == "A"]
    if not mr:
        return

    colors = [_bar_color(s) for s in strategies]
    short_labels = [s.replace("weighted_curvature", "w_curv") for s in strategies]

    # ── Fig 1: Viewpoints and coverage side by side ─────────────────────
    fig, (ax_vp, ax_cov) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.5))

    vp_m = [_agg(mr, s, "num_viewpoints")[0] for s in strategies]
    vp_s = [_agg(mr, s, "num_viewpoints")[1] for s in strategies]
    ax_vp.bar(short_labels, vp_m, yerr=vp_s, color=colors, capsize=3, alpha=0.85)
    ax_vp.set_ylabel("Selected viewpoints")
    ax_vp.set_title(f"Viewpoints ({model_name})")
    ax_vp.tick_params(axis="x", rotation=40)

    cov_m = [_agg(mr, s, "coverage")[0] * 100 for s in strategies]
    cov_s = [_agg(mr, s, "coverage")[1] * 100 for s in strategies]
    ax_cov.bar(short_labels, cov_m, yerr=cov_s, color=colors, capsize=3, alpha=0.85)
    ax_cov.axhline(95, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
    ax_cov.set_ylabel("Coverage (%)")
    ax_cov.set_title(f"Coverage ({model_name})")
    ax_cov.tick_params(axis="x", rotation=40)

    fig.tight_layout()
    save_figure(fig, os.path.join(fig_dir, f"{model_name}_e01_A_main"))

    # ── Fig 2: Candidate split (stacked bar: base | iterative + actual dot) ─
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.5))
    base_vals = [_agg(mr, s, "n_base_candidates")[0] for s in strategies]
    iter_vals = [_agg(mr, s, "n_iterative_candidates")[0] for s in strategies]
    actual_vals = [_agg(mr, s, "num_candidates")[0] for s in strategies]

    x = np.arange(len(strategies))
    ax.bar(x, base_vals, color="lightgrey", edgecolor="grey", label="Base (weighted)")
    ax.bar(x, iter_vals, bottom=base_vals, color=colors, alpha=0.8, label="Iterative/CMA-ES")
    ax.scatter(x, actual_vals, color="black", zorder=5, s=20, label="Actual generated")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=40, ha="right")
    ax.set_ylabel("Candidates")
    ax.set_title(f"Candidate Split: Base vs Iterative ({model_name})\n"
                 "Base sampler for hybrid strategies: weighted")
    ax.legend(fontsize=7, ncol=3)
    save_figure(fig, os.path.join(fig_dir, f"{model_name}_e01_A_split"))

    # ── Fig 3: Pool redundancy vs selected redundancy ────────────────────
    fig, (ax_pool, ax_sel) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.5))

    pool_m = [_agg(mr, s, "pool_redundancy")[0] for s in strategies]
    pool_s = [_agg(mr, s, "pool_redundancy")[1] for s in strategies]
    ax_pool.bar(short_labels, pool_m, yerr=pool_s, color=colors, capsize=3, alpha=0.85)
    ax_pool.set_ylabel("Avg candidates per surface point")
    ax_pool.set_title("Pool Redundancy (pre-set-cover)")
    ax_pool.tick_params(axis="x", rotation=40)

    sel_m = [_agg(mr, s, "redundancy")[0] for s in strategies]
    sel_s = [_agg(mr, s, "redundancy")[1] for s in strategies]
    ax_sel.bar(short_labels, sel_m, yerr=sel_s, color=colors, capsize=3, alpha=0.85)
    ax_sel.set_ylabel("Avg viewpoints per covered point")
    ax_sel.set_title("Selected Redundancy (post-set-cover)")
    ax_sel.tick_params(axis="x", rotation=40)

    fig.tight_layout()
    save_figure(fig, os.path.join(fig_dir, f"{model_name}_e01_A_redundancy"))

    # ── Fig 4: Timing breakdown (stacked bar) ────────────────────────────
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.5))
    stacked_bar(ax, short_labels,
                {
                    "Sampling": [_agg(mr, s, "sampling_time")[0] for s in strategies],
                    "Visibility": [_agg(mr, s, "visibility_time")[0] for s in strategies],
                    "Optimization": [_agg(mr, s, "optimization_time")[0] for s in strategies],
                },
                ylabel="Time (s)",
                title=f"Per-Stage Timing ({model_name})")
    ax.tick_params(axis="x", rotation=40)
    save_figure(fig, os.path.join(fig_dir, f"{model_name}_e01_A_timing"))

    logger.info("Section A figures saved for %s", model_name)


# ═══════════════════════════════════════════════════════════════════════════
# Section B plots
# ═══════════════════════════════════════════════════════════════════════════

def generate_plots_B(results: list[dict], strategies: list[str],
                     k_values: list[int], fig_dir: str):
    """Generate Section B figures (k-coverage sweep, averaged over TOSCA models)."""
    br = [r for r in results if r["section"] == "B"]
    if not br:
        return

    def _mean_over_models(strategy, k, metric):
        vals = [r[metric] for r in br if r["strategy"] == strategy and r["k_coverage"] == k]
        return float(np.mean(vals)) if vals else float("nan")

    line_styles = {"targeted_50": "solid", "cmaes_50": "dashed"}
    line_colors = {"targeted_50": _GROUP_COLORS["targeted"], "cmaes_50": _GROUP_COLORS["cmaes"]}

    # ── Fig 5: Viewpoints vs k ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3))
    for strat in strategies:
        ys = [_mean_over_models(strat, k, "num_viewpoints") for k in k_values]
        ax.plot(k_values, ys, marker="o", linestyle=line_styles.get(strat, "solid"),
                color=line_colors.get(strat, "grey"), label=strat, linewidth=1.5)
    ax.set_xlabel("k-coverage requested")
    ax.set_ylabel("Selected viewpoints")
    ax.set_title("Viewpoints vs k-Coverage (TOSCA, N=1500·k)")
    ax.set_xticks(k_values)
    ax.legend()
    save_figure(fig, os.path.join(fig_dir, "e01_B_viewpoints_vs_k"))

    # ── Fig 6: Coverage vs k ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3))
    for strat in strategies:
        ys = [_mean_over_models(strat, k, "coverage") * 100 for k in k_values]
        ax.plot(k_values, ys, marker="o", linestyle=line_styles.get(strat, "solid"),
                color=line_colors.get(strat, "grey"), label=strat, linewidth=1.5)
    ax.axhline(95, color="red", linestyle="--", alpha=0.5, linewidth=0.8, label="95% target")
    ax.set_xlabel("k-coverage requested")
    ax.set_ylabel("Coverage (%)")
    ax.set_title("Coverage vs k-Coverage (TOSCA, N=1500·k)")
    ax.set_xticks(k_values)
    ax.legend()
    save_figure(fig, os.path.join(fig_dir, "e01_B_coverage_vs_k"))

    # ── Fig 7: Sampling time vs k ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3))
    for strat in strategies:
        ys = [_mean_over_models(strat, k, "sampling_time") for k in k_values]
        ax.plot(k_values, ys, marker="o", linestyle=line_styles.get(strat, "solid"),
                color=line_colors.get(strat, "grey"), label=strat, linewidth=1.5)
    ax.set_xlabel("k-coverage requested")
    ax.set_ylabel("Sampling time (s)")
    ax.set_title("Sampling Cost vs k-Coverage (TOSCA)")
    ax.set_xticks(k_values)
    ax.legend()
    save_figure(fig, os.path.join(fig_dir, "e01_B_samplingtime_vs_k"))

    # ── Fig 8: Actual candidates generated vs requested ───────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3))
    for strat in strategies:
        xs = [_SECTION_B_BASE_N * k for k in k_values]
        ys = [_mean_over_models(strat, k, "num_candidates") for k in k_values]
        ax.plot(xs, ys, marker="o", linestyle=line_styles.get(strat, "solid"),
                color=line_colors.get(strat, "grey"), label=strat, linewidth=1.5)
    ax.plot(xs, xs, "k--", alpha=0.3, label="N requested")
    ax.set_xlabel("Candidates requested (1500·k)")
    ax.set_ylabel("Candidates generated")
    ax.set_title("CMA-ES Early Stopping vs k-Coverage (TOSCA)")
    ax.legend()
    save_figure(fig, os.path.join(fig_dir, "e01_B_candidates_generated"))

    logger.info("Section B figures saved to %s", fig_dir)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="E1: Sampling Strategy Comparison")
    p.add_argument("--section", choices=["A", "B", "both"], default="both",
                   help="Which section to run (default: both)")
    p.add_argument("--models_A", nargs="+",
                   default=["duke_of_lancaster"] + TOSCA_REPRESENTATIVE,
                   help="Models for Section A")
    p.add_argument("--models_B", nargs="+",
                   default=TOSCA_REPRESENTATIVE,
                   help="Models for Section B (TOSCA only)")
    p.add_argument("--strategies_A", nargs="+", default=E01_STRATEGIES_A)
    p.add_argument("--strategies_B", nargs="+", default=E01_STRATEGIES_B)
    p.add_argument("--k_values", type=int, nargs="+", default=E01_K_VALUES)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_3)
    p.add_argument("--target_coverage", type=float, default=0.95)
    p.add_argument("--output_dir",
                   default=os.path.join(RESULTS_DIR, "e01_sampling_strategy"))
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--plots_only", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
    )

    raw_dir = os.path.join(args.output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    all_results: list[dict] = []

    # ── Build model configs ───────────────────────────────────────────────
    def _make_cfg(name: str):
        if name == "duke_of_lancaster":
            return ModelConfig.duke_of_lancaster()
        try:
            return ModelConfig.tosca(name)
        except FileNotFoundError:
            logger.warning("Model not found: %s", name)
            return None

    run_section_A = args.section in ("A", "both")
    run_section_B = args.section in ("B", "both")

    if not args.plots_only:
        # ── Section A ─────────────────────────────────────────────────────
        if run_section_A:
            a_cfgs = [c for name in args.models_A if (c := _make_cfg(name)) is not None]
            combos_A = [
                (cfg, s, seed)
                for cfg in a_cfgs
                for s in args.strategies_A
                for seed in args.seeds
            ]
            total_A = len(combos_A)
            logger.info("Section A: %d runs (%d models × %d strategies × %d seeds)",
                        total_A, len(a_cfgs), len(args.strategies_A), len(args.seeds))

            # Group by model so we load mesh once per model
            for cfg in a_cfgs:
                logger.info("=" * 60)
                logger.info("Section A — Model: %s", cfg.name)
                try:
                    ctx = PipelineContext(cfg)
                    ctx.load_mesh()
                    ctx.sample_surface(seed=42)
                    ctx.build_sampling_og()
                except DegenerateNormalsError as e:
                    logger.warning("Skipping %s: %s", cfg.name, e)
                    continue

                model_combos = [(s, seed) for s in args.strategies_A for seed in args.seeds]
                for idx, (strategy, seed) in enumerate(model_combos, 1):
                    rpath = os.path.join(
                        raw_dir, f"A_model={cfg.name}_strategy={strategy}_seed={seed}")
                    if args.skip_existing and os.path.exists(rpath + ".json"):
                        logger.info("[A %d/%d] SKIP %s %s seed=%d",
                                    idx, len(model_combos), cfg.name, strategy, seed)
                        all_results.append(load_run_result(rpath))
                        continue

                    logger.info("[A %d/%d] model=%s strategy=%s seed=%d",
                                idx, len(model_combos), cfg.name, strategy, seed)
                    try:
                        result = run_single_A(ctx, strategy, seed, args.target_coverage)
                        all_results.append(result)
                        save_run_result(result, rpath)
                        logger.info("  VPs=%d cov=%.2f%% actual=%d time=%.1fs",
                                    result["num_viewpoints"], result["coverage"] * 100,
                                    result["num_candidates"], result["total_time"])
                    except Exception as e:
                        logger.error("  FAILED: %s", e, exc_info=True)
                    finally:
                        free_gpu_memory()

        # ── Section B ─────────────────────────────────────────────────────
        if run_section_B:
            b_cfgs = [c for name in args.models_B if (c := _make_cfg(name)) is not None]
            logger.info("Section B: %d models × %d strategies × %d k-values × %d seeds",
                        len(b_cfgs), len(args.strategies_B),
                        len(args.k_values), len(args.seeds))

            for cfg in b_cfgs:
                logger.info("=" * 60)
                logger.info("Section B — Model: %s", cfg.name)
                try:
                    ctx = PipelineContext(cfg)
                    ctx.load_mesh()
                    ctx.sample_surface(seed=42)
                    ctx.build_sampling_og()
                except DegenerateNormalsError as e:
                    logger.warning("Skipping %s: %s", cfg.name, e)
                    continue

                b_combos = [
                    (s, k, seed)
                    for s in args.strategies_B
                    for k in args.k_values
                    for seed in args.seeds
                ]
                for idx, (strategy, k, seed) in enumerate(b_combos, 1):
                    rpath = os.path.join(
                        raw_dir,
                        f"B_model={cfg.name}_strategy={strategy}_k={k}_seed={seed}")
                    if args.skip_existing and os.path.exists(rpath + ".json"):
                        logger.info("[B %d/%d] SKIP %s %s k=%d seed=%d",
                                    idx, len(b_combos), cfg.name, strategy, k, seed)
                        all_results.append(load_run_result(rpath))
                        continue

                    logger.info("[B %d/%d] model=%s strategy=%s k=%d seed=%d",
                                idx, len(b_combos), cfg.name, strategy, k, seed)
                    try:
                        result = run_single_B(ctx, strategy, k, seed, args.target_coverage)
                        all_results.append(result)
                        save_run_result(result, rpath)
                        logger.info("  VPs=%d cov=%.2f%% actual=%d time=%.1fs",
                                    result["num_viewpoints"], result["coverage"] * 100,
                                    result["num_candidates"], result["total_time"])
                    except Exception as e:
                        logger.error("  FAILED: %s", e, exc_info=True)
                    finally:
                        free_gpu_memory()

    else:
        for fname in sorted(os.listdir(raw_dir)):
            if fname.endswith(".json"):
                all_results.append(load_run_result(
                    os.path.join(raw_dir, fname.replace(".json", ""))))

    if all_results:
        setup_thesis_style()
        fig_dir = os.path.join(args.output_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)

        a_models = sorted(set(r["model"] for r in all_results if r["section"] == "A"))
        if a_models and run_section_A:
            for model_name in a_models:
                generate_plots_A(all_results, args.strategies_A, fig_dir, model_name)

        b_results = [r for r in all_results if r["section"] == "B"]
        if b_results and run_section_B:
            generate_plots_B(b_results, args.strategies_B, args.k_values, fig_dir)

        # ── Summary table ──────────────────────────────────────────────
        logger.info("\n%s\nE1 SECTION A SUMMARY\n%s", "=" * 80, "=" * 80)
        for model_name in a_models:
            logger.info("\nModel: %s", model_name)
            logger.info("%-22s %8s %10s %10s %8s %8s",
                        "Strategy", "VPs", "Coverage%", "Time(s)", "PoolRed", "SelRed")
            logger.info("-" * 72)
            mr = [r for r in all_results if r["model"] == model_name and r["section"] == "A"]
            for s in args.strategies_A:
                sr = [r for r in mr if r["strategy"] == s]
                if sr:
                    logger.info(
                        "%-22s %8.0f %10.2f %10.1f %8.2f %8.2f",
                        s,
                        np.mean([r["num_viewpoints"] for r in sr]),
                        np.mean([r["coverage"] * 100 for r in sr]),
                        np.mean([r["total_time"] for r in sr]),
                        np.mean([r["pool_redundancy"] for r in sr]),
                        np.mean([r["redundancy"] for r in sr]),
                    )


if __name__ == "__main__":
    main()
