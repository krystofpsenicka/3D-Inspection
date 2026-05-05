#!/usr/bin/env python3
"""E02: Candidate Count Scaling - coverage / VPs / runtime vs. ``num_candidates`` per strategy.

Each strategy is one line on the coverage-vs-N and VPs-vs-N plots, revealing whether scaling
is strategy-specific, where each strategy plateaus, and which wins at a given budget.

Strategies:
  weighted             - SDF^2 uniform, no curvature (one-shot)
  weighted_curvature   - SDF^2 + curvature (one-shot)
  targeted             - targeted toward uncovered (iterative)
  cmaes                - CMA-ES optimised (iterative)

    python -m experiments.e02_candidate_scaling
    python -m experiments.e02_candidate_scaling --plots_only
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.common.config import (
    E02_CANDIDATE_COUNTS,
    RESULTS_DIR,
    SEEDS_3,
    ModelConfig,
)

_RUNTIME_IMPORT_ERROR: ImportError | None = None
try:
    import cupy as cp

    from experiments.common.pipeline_setup import PipelineContext
    from experiments.common.runner import free_gpu_memory, set_seed, timed
    from experiments.common.sampling_dispatch import sample_strategy

    _RUNTIME_AVAILABLE = True
except ImportError as _e:
    cp = None
    set_seed = timed = free_gpu_memory = None
    PipelineContext = None
    sample_strategy = None
    _RUNTIME_AVAILABLE = False
    _RUNTIME_IMPORT_ERROR = _e

_E02_STRATEGIES = ["weighted", "weighted_curvature", "targeted", "cmaes"]


def _strategy_kwargs(strategy: str) -> dict:
    if strategy == "targeted":
        return {"k_coverage": 3}
    if strategy == "cmaes":
        return {
            "k_coverage": 3,
            "travel_weight": 0.0,
            "popsize": 40,
            "maxiter": 40,
        }
    return {}


from experiments.common.persistence import load_run_result, save_run_result
from experiments.common.plotting import (
    CATEGORICAL_COLORS,
    DOUBLE_COL,
    save_figure,
    setup_thesis_style,
)

logger = logging.getLogger(__name__)


def run_single(
    ctx: PipelineContext,
    strategy: str,
    num_candidates: int,
    seed: int,
    target_coverage: float = 0.95,
) -> dict:
    target_points, normals = ctx.sample_surface()
    set_seed(seed)
    vis_query = ctx.build_visibility_query("raycast")
    model = ctx.model

    with timed() as t_sample:
        pos_gpu, rot_gpu, n_base, n_iter, base_name, _ = sample_strategy(
            ctx,
            strategy,
            num_candidates,
            target_points,
            normals,
            vis_query,
            model,
            **_strategy_kwargs(strategy),
        )

    with timed() as t_vis:
        V, _ = vis_query.compute_visibility_batch(pos_gpu, rot_gpu)

    from visibility.set_cover import LazyGreedySetCover  # CPU - fastest per e06 results

    V_np = cp.asnumpy(V)
    pos_np = cp.asnumpy(pos_gpu)
    rot_np = cp.asnumpy(rot_gpu)
    with timed() as t_opt:
        optimizer = LazyGreedySetCover(len(target_points), pos_np, rot_np, V_np)
        opt_result = optimizer.optimize(target_coverage=target_coverage, max_viewpoints=1000)

    return {
        "model": ctx.model.name,
        "strategy": strategy,
        "num_candidates_requested": num_candidates,
        "num_candidates": int(len(pos_gpu)),
        "n_base_candidates": n_base,
        "n_iterative_candidates": n_iter,
        "base_sampler": base_name,
        "seed": seed,
        "num_viewpoints": opt_result.num_viewpoints,
        "coverage": float(opt_result.total_coverage),
        "redundancy": float(opt_result.redundancy),
        "sampling_time": t_sample.elapsed,
        "visibility_time": t_vis.elapsed,
        "optimization_time": t_opt.elapsed,
        "total_time": t_sample.elapsed + t_vis.elapsed + t_opt.elapsed,
    }


def generate_plots(results: list[dict], strategies: list[str], output_dir: str):
    setup_thesis_style()
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    candidates = sorted(set(r["num_candidates_requested"] for r in results))

    def _agg(strategy, metric):
        means, stds = [], []
        for nc in candidates:
            vals = [
                r[metric]
                for r in results
                if r["strategy"] == strategy and r["num_candidates_requested"] == nc
            ]
            if vals:
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals)))
            else:
                means.append(float("nan"))
                stds.append(0.0)
        return np.array(means), np.array(stds)

    group_colors = {
        "weighted": CATEGORICAL_COLORS[0],
        "weighted_curvature": CATEGORICAL_COLORS[1],
        "targeted": CATEGORICAL_COLORS[2],
        "cmaes": CATEGORICAL_COLORS[3],
    }

    def _style(strategy):
        if strategy == "weighted":
            return group_colors["weighted"], "solid", "o"
        if strategy == "weighted_curvature":
            return group_colors["weighted_curvature"], "solid", "s"
        if strategy == "targeted":
            return group_colors["targeted"], "solid", "^"
        if strategy == "cmaes":
            return group_colors["cmaes"], "solid", "D"
        return "grey", "solid", "x"

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4))
    for strat in strategies:
        m, s = _agg(strat, "coverage")
        color, ls, marker = _style(strat)
        ax.plot(
            candidates,
            m * 100,
            marker=marker,
            linestyle=ls,
            color=color,
            label=strat,
            linewidth=1.5,
        )
        ax.fill_between(candidates, (m - s) * 100, (m + s) * 100, alpha=0.08, color=color)
    ax.axhline(95, color="red", linestyle="--", alpha=0.5, label="95% target")
    ax.set_xlabel("Candidates requested")
    ax.set_ylabel("Achieved coverage (%)")
    ax.set_title("Coverage vs. Candidate Count - All Strategies")
    ax.legend(fontsize=6, ncol=2)
    save_figure(fig, os.path.join(fig_dir, "e02_coverage_vs_candidates"))

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4))
    for strat in strategies:
        m, s = _agg(strat, "num_viewpoints")
        color, ls, marker = _style(strat)
        ax.plot(
            candidates,
            m,
            marker=marker,
            linestyle=ls,
            color=color,
            label=strat,
            linewidth=1.5,
        )
    ax.set_xlabel("Candidates requested")
    ax.set_ylabel("Selected viewpoints")
    ax.set_title("Selected Viewpoints vs. Candidate Count - All Strategies")
    ax.legend(fontsize=6, ncol=2)
    save_figure(fig, os.path.join(fig_dir, "e02_viewpoints_vs_candidates"))

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4))
    for strat in strategies:
        m, s = _agg(strat, "num_candidates")
        color, ls, marker = _style(strat)
        ax.plot(
            candidates,
            m,
            marker=marker,
            linestyle=ls,
            color=color,
            label=strat,
            linewidth=1.5,
        )
    ax.plot(candidates, candidates, "k--", alpha=0.3, label="N requested")
    ax.set_xlabel("Candidates requested")
    ax.set_ylabel("Candidates generated")
    ax.set_title("Actual Candidates Generated\n(CMA-ES may generate fewer due to early stopping)")
    ax.legend(fontsize=6, ncol=2)
    save_figure(fig, os.path.join(fig_dir, "e02_candidates_generated"))

    logger.info("E02 figures saved to %s", fig_dir)


def main():
    p = argparse.ArgumentParser(description="E02: Candidate Count Scaling")
    p.add_argument("--model", default="duke_of_lancaster")
    p.add_argument("--candidates", type=int, nargs="+", default=E02_CANDIDATE_COUNTS)
    p.add_argument("--strategies", nargs="+", default=_E02_STRATEGIES)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_3)
    p.add_argument("--target_coverage", type=float, default=0.95)
    p.add_argument("--output_dir", default=os.path.join(RESULTS_DIR, "e02_candidate_scaling"))
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

    raw_dir = os.path.join(args.output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    all_results = []

    if not args.plots_only:
        if args.model == "duke_of_lancaster":
            model_cfg = ModelConfig.duke_of_lancaster()
        else:
            model_cfg = ModelConfig.tosca(args.model)

        ctx = PipelineContext(model_cfg)
        ctx.load_mesh()
        ctx.sample_surface(seed=42)
        ctx.build_sampling_og()

        combos = [
            (s, nc, seed) for s in args.strategies for nc in args.candidates for seed in args.seeds
        ]
        total = len(combos)

        for idx, (strategy, nc, seed) in enumerate(combos, 1):
            rpath = os.path.join(raw_dir, f"strategy={strategy}_candidates={nc}_seed={seed}")

            if args.resume and os.path.exists(rpath + ".json"):
                logger.info("[%d/%d] SKIP %s nc=%d seed=%d", idx, total, strategy, nc, seed)
                all_results.append(load_run_result(rpath))
                continue

            logger.info("[%d/%d] strategy=%s candidates=%d seed=%d", idx, total, strategy, nc, seed)
            try:
                result = run_single(ctx, strategy, nc, seed, args.target_coverage)
                all_results.append(result)
                save_run_result(result, rpath)
                logger.info(
                    "  VPs=%d cov=%.2f%% generated=%d time=%.1fs",
                    result["num_viewpoints"],
                    result["coverage"] * 100,
                    result["num_candidates"],
                    result["total_time"],
                )
            except Exception as e:
                logger.error("  FAILED: %s", e, exc_info=True)
            finally:
                free_gpu_memory()
    else:
        for fname in sorted(os.listdir(raw_dir)):
            if fname.endswith(".json"):
                all_results.append(
                    load_run_result(os.path.join(raw_dir, fname.replace(".json", "")))
                )

    # Older raw files predate the per-strategy schema and lack the "strategy" field.
    all_results = [r for r in all_results if "strategy" in r]

    if all_results:
        generate_plots(all_results, args.strategies, args.output_dir)


if __name__ == "__main__":
    main()
