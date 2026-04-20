#!/usr/bin/env python3
"""E13: k-Coverage Impact on Sampling → Set Cover

Sweeps the k-coverage parameter for two strategies (targeted_50, cmaes_50) across
TOSCA representative models.  k-coverage tells the iterative sampler to keep placing
candidates until each surface point has at least k viewpoints covering it.

Higher k biases the candidate pool toward currently under-covered regions.
The key question: does requesting higher k from the sampler lead to fewer or more
viewpoints in the final set-cover solution — i.e., does the denser, more redundant
candidate pool help or hurt the greedy optimizer?

Parameters:
  strategy     {targeted_50, cmaes_50}
  k            {1, 2, 3, 5}
  target_cov   {0.90, 0.95}
  models       TOSCA representative (wolf0, cat0, david0) — Duke excluded (too slow)
  3 seeds

Candidate budget scales with k: N = 1500·k, so the sampler always has headroom
to achieve k-fold coverage.

Set-cover: LazyGreedySetCover (CPU, O(log N) heap) — fastest per e04 results.

Usage:
    conda run -n isaaclab python -m experiments.e13_viewpoint_redundancy
    conda run -n isaaclab python -m experiments.e13_viewpoint_redundancy --plots_only
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
    ModelConfig, SEEDS_3, TOSCA_REPRESENTATIVE, RESULTS_DIR,
    E13_K_VALUES, E13_COVERAGE_TARGETS, E13_STRATEGIES,
)
from experiments.common.runner import set_seed, timed, free_gpu_memory
from experiments.common.pipeline_setup import PipelineContext, DegenerateNormalsError
from experiments.common.persistence import save_run_result, load_run_result
from experiments.common.sampling_dispatch import sample_strategy
from experiments.common.plotting import (
    setup_thesis_style, save_figure, grouped_bar,
    THESIS_COL, DOUBLE_COL, CATEGORICAL_COLORS,
)
from visibility.set_cover import LazyGreedySetCover

logger = logging.getLogger(__name__)

# Budget base: N = BASE_N * k
_BASE_N = 1500


# ═══════════════════════════════════════════════════════════════════════════
# Single run logic
# ═══════════════════════════════════════════════════════════════════════════

def run_single(ctx: PipelineContext, strategy: str, k_coverage: int,
               target_coverage: float, seed: int) -> dict:
    """Run one (strategy, k, target_coverage, seed) combination."""
    target_points, normals = ctx.sample_surface()
    set_seed(seed)
    vis_query = ctx.build_visibility_query("raycast")
    model = ctx.model
    num_candidates = _BASE_N * k_coverage

    with timed() as t_sample:
        pos_gpu, rot_gpu, n_base, n_iter, base_name, _ = sample_strategy(
            ctx, strategy, num_candidates,
            target_points, normals, vis_query, model,
            k_coverage=k_coverage,
        )

    with timed() as t_vis:
        V, _ = vis_query.compute_visibility_batch(pos_gpu, rot_gpu)

    V_np = cp.asnumpy(V)
    pos_np = cp.asnumpy(pos_gpu)
    rot_np = cp.asnumpy(rot_gpu)
    pool_redundancy = float(V_np.sum() / len(target_points))

    with timed() as t_opt:
        optimizer = LazyGreedySetCover(len(target_points), pos_np, rot_np, V_np)
        opt_result = optimizer.optimize(
            target_coverage=target_coverage, max_viewpoints=1000)

    # Per-point coverage histogram from selected viewpoints
    selected_indices = opt_result.selected_indices
    if hasattr(selected_indices, "get"):
        selected_indices = selected_indices.get()
    selected_indices = np.asarray(selected_indices)
    V_selected = V_np[selected_indices]  # (n_vp, M)
    per_point_count = V_selected.astype(np.int32).sum(axis=0)  # (M,)
    max_count = int(per_point_count.max()) if len(per_point_count) > 0 else 0
    hist_counts = np.bincount(per_point_count, minlength=max_count + 1).tolist()

    return {
        "model": model.name,
        "strategy": strategy,
        "k_coverage": k_coverage,
        "target_coverage": target_coverage,
        "seed": seed,
        "num_candidates_requested": num_candidates,
        "num_candidates": int(len(pos_gpu)),
        "n_base_candidates": n_base,
        "n_iterative_candidates": n_iter,
        "num_viewpoints": opt_result.num_viewpoints,
        "coverage": float(opt_result.total_coverage),
        "pool_redundancy": pool_redundancy,
        "redundancy": float(opt_result.redundancy),
        "per_point_hist": hist_counts,
        "mean_point_coverage": float(per_point_count.mean()),
        "min_point_coverage": int(per_point_count.min()),
        "sampling_time": t_sample.elapsed,
        "visibility_time": t_vis.elapsed,
        "optimization_time": t_opt.elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Plot generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_plots(results: list[dict], strategies: list[str],
                   k_values: list[int], output_dir: str):
    """Generate all E13 figures (results averaged over models and seeds)."""
    setup_thesis_style()
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    if not results:
        logger.warning("No results to plot")
        return

    targets = sorted(set(r["target_coverage"] for r in results))

    strat_colors = {
        "targeted_50": CATEGORICAL_COLORS[2],
        "cmaes_50": CATEGORICAL_COLORS[3],
    }
    strat_ls = {"targeted_50": "solid", "cmaes_50": "dashed"}

    def _mean(strategy, k, target, metric):
        vals = [r[metric] for r in results
                if r["strategy"] == strategy
                and r["k_coverage"] == k
                and r["target_coverage"] == target]
        return float(np.mean(vals)) if vals else float("nan")

    def _std(strategy, k, target, metric):
        vals = [r[metric] for r in results
                if r["strategy"] == strategy
                and r["k_coverage"] == k
                and r["target_coverage"] == target]
        return float(np.std(vals)) if vals else 0.0

    # ── Fig 1: Viewpoints vs k (one line per strategy × target) ─────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3.5))
    for strat in strategies:
        for ti, target in enumerate(targets):
            m = [_mean(strat, k, target, "num_viewpoints") for k in k_values]
            s = [_std(strat, k, target, "num_viewpoints") for k in k_values]
            label = f"{strat} ({target*100:.0f}%)"
            ax.errorbar(k_values, m, yerr=s, marker="o", capsize=3,
                        color=strat_colors.get(strat, "grey"),
                        linestyle="solid" if ti == 0 else "dashed",
                        alpha=0.9 if ti == 0 else 0.6,
                        label=label, linewidth=1.5)
    ax.set_xlabel("k-coverage parameter")
    ax.set_ylabel("Selected viewpoints")
    ax.set_title("Viewpoints vs k-Coverage (N=1500·k, TOSCA)")
    ax.set_xticks(k_values)
    ax.legend(fontsize=7)
    save_figure(fig, os.path.join(fig_dir, "e13_viewpoints_vs_k"))

    # ── Fig 2: Coverage vs k ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3.5))
    for strat in strategies:
        for ti, target in enumerate(targets):
            m = [_mean(strat, k, target, "coverage") * 100 for k in k_values]
            label = f"{strat} ({target*100:.0f}%)"
            ax.plot(k_values, m, marker="o",
                    color=strat_colors.get(strat, "grey"),
                    linestyle="solid" if ti == 0 else "dashed",
                    alpha=0.9 if ti == 0 else 0.6,
                    label=label, linewidth=1.5)
            ax.axhline(target * 100, color=strat_colors.get(strat, "grey"),
                       linestyle=":", alpha=0.3, linewidth=0.8)
    ax.set_xlabel("k-coverage parameter")
    ax.set_ylabel("Coverage (%)")
    ax.set_title("Coverage vs k-Coverage (N=1500·k, TOSCA)")
    ax.set_xticks(k_values)
    ax.legend(fontsize=7)
    save_figure(fig, os.path.join(fig_dir, "e13_coverage_vs_k"))

    # ── Fig 3: Pool redundancy vs k ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3.5))
    target_main = 0.95 if 0.95 in targets else targets[-1]
    for strat in strategies:
        m = [_mean(strat, k, target_main, "pool_redundancy") for k in k_values]
        ax.plot(k_values, m, marker="s",
                color=strat_colors.get(strat, "grey"),
                linestyle=strat_ls.get(strat, "solid"),
                label=strat, linewidth=1.5)
    ax.set_xlabel("k-coverage parameter")
    ax.set_ylabel("Avg candidates per surface point")
    ax.set_title(f"Candidate Pool Redundancy vs k (target={target_main*100:.0f}%)")
    ax.set_xticks(k_values)
    ax.legend(fontsize=7)
    save_figure(fig, os.path.join(fig_dir, "e13_pool_redundancy_vs_k"))

    # ── Fig 4: Per-point coverage histogram at k=1 and k=max ────────────
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.5))
    for ax, k_plot in zip(axes, [1, max(k_values)]):
        for si, strat in enumerate(strategies):
            k_results = [r for r in results
                         if r["strategy"] == strat and r["k_coverage"] == k_plot
                         and r["target_coverage"] == target_main]
            if not k_results:
                continue
            max_len = max(len(r["per_point_hist"]) for r in k_results)
            padded = np.zeros((len(k_results), max_len))
            for i, r in enumerate(k_results):
                h = r["per_point_hist"]
                padded[i, :len(h)] = h
            mean_hist = padded.mean(axis=0)
            total = mean_hist.sum()
            bins = np.arange(len(mean_hist))
            ax.bar(bins + si * 0.35, mean_hist / (total + 1e-9), width=0.33,
                   alpha=0.7, label=strat,
                   color=strat_colors.get(strat, "grey"))
        ax.set_xlabel("Per-point coverage count")
        ax.set_ylabel("Fraction of points")
        ax.set_title(f"Coverage distribution k={k_plot}")
        ax.legend(fontsize=7)
    fig.suptitle(f"Point Coverage Histogram (target={target_main*100:.0f}%, TOSCA avg)")
    fig.tight_layout()
    save_figure(fig, os.path.join(fig_dir, "e13_coverage_histogram"))

    # ── Fig 5: Sampling time vs k ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3))
    for strat in strategies:
        m = [_mean(strat, k, target_main, "sampling_time") for k in k_values]
        ax.plot(k_values, m, marker="o",
                color=strat_colors.get(strat, "grey"),
                linestyle=strat_ls.get(strat, "solid"),
                label=strat, linewidth=1.5)
    ax.set_xlabel("k-coverage parameter")
    ax.set_ylabel("Sampling time (s)")
    ax.set_title("Sampling Cost vs k (N=1500·k, TOSCA)")
    ax.set_xticks(k_values)
    ax.legend(fontsize=7)
    save_figure(fig, os.path.join(fig_dir, "e13_samplingtime_vs_k"))

    logger.info("E13 figures saved to %s", fig_dir)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="E13: k-Coverage Impact")
    p.add_argument("--models", nargs="+", default=TOSCA_REPRESENTATIVE,
                   help="TOSCA models to evaluate (Duke excluded — too slow)")
    p.add_argument("--strategies", nargs="+", default=E13_STRATEGIES)
    p.add_argument("--k_values", type=int, nargs="+", default=E13_K_VALUES)
    p.add_argument("--targets", type=float, nargs="+", default=E13_COVERAGE_TARGETS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_3)
    p.add_argument("--output_dir",
                   default=os.path.join(RESULTS_DIR, "e13_viewpoint_redundancy"))
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

    if not args.plots_only:
        for model_name in args.models:
            try:
                cfg = ModelConfig.tosca(model_name)
            except FileNotFoundError:
                logger.warning("Model not found: %s — skipping", model_name)
                continue

            logger.info("=" * 60)
            logger.info("Model: %s", model_name)
            try:
                ctx = PipelineContext(cfg)
                ctx.load_mesh()
                ctx.sample_surface(seed=42)
                ctx.build_sampling_og()
            except DegenerateNormalsError as e:
                logger.warning("Skipping %s: %s", model_name, e)
                continue

            combos = [
                (s, k, target, seed)
                for s in args.strategies
                for k in args.k_values
                for target in args.targets
                for seed in args.seeds
            ]
            total = len(combos)
            for idx, (strategy, k, target, seed) in enumerate(combos, 1):
                rpath = os.path.join(
                    raw_dir,
                    f"model={model_name}_strategy={strategy}_k={k}"
                    f"_target={target}_seed={seed}",
                )
                if args.skip_existing and os.path.exists(rpath + ".json"):
                    logger.info("[%d/%d] SKIP %s %s k=%d target=%.2f seed=%d",
                                idx, total, model_name, strategy, k, target, seed)
                    all_results.append(load_run_result(rpath))
                    continue

                logger.info("[%d/%d] model=%s strategy=%s k=%d target=%.2f seed=%d",
                            idx, total, model_name, strategy, k, target, seed)
                try:
                    result = run_single(ctx, strategy, k, target, seed)
                    all_results.append(result)
                    save_run_result(result, rpath)
                    logger.info("  VPs=%d cov=%.2f%% pool_red=%.2f time=%.1fs",
                                result["num_viewpoints"],
                                result["coverage"] * 100,
                                result["pool_redundancy"],
                                result["sampling_time"] + result["visibility_time"]
                                + result["optimization_time"])
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
        generate_plots(all_results, args.strategies, args.k_values, args.output_dir)


if __name__ == "__main__":
    main()
