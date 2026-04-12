#!/usr/bin/env python3
"""E1: Sampling Strategy Comparison

Compares 5 viewpoint sampling strategies across multiple models:
  - weighted (SDF² weighting, no curvature)
  - weighted_curvature (SDF² + curvature weighting)
  - targeted_25 (25% targeted resampling)
  - targeted_50 (50% targeted resampling)
  - cmaes (CMA-ES optimization-based sampling)

Each strategy is evaluated by running the full visibility + set cover
pipeline and measuring: viewpoints selected, coverage achieved, timing
breakdown, and redundancy.

Usage:
    conda run -n isaaclab python -m experiments.e01_sampling_strategy
    conda run -n isaaclab python -m experiments.e01_sampling_strategy --models duke_of_lancaster
    conda run -n isaaclab python -m experiments.e01_sampling_strategy --models wolf0 cat0 --seeds 42 123
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import cupy as cp
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.common.config import (
    ModelConfig, SEEDS_5, E01_STRATEGIES, TOSCA_REPRESENTATIVE, RESULTS_DIR,
)
from experiments.common.runner import ExperimentRunner, set_seed, timed
from experiments.common.pipeline_setup import PipelineContext, DegenerateNormalsError
from experiments.common.plotting import (
    setup_thesis_style, save_figure, grouped_bar, violin_with_swarm,
    stacked_bar, cdf_plot, radar_chart, THESIS_COL, DOUBLE_COL,
    CATEGORICAL_COLORS,
)
from experiments.common.stats import mean_ci, format_mean_std

import matplotlib.pyplot as plt

from shared.types import Side

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Single run logic
# ═══════════════════════════════════════════════════════════════════════════

def run_single_strategy(
    ctx: PipelineContext,
    strategy: str,
    seed: int,
    target_coverage: float = 0.95,
) -> dict:
    """Run one strategy on one model with one seed. Returns metrics dict."""
    model = ctx.model
    target_points, normals = ctx.sample_surface()  # fixed seed, disk-cached
    set_seed(seed)  # experiment seed for candidate generation
    og = ctx.build_sampling_og()
    vis_query = ctx.build_visibility_query("raycast")

    num_candidates = model.num_candidates

    # ── Sample candidates ────────────────────────────────────────────
    with timed() as t_sample:
        if strategy == "weighted":
            sampler = ctx.build_sampler("targeted")
            pos_gpu, rot_gpu = sampler.sample(
                cp.arange(len(target_points)),
                num_candidates, side=Side.OUTSIDE,
                curvature_weighting=False,
            )

        elif strategy == "weighted_curvature":
            sampler = ctx.build_sampler("targeted")
            pos_gpu, rot_gpu = sampler.sample(
                cp.arange(len(target_points)),
                num_candidates, side=Side.OUTSIDE,
                curvature_weighting=True,
            )

        elif strategy == "targeted_25":
            sampler = ctx.build_sampler("targeted")
            n_uniform = int(num_candidates * 0.75)
            n_targeted = num_candidates - n_uniform
            pos_gpu, rot_gpu = sampler.sample(
                cp.arange(len(target_points)),
                n_uniform, side=Side.OUTSIDE,
                curvature_weighting=False,
            )
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

        elif strategy == "targeted_50":
            sampler = ctx.build_sampler("targeted")
            n_uniform = int(num_candidates * 0.50)
            n_targeted = num_candidates - n_uniform
            pos_gpu, rot_gpu = sampler.sample(
                cp.arange(len(target_points)),
                n_uniform, side=Side.OUTSIDE,
                curvature_weighting=False,
            )
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

        elif strategy == "cmaes":
            sampler = ctx.build_sampler("targeted")
            n_uniform = int(num_candidates * 0.50)
            n_opt = num_candidates - n_uniform
            pos_gpu, rot_gpu = sampler.sample(
                cp.arange(len(target_points)),
                n_uniform, side=Side.OUTSIDE,
                curvature_weighting=False,
            )
            V_init, _ = vis_query.compute_visibility_batch(pos_gpu, rot_gpu)
            coverage_count = V_init.astype(cp.int32).sum(axis=0)

            from visibility.sampling import OptimizingSampler, CMAESBackend
            opt_sampler = OptimizingSampler(
                mesh=ctx._o3d_mesh, target_points=target_points,
                normals=normals, frustum_far=model.frustum.far,
                collision_radius=model.collision_radius,
                occupancy_grid=og,
                backend=CMAESBackend(),
                random_sampler=sampler,
            )
            opt_pos, opt_rot = opt_sampler.sample_optimized(
                n_opt, coverage_count, vis_query,
                existing_pos_gpu=pos_gpu, existing_rot_gpu=rot_gpu,
                k_coverage=1,
            )
            if len(opt_pos) > 0:
                pos_gpu = cp.concatenate([pos_gpu, opt_pos])
                rot_gpu = cp.concatenate([rot_gpu, opt_rot])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    sampling_time = t_sample.elapsed

    # ── Compute visibility ───────────────────────────────────────────
    with timed() as t_vis:
        V, _ = vis_query.compute_visibility_batch(pos_gpu, rot_gpu)
    visibility_time = t_vis.elapsed

    # ── Set cover optimization ───────────────────────────────────────
    from visibility.set_cover import LazyGreedySetCoverCuda
    with timed() as t_opt:
        optimizer = LazyGreedySetCoverCuda(
            len(target_points), pos_gpu, rot_gpu, V,
        )
        opt_result = optimizer.optimize(
            target_coverage=target_coverage,
            max_viewpoints=1000,
        )
    optimization_time = t_opt.elapsed

    return {
        "model": model.name,
        "strategy": strategy,
        "seed": seed,
        "num_candidates": int(len(pos_gpu)),
        "num_viewpoints": opt_result.num_viewpoints,
        "coverage": float(opt_result.total_coverage),
        "redundancy": float(opt_result.redundancy),
        "sampling_time": sampling_time,
        "visibility_time": visibility_time,
        "optimization_time": optimization_time,
        "total_time": sampling_time + visibility_time + optimization_time,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Plot generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_plots(results: list[dict], output_dir: str):
    """Generate all E1 figures from collected results."""
    setup_thesis_style()
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    models = sorted(set(r["model"] for r in results))
    strategies = E01_STRATEGIES

    for model_name in models:
        model_results = [r for r in results if r["model"] == model_name]
        _plot_single_model(model_results, strategies, model_name, fig_dir)

    # Cross-model comparison (if multiple models)
    if len(models) > 1:
        _plot_cross_model(results, strategies, models, fig_dir)


def _plot_single_model(results: list[dict], strategies: list[str],
                       model_name: str, fig_dir: str):
    """Generate per-model figures."""
    # Collect per-strategy data
    data = {s: [r for r in results if r["strategy"] == s] for s in strategies}

    # ── Fig 1: Grouped bar - viewpoints by strategy ──────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3.5))
    vp_means = {s: np.mean([r["num_viewpoints"] for r in data[s]]) for s in strategies}
    vp_stds = {s: np.std([r["num_viewpoints"] for r in data[s]]) for s in strategies}
    grouped_bar(ax,
                {"Viewpoints": [vp_means[s] for s in strategies]},
                strategies,
                yerr={"Viewpoints": [vp_stds[s] for s in strategies]},
                ylabel="Selected viewpoints",
                title=f"Viewpoints by Strategy ({model_name})",
                value_labels=True, fmt="%.0f")
    ax.tick_params(axis="x", rotation=25)
    save_figure(fig, os.path.join(fig_dir, f"{model_name}_e01_viewpoints"))

    # ── Fig 2: Violin - coverage distribution ────────────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3.5))
    cov_data = {s: [r["coverage"] * 100 for r in data[s]] for s in strategies}
    violin_with_swarm(ax, cov_data, ylabel="Coverage (%)",
                      title=f"Coverage Distribution ({model_name})")
    save_figure(fig, os.path.join(fig_dir, f"{model_name}_e01_coverage_violin"))

    # ── Fig 3: Stacked bar - timing breakdown ────────────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3.5))
    stacked_bar(ax, strategies,
                {
                    "Sampling": [np.mean([r["sampling_time"] for r in data[s]]) for s in strategies],
                    "Visibility": [np.mean([r["visibility_time"] for r in data[s]]) for s in strategies],
                    "Optimization": [np.mean([r["optimization_time"] for r in data[s]]) for s in strategies],
                },
                ylabel="Time (s)",
                title=f"Time Breakdown ({model_name})")
    ax.tick_params(axis="x", rotation=25)
    save_figure(fig, os.path.join(fig_dir, f"{model_name}_e01_timing"))

    # ── Fig 4: Box plot - redundancy ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3.5))
    red_data = {s: [r["redundancy"] for r in data[s]] for s in strategies}
    violin_with_swarm(ax, red_data, ylabel="Redundancy (viewpoints/point)",
                      title=f"Coverage Redundancy ({model_name})")
    save_figure(fig, os.path.join(fig_dir, f"{model_name}_e01_redundancy"))

    # ── Fig 5: Radar chart - multi-metric ────────────────────────────
    fig = plt.figure(figsize=(THESIS_COL, THESIS_COL))
    # Normalize metrics to [0, 1] (lower viewpoints = better, so invert)
    all_vps = [r["num_viewpoints"] for r in results]
    all_times = [r["total_time"] for r in results]
    all_covs = [r["coverage"] for r in results]
    all_reds = [r["redundancy"] for r in results]

    def _norm_inv(vals, v):
        """Normalize inversely (lower is better → higher score)."""
        mn, mx = min(vals), max(vals)
        return 1.0 - (v - mn) / (mx - mn + 1e-9)

    def _norm(vals, v):
        mn, mx = min(vals), max(vals)
        return (v - mn) / (mx - mn + 1e-9)

    radar_metrics = {}
    for s in strategies:
        s_data = data[s]
        radar_metrics[s] = {
            "Fewer VPs": _norm_inv(all_vps, np.mean([r["num_viewpoints"] for r in s_data])),
            "Coverage": _norm(all_covs, np.mean([r["coverage"] for r in s_data])),
            "Speed": _norm_inv(all_times, np.mean([r["total_time"] for r in s_data])),
            "Low Redund.": _norm_inv(all_reds, np.mean([r["redundancy"] for r in s_data])),
        }
    radar_chart(fig, radar_metrics, title=f"Strategy Comparison ({model_name})")
    save_figure(fig, os.path.join(fig_dir, f"{model_name}_e01_radar"))

    # ── Fig 6: CDF - per-point coverage count ────────────────────────
    # This would require per-point data; skip if not available
    logger.info("Figures for %s saved to %s", model_name, fig_dir)


def _plot_cross_model(results: list[dict], strategies: list[str],
                      models: list[str], fig_dir: str):
    """Cross-model comparison figures."""

    # ── Fig 7: Grouped bar - viewpoints across models ────────────────
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4))
    x_labels = [f"{m}\n{s}" for m in models for s in strategies]
    vp_data = {}
    for s in strategies:
        vals = []
        for m in models:
            mr = [r for r in results if r["model"] == m and r["strategy"] == s]
            vals.append(np.mean([r["num_viewpoints"] for r in mr]) if mr else 0)
        vp_data[s] = vals
    grouped_bar(ax, vp_data, models, ylabel="Selected viewpoints",
                title="Viewpoints by Strategy Across Models")
    save_figure(fig, os.path.join(fig_dir, "cross_model_e01_viewpoints"))

    # ── Fig 8: Grouped bar - coverage across models ──────────────────
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4))
    cov_data = {}
    for s in strategies:
        vals = []
        for m in models:
            mr = [r for r in results if r["model"] == m and r["strategy"] == s]
            vals.append(np.mean([r["coverage"] * 100 for r in mr]) if mr else 0)
        cov_data[s] = vals
    grouped_bar(ax, cov_data, models, ylabel="Coverage (%)",
                title="Coverage by Strategy Across Models")
    save_figure(fig, os.path.join(fig_dir, "cross_model_e01_coverage"))

    logger.info("Cross-model figures saved to %s", fig_dir)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="E1: Sampling Strategy Comparison")
    p.add_argument("--models", nargs="+",
                   default=["duke_of_lancaster"] + TOSCA_REPRESENTATIVE,
                   help="Models to evaluate")
    p.add_argument("--strategies", nargs="+", default=E01_STRATEGIES,
                   help="Strategies to compare")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_5,
                   help="Random seeds")
    p.add_argument("--target_coverage", type=float, default=0.95)
    p.add_argument("--output_dir", default=os.path.join(RESULTS_DIR, "e01_sampling_strategy"))
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--plots_only", action="store_true",
                   help="Only regenerate plots from existing results")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
    )

    raw_dir = os.path.join(args.output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    all_results = []

    if not args.plots_only:
        # Build model configs
        model_configs = []
        for name in args.models:
            if name == "duke_of_lancaster":
                model_configs.append(ModelConfig.duke_of_lancaster())
            else:
                try:
                    model_configs.append(ModelConfig.tosca(name))
                except FileNotFoundError:
                    logger.warning("Model %s not found, skipping", name)

        total = len(model_configs) * len(args.strategies) * len(args.seeds)
        run_idx = 0

        for model_cfg in model_configs:
            logger.info("=" * 60)
            logger.info("Model: %s", model_cfg.name)
            logger.info("=" * 60)

            try:
                ctx = PipelineContext(model_cfg)
                ctx.load_mesh()
                ctx.sample_surface(seed=42)
                ctx.build_sampling_og()
            except DegenerateNormalsError as e:
                logger.warning("Skipping %s: %s", model_cfg.name, e)
                continue

            for strategy in args.strategies:
                for seed in args.seeds:
                    run_idx += 1
                    result_path = os.path.join(
                        raw_dir,
                        f"model={model_cfg.name}_strategy={strategy}_seed={seed}",
                    )

                    if args.skip_existing and os.path.exists(result_path + ".json"):
                        logger.info("[%d/%d] SKIP: %s %s seed=%d",
                                    run_idx, total, model_cfg.name, strategy, seed)
                        from experiments.common.persistence import load_run_result
                        all_results.append(load_run_result(result_path))
                        continue

                    logger.info("[%d/%d] Running: %s %s seed=%d",
                                run_idx, total, model_cfg.name, strategy, seed)

                    try:
                        result = run_single_strategy(
                            ctx, strategy, seed, args.target_coverage,
                        )
                        all_results.append(result)

                        from experiments.common.persistence import save_run_result
                        save_run_result(result, result_path)

                        logger.info("  VPs=%d  cov=%.2f%%  time=%.1fs",
                                    result["num_viewpoints"],
                                    result["coverage"] * 100,
                                    result["total_time"])
                    except Exception as e:
                        logger.error("  FAILED: %s", e, exc_info=True)

    else:
        # Load existing results
        from experiments.common.persistence import load_run_result
        for fname in sorted(os.listdir(raw_dir)):
            if fname.endswith(".json"):
                all_results.append(load_run_result(
                    os.path.join(raw_dir, fname.replace(".json", ""))))

    if all_results:
        generate_plots(all_results, args.output_dir)

        # Print summary table
        logger.info("\n" + "=" * 80)
        logger.info("E1 SUMMARY")
        logger.info("=" * 80)
        models = sorted(set(r["model"] for r in all_results))
        for model_name in models:
            logger.info("\nModel: %s", model_name)
            logger.info("%-20s %10s %10s %10s %10s",
                        "Strategy", "VPs", "Coverage%", "Time(s)", "Redundancy")
            logger.info("-" * 62)
            for s in args.strategies:
                sr = [r for r in all_results
                      if r["model"] == model_name and r["strategy"] == s]
                if sr:
                    logger.info("%-20s %10s %10s %10s %10s",
                                s,
                                format_mean_std([r["num_viewpoints"] for r in sr]),
                                format_mean_std([r["coverage"] * 100 for r in sr]),
                                format_mean_std([r["total_time"] for r in sr]),
                                format_mean_std([r["redundancy"] for r in sr]))


if __name__ == "__main__":
    main()
