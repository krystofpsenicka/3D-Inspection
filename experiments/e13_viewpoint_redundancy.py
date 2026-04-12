#!/usr/bin/env python3
"""E13: Viewpoint Redundancy (k-Coverage)

Tests the k-coverage parameter: each surface point must be visible from
at least k viewpoints. Higher k provides redundancy but requires more
viewpoints.

Parameters:
  k                {1, 2, 3, 5}
  target_coverage  {0.90, 0.95}
  3 seeds

Uses targeted_50 sampling on Duke of Lancaster.

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

from experiments.common.config import ModelConfig, SEEDS_3, RESULTS_DIR
from experiments.common.runner import set_seed, timed
from experiments.common.pipeline_setup import PipelineContext, DegenerateNormalsError
from experiments.common.persistence import save_run_result, load_run_result
from experiments.common.plotting import (
    setup_thesis_style, save_figure, grouped_bar,
    THESIS_COL, DOUBLE_COL, CATEGORICAL_COLORS,
)

from shared.types import Side
from visibility.set_cover import LazyGreedySetCoverCuda

logger = logging.getLogger(__name__)

K_VALUES = [1, 2, 3, 5]
COVERAGE_TARGETS = [0.90, 0.95]
N_CANDIDATES = 1500


# ═══════════════════════════════════════════════════════════════════════════
# Single run logic
# ═══════════════════════════════════════════════════════════════════════════

def run_single(k_coverage: int, target_coverage: float, seed: int) -> dict:
    """Run visibility + set cover with k-coverage constraint."""
    set_seed(seed)
    model_cfg = ModelConfig.duke_of_lancaster()
    ctx = PipelineContext(model_cfg)
    ctx.load_mesh()
    target_points, normals = ctx.sample_surface()  # fixed seed, disk-cached
    ctx.build_sampling_og()
    set_seed(seed)  # experiment seed for candidate generation

    # Targeted_50 sampling
    sampler = ctx.build_sampler("targeted")
    vis_query = ctx.build_visibility_query("raycast")

    n_uniform = int(N_CANDIDATES * 0.50)
    n_targeted = N_CANDIDATES - n_uniform

    pos_gpu, rot_gpu = sampler.sample(
        cp.arange(len(target_points)), n_uniform,
        side=Side.OUTSIDE, curvature_weighting=False,
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

    # Full visibility
    with timed() as t_vis:
        V, _ = vis_query.compute_visibility_batch(pos_gpu, rot_gpu)

    # Set cover with k-coverage
    with timed() as t_opt:
        optimizer = LazyGreedySetCoverCuda(
            len(target_points), pos_gpu, rot_gpu, V,
        )
        opt_result = optimizer.optimize(
            target_coverage=target_coverage,
            max_viewpoints=1000,
            k_coverage=k_coverage,
        )

    # Per-point coverage count histogram
    selected_indices = opt_result.selected_indices
    if hasattr(selected_indices, "get"):
        selected_indices = selected_indices.get()
    V_selected = V[selected_indices]
    per_point_count = V_selected.astype(cp.int32).sum(axis=0)
    if hasattr(per_point_count, "get"):
        per_point_count = per_point_count.get()
    per_point_count = np.asarray(per_point_count)

    # Histogram bins 0..max_count
    max_count = int(per_point_count.max()) if len(per_point_count) > 0 else 0
    hist_counts = np.bincount(per_point_count, minlength=max_count + 1).tolist()

    return {
        "k_coverage": k_coverage,
        "target_coverage": target_coverage,
        "seed": seed,
        "num_viewpoints": opt_result.num_viewpoints,
        "coverage": float(opt_result.total_coverage),
        "redundancy": float(opt_result.redundancy),
        "visibility_time": t_vis.elapsed,
        "optimization_time": t_opt.elapsed,
        "per_point_hist": hist_counts,
        "mean_point_coverage": float(per_point_count.mean()),
        "min_point_coverage": int(per_point_count.min()),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Plot generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_plots(results: list[dict], output_dir: str):
    """Generate all E13 figures."""
    setup_thesis_style()
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    if not results:
        logger.warning("No results for plotting")
        return

    k_vals = sorted(set(r["k_coverage"] for r in results))
    targets = sorted(set(r["target_coverage"] for r in results))

    # ── Fig 1: Line - viewpoints vs k ───────────────────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3.5))
    for ti, target in enumerate(targets):
        xs, means, stds = [], [], []
        for k in k_vals:
            vals = [r["num_viewpoints"] for r in results
                    if r["k_coverage"] == k and r["target_coverage"] == target]
            if vals:
                xs.append(k)
                means.append(np.mean(vals))
                stds.append(np.std(vals))
        if xs:
            ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3,
                        color=CATEGORICAL_COLORS[ti % len(CATEGORICAL_COLORS)],
                        label=f"target={target*100:.0f}%")
    ax.set_xlabel("k-coverage")
    ax.set_ylabel("Selected viewpoints")
    ax.set_title("Viewpoints vs k-Coverage")
    ax.set_xticks(k_vals)
    ax.legend()
    save_figure(fig, os.path.join(fig_dir, "e13_viewpoints_vs_k"))

    # ── Fig 2: Histogram overlay - per-point coverage count ─────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3.5))
    # Pick target=0.95 for histogram overlay
    target_for_hist = 0.95 if 0.95 in targets else targets[-1]
    for ki, k in enumerate(k_vals):
        k_results = [r for r in results
                     if r["k_coverage"] == k and r["target_coverage"] == target_for_hist]
        if k_results:
            # Average histograms across seeds
            max_len = max(len(r["per_point_hist"]) for r in k_results)
            padded = np.zeros((len(k_results), max_len))
            for i, r in enumerate(k_results):
                h = r["per_point_hist"]
                padded[i, :len(h)] = h
            mean_hist = padded.mean(axis=0)
            bins = np.arange(len(mean_hist))
            ax.bar(bins + ki * 0.2, mean_hist / mean_hist.sum(), width=0.18,
                   alpha=0.7, label=f"k={k}",
                   color=CATEGORICAL_COLORS[ki % len(CATEGORICAL_COLORS)])
    ax.set_xlabel("Per-point coverage count")
    ax.set_ylabel("Fraction of points")
    ax.set_title(f"Point Coverage Distribution (target={target_for_hist*100:.0f}%)")
    ax.legend(fontsize=7)
    save_figure(fig, os.path.join(fig_dir, "e13_coverage_histogram"))

    # ── Fig 3: Grouped bar - coverage achieved ──────────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3.5))
    k_labels = [f"k={k}" for k in k_vals]
    cov_data = {}
    cov_errs = {}
    for target in targets:
        means = []
        stds = []
        for k in k_vals:
            vals = [r["coverage"] * 100 for r in results
                    if r["k_coverage"] == k and r["target_coverage"] == target]
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if vals else 0)
        cov_data[f"{target*100:.0f}%"] = means
        cov_errs[f"{target*100:.0f}%"] = stds
    grouped_bar(ax, cov_data, k_labels, yerr=cov_errs,
                ylabel="Coverage (%)",
                title="Achieved Coverage by k-Coverage")
    save_figure(fig, os.path.join(fig_dir, "e13_coverage_grouped"))

    logger.info("E13 figures saved to %s", fig_dir)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="E13: Viewpoint Redundancy (k-Coverage)")
    p.add_argument("--k_values", type=int, nargs="+", default=K_VALUES)
    p.add_argument("--targets", type=float, nargs="+", default=COVERAGE_TARGETS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_3)
    p.add_argument("--output_dir",
                   default=os.path.join(RESULTS_DIR, "e13_viewpoint_redundancy"))
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
        total = len(args.k_values) * len(args.targets) * len(args.seeds)
        run_idx = 0

        for k in args.k_values:
            for target in args.targets:
                for seed in args.seeds:
                    run_idx += 1
                    rpath = os.path.join(
                        raw_dir,
                        f"k={k}_target={target}_seed={seed}",
                    )
                    logger.info("[%d/%d] k=%d target=%.2f seed=%d",
                                run_idx, total, k, target, seed)

                    try:
                        result = run_single(k, target, seed)
                        all_results.append(result)
                        save_run_result(result, rpath)
                        logger.info("  vps=%d cov=%.2f%% redundancy=%.2f",
                                    result["num_viewpoints"],
                                    result["coverage"] * 100,
                                    result["redundancy"])
                    except Exception as e:
                        logger.error("  FAILED: %s", e, exc_info=True)
    else:
        for fname in sorted(os.listdir(raw_dir)):
            if fname.endswith(".json"):
                all_results.append(
                    load_run_result(os.path.join(raw_dir, fname.replace(".json", ""))))

    if all_results:
        generate_plots(all_results, args.output_dir)


if __name__ == "__main__":
    main()
