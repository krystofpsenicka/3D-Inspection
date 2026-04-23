#!/usr/bin/env python3
"""E14: Curvature Sensitivity

Curvature weighting parameter sensitivity study. Sweeps curvature_knn_k
(number of neighbors for curvature estimation) and position_weight
(relative weight of position vs curvature in sampling).

Parameters:
  curvature_knn_k  {5, 10, 20, 40, 80}
  position_weight  {1.0, 2.5, 5.0, 10.0}
  3 seeds

For each combo: sample with curvature weighting, run visibility + set cover.

Usage:
    conda run -n isaaclab python -m experiments.e14_curvature_sensitivity
    conda run -n isaaclab python -m experiments.e14_curvature_sensitivity --plots_only
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
from experiments.common.runner import set_seed, timed, free_gpu_memory
from experiments.common.pipeline_setup import PipelineContext, DegenerateNormalsError
from experiments.common.persistence import save_run_result, load_run_result
from experiments.common.plotting import (
    setup_thesis_style, save_figure, heatmap_annotated,
    THESIS_COL, DOUBLE_COL, CATEGORICAL_COLORS,
)

from shared.types import Side
from visibility.set_cover import LazyGreedySetCover

logger = logging.getLogger(__name__)

KNN_K_VALUES = [5, 10, 20, 40, 80]
POSITION_WEIGHTS = [1.0, 2.5, 5.0, 10.0]
N_CANDIDATES = 2000


# ═══════════════════════════════════════════════════════════════════════════
# Single run logic
# ═══════════════════════════════════════════════════════════════════════════

def run_single(knn_k: int, position_weight: float, seed: int) -> dict:
    """Run sampling + visibility + set cover with curvature parameters."""
    set_seed(seed)
    model_cfg = ModelConfig.duke_of_lancaster()
    ctx = PipelineContext(model_cfg)
    ctx.load_mesh()
    target_points, normals = ctx.sample_surface()  # fixed seed, disk-cached
    ctx.build_sampling_og()

    sampler = ctx.build_sampler("targeted")
    vis_query = ctx.build_visibility_query("raycast")
    set_seed(seed)  # experiment seed for candidate generation

    # Sample with curvature weighting and the given parameters
    with timed() as t_sample:
        pos_gpu, rot_gpu = sampler.sample(
            cp.arange(len(target_points)), N_CANDIDATES,
            side=Side.OUTSIDE, curvature_weighting=True,
            curvature_knn_k=knn_k, position_weight=position_weight,
        )

    # Visibility
    with timed() as t_vis:
        V, _ = vis_query.compute_visibility_batch(pos_gpu, rot_gpu)

    # Set cover
    with timed() as t_opt:
        optimizer = LazyGreedySetCover(
            len(target_points), pos_gpu, rot_gpu, V,
        )
        opt_result = optimizer.optimize(
            target_coverage=0.95, max_viewpoints=1000,
        )

    return {
        "knn_k": knn_k,
        "position_weight": position_weight,
        "seed": seed,
        "num_viewpoints": opt_result.num_viewpoints,
        "coverage": float(opt_result.total_coverage),
        "redundancy": float(opt_result.redundancy),
        "sampling_time": t_sample.elapsed,
        "visibility_time": t_vis.elapsed,
        "optimization_time": t_opt.elapsed,
        "total_time": t_sample.elapsed + t_vis.elapsed + t_opt.elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Plot generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_plots(results: list[dict], output_dir: str):
    """Generate all E14 figures."""
    setup_thesis_style()
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    if not results:
        logger.warning("No results for plotting")
        return

    k_vals = sorted(set(r["knn_k"] for r in results))
    w_vals = sorted(set(r["position_weight"] for r in results))

    k_labels = [str(k) for k in k_vals]
    w_labels = [str(w) for w in w_vals]

    # ── Fig 1: Heatmap - knn_k x weight -> viewpoints ──────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 4))
    vp_vals = np.zeros((len(k_vals), len(w_vals)))
    for i, k in enumerate(k_vals):
        for j, w in enumerate(w_vals):
            vals = [r["num_viewpoints"] for r in results
                    if r["knn_k"] == k and r["position_weight"] == w]
            vp_vals[i, j] = np.mean(vals) if vals else 0
    heatmap_annotated(ax, k_labels, w_labels, vp_vals, fmt=".0f",
                      title="Viewpoints Selected",
                      xlabel="Position weight", ylabel="Curvature knn_k")
    save_figure(fig, os.path.join(fig_dir, "e14_heatmap_viewpoints"))

    # ── Fig 2: Heatmap - knn_k x weight -> coverage ────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 4))
    cov_vals = np.zeros((len(k_vals), len(w_vals)))
    for i, k in enumerate(k_vals):
        for j, w in enumerate(w_vals):
            vals = [r["coverage"] * 100 for r in results
                    if r["knn_k"] == k and r["position_weight"] == w]
            cov_vals[i, j] = np.mean(vals) if vals else 0
    heatmap_annotated(ax, k_labels, w_labels, cov_vals, fmt=".1f",
                      title="Coverage (%)",
                      xlabel="Position weight", ylabel="Curvature knn_k")
    save_figure(fig, os.path.join(fig_dir, "e14_heatmap_coverage"))

    logger.info("E14 figures saved to %s", fig_dir)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="E14: Curvature Sensitivity")
    p.add_argument("--knn_k_values", type=int, nargs="+", default=KNN_K_VALUES)
    p.add_argument("--position_weights", type=float, nargs="+", default=POSITION_WEIGHTS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_3)
    p.add_argument("--output_dir",
                   default=os.path.join(RESULTS_DIR, "e14_curvature_sensitivity"))
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
        total = len(args.knn_k_values) * len(args.position_weights) * len(args.seeds)
        run_idx = 0

        for knn_k in args.knn_k_values:
            for pw in args.position_weights:
                for seed in args.seeds:
                    run_idx += 1
                    rpath = os.path.join(
                        raw_dir,
                        f"knn_k={knn_k}_pw={pw}_seed={seed}",
                    )
                    logger.info("[%d/%d] knn_k=%d position_weight=%.1f seed=%d",
                                run_idx, total, knn_k, pw, seed)

                    try:
                        result = run_single(knn_k, pw, seed)
                        all_results.append(result)
                        save_run_result(result, rpath)
                        logger.info("  vps=%d cov=%.2f%% time=%.1fs",
                                    result["num_viewpoints"],
                                    result["coverage"] * 100,
                                    result["total_time"])
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
