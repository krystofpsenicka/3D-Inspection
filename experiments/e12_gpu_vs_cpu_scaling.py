#!/usr/bin/env python3
"""E12: GPU vs CPU Scaling

Systematic GPU speedup measurement across problem sizes. Compares GPU and CPU
implementations for frustum culling, visibility computation, and set cover
optimization.

Parameters:
  num_points    {10000, 50000, 100000, 200000}
  num_candidates {100, 500, 1000, 2000}
  3 seeds

Usage:
    conda run -n isaaclab python -m experiments.e12_gpu_vs_cpu_scaling
    conda run -n isaaclab python -m experiments.e12_gpu_vs_cpu_scaling --plots_only
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
    setup_thesis_style, save_figure, grouped_bar, heatmap_annotated,
    THESIS_COL, DOUBLE_COL, CATEGORICAL_COLORS,
)

from shared.types import Side

logger = logging.getLogger(__name__)

NUM_POINTS = [10_000, 50_000, 100_000, 200_000]
NUM_CANDIDATES = [100, 500, 1000, 2000]


# ═══════════════════════════════════════════════════════════════════════════
# Single run logic
# ═══════════════════════════════════════════════════════════════════════════

def run_single(num_points: int, num_candidates: int, seed: int) -> dict:
    """Run GPU vs CPU timing for one problem size."""
    set_seed(seed)
    model_cfg = ModelConfig.duke_of_lancaster()
    ctx = PipelineContext(model_cfg)
    ctx.load_mesh()

    target_points, normals = ctx.sample_surface(num_points=num_points, seed=seed)
    ctx.build_sampling_og()
    sampler = ctx.build_sampler("targeted")

    # Generate candidates
    pos_gpu, rot_gpu = sampler.sample(
        cp.arange(len(target_points)), num_candidates,
        side=Side.OUTSIDE, curvature_weighting=False,
    )

    # ── GPU visibility (raycast CUDA) ────────────────────────────────
    vis_gpu = ctx.build_visibility_query("raycast")
    with timed() as t_vis_gpu:
        V_gpu, _ = vis_gpu.compute_visibility_batch(pos_gpu, rot_gpu)
    gpu_vis_time = t_vis_gpu.elapsed

    # ── CPU visibility (epsilon approximation as CPU proxy) ──────────
    vis_cpu = ctx.build_visibility_query("epsilon")
    with timed() as t_vis_cpu:
        V_cpu, _ = vis_cpu.compute_visibility_batch(pos_gpu, rot_gpu)
    cpu_vis_time = t_vis_cpu.elapsed

    # ── GPU set cover ────────────────────────────────────────────────
    from visibility.set_cover import LazyGreedySetCoverCuda
    with timed() as t_sc_gpu:
        opt_gpu = LazyGreedySetCoverCuda(
            len(target_points), pos_gpu, rot_gpu, V_gpu,
        )
        res_gpu = opt_gpu.optimize(target_coverage=0.95, max_viewpoints=500)
    gpu_sc_time = t_sc_gpu.elapsed

    # ── CPU set cover ────────────────────────────────────────────────
    try:
        from visibility.set_cover import LazyGreedySetCover
        V_np = V_gpu.get() if hasattr(V_gpu, "get") else np.asarray(V_gpu)
        pos_np = pos_gpu.get() if hasattr(pos_gpu, "get") else np.asarray(pos_gpu)
        rot_np = rot_gpu.get() if hasattr(rot_gpu, "get") else np.asarray(rot_gpu)
        with timed() as t_sc_cpu:
            opt_cpu = LazyGreedySetCover(
                len(target_points), pos_np, rot_np, V_np,
            )
            res_cpu = opt_cpu.optimize(target_coverage=0.95, max_viewpoints=500)
        cpu_sc_time = t_sc_cpu.elapsed
    except Exception as e:
        logger.warning("CPU set cover failed: %s", e)
        cpu_sc_time = float("nan")

    vis_speedup = cpu_vis_time / gpu_vis_time if gpu_vis_time > 0 else float("nan")
    sc_speedup = cpu_sc_time / gpu_sc_time if gpu_sc_time > 0 else float("nan")

    return {
        "num_points": num_points,
        "num_candidates": num_candidates,
        "seed": seed,
        "gpu_vis_time": gpu_vis_time,
        "cpu_vis_time": cpu_vis_time,
        "vis_speedup": vis_speedup,
        "gpu_sc_time": gpu_sc_time,
        "cpu_sc_time": cpu_sc_time,
        "sc_speedup": sc_speedup,
        "gpu_total": gpu_vis_time + gpu_sc_time,
        "cpu_total": cpu_vis_time + cpu_sc_time,
        "total_speedup": (cpu_vis_time + cpu_sc_time) / (gpu_vis_time + gpu_sc_time)
        if (gpu_vis_time + gpu_sc_time) > 0 else float("nan"),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Plot generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_plots(results: list[dict], output_dir: str):
    """Generate all E12 figures."""
    setup_thesis_style()
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    if not results:
        logger.warning("No results for plotting")
        return

    points_list = sorted(set(r["num_points"] for r in results))
    cands_list = sorted(set(r["num_candidates"] for r in results))

    # ── Fig 1: Heatmap - total speedup ──────────────────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 4))
    point_labels = [f"{p//1000}K" for p in points_list]
    cand_labels = [str(c) for c in cands_list]
    hm_vals = np.zeros((len(points_list), len(cands_list)))
    for i, np_ in enumerate(points_list):
        for j, nc in enumerate(cands_list):
            vals = [r["total_speedup"] for r in results
                    if r["num_points"] == np_ and r["num_candidates"] == nc
                    and not np.isnan(r["total_speedup"])]
            hm_vals[i, j] = np.mean(vals) if vals else 0
    heatmap_annotated(ax, point_labels, cand_labels, hm_vals, fmt=".1f",
                      title="GPU Speedup (total)",
                      xlabel="Candidates", ylabel="Surface points")
    save_figure(fig, os.path.join(fig_dir, "e12_heatmap_speedup"))

    # ── Fig 2: Line - speedup vs problem size ───────────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3.5))
    for ci, nc in enumerate(cands_list):
        xs, ys = [], []
        for np_ in points_list:
            vals = [r["total_speedup"] for r in results
                    if r["num_points"] == np_ and r["num_candidates"] == nc
                    and not np.isnan(r["total_speedup"])]
            if vals:
                xs.append(np_)
                ys.append(np.mean(vals))
        if xs:
            ax.plot(xs, ys, "o-",
                    color=CATEGORICAL_COLORS[ci % len(CATEGORICAL_COLORS)],
                    label=f"{nc} cands")
    ax.set_xlabel("Number of surface points")
    ax.set_ylabel("Speedup (CPU/GPU)")
    ax.set_title("GPU Speedup vs Problem Size")
    ax.legend(fontsize=7)
    save_figure(fig, os.path.join(fig_dir, "e12_speedup_vs_size"))

    # ── Fig 3: Grouped bar - per-component speedup ──────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3.5))
    # Average across all configurations
    vis_speedups = [r["vis_speedup"] for r in results if not np.isnan(r["vis_speedup"])]
    sc_speedups = [r["sc_speedup"] for r in results if not np.isnan(r["sc_speedup"])]
    components = ["Visibility", "Set Cover"]
    means = [np.mean(vis_speedups) if vis_speedups else 0,
             np.mean(sc_speedups) if sc_speedups else 0]
    stds = [np.std(vis_speedups) if vis_speedups else 0,
            np.std(sc_speedups) if sc_speedups else 0]
    x = np.arange(len(components))
    bars = ax.bar(x, means, 0.5, yerr=stds, capsize=3,
                  color=[CATEGORICAL_COLORS[0], CATEGORICAL_COLORS[1]])
    ax.bar_label(bars, fmt="%.1fx", fontsize=8, padding=3)
    ax.set_xticks(x)
    ax.set_xticklabels(components)
    ax.set_ylabel("Speedup (CPU/GPU)")
    ax.set_title("Per-Component GPU Speedup")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    save_figure(fig, os.path.join(fig_dir, "e12_component_speedup"))

    logger.info("E12 figures saved to %s", fig_dir)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="E12: GPU vs CPU Scaling")
    p.add_argument("--num_points", type=int, nargs="+", default=NUM_POINTS)
    p.add_argument("--num_candidates", type=int, nargs="+", default=NUM_CANDIDATES)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_3)
    p.add_argument("--output_dir",
                   default=os.path.join(RESULTS_DIR, "e12_gpu_vs_cpu_scaling"))
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
        total = len(args.num_points) * len(args.num_candidates) * len(args.seeds)
        run_idx = 0

        for np_ in args.num_points:
            for nc in args.num_candidates:
                for seed in args.seeds:
                    run_idx += 1
                    rpath = os.path.join(
                        raw_dir,
                        f"points={np_}_cands={nc}_seed={seed}",
                    )
                    logger.info("[%d/%d] points=%d cands=%d seed=%d",
                                run_idx, total, np_, nc, seed)

                    try:
                        result = run_single(np_, nc, seed)
                        all_results.append(result)
                        save_run_result(result, rpath)
                        logger.info("  vis_speedup=%.1fx sc_speedup=%.1fx total=%.1fx",
                                    result["vis_speedup"], result["sc_speedup"],
                                    result["total_speedup"])
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
