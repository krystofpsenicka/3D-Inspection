#!/usr/bin/env python3
"""E5: Set Cover Scaling

Measures how optimization time scales with problem size
(candidates x surface points).

Usage:
    conda run -n isaaclab python -m experiments.e05_set_cover_scaling
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.common.config import (
    ModelConfig, SEEDS_3, E05_CANDIDATE_COUNTS, E05_POINT_COUNTS, RESULTS_DIR,
)
from experiments.common.runner import set_seed, timed, free_gpu_memory
from experiments.common.pipeline_setup import PipelineContext
from experiments.common.persistence import save_run_result, load_run_result
from experiments.common.plotting import (
    setup_thesis_style, save_figure, log_log_with_fit, heatmap_annotated,
    THESIS_COL, CATEGORICAL_COLORS,
)

from shared.types import Side

logger = logging.getLogger(__name__)


def run_single(ctx: PipelineContext, num_candidates: int,
               num_points: int, seed: int) -> dict:
    set_seed(seed)
    target_points, normals = ctx.sample_surface(num_points=num_points)  # fixed seed
    set_seed(seed)  # experiment seed for candidate generation
    sampler = ctx.build_sampler("targeted")
    vis_query = ctx.build_visibility_query("raycast")

    pos_gpu, rot_gpu = sampler.sample(
        cp.arange(len(target_points)), num_candidates,
        side=Side.OUTSIDE, curvature_weighting=False,
    )

    with timed() as t_vis:
        V, _ = vis_query.compute_visibility_batch(pos_gpu, rot_gpu)

    from visibility.set_cover import LazyGreedySetCover  # CPU — fastest per e04 results
    V_np = cp.asnumpy(V)
    pos_np = cp.asnumpy(pos_gpu)
    rot_np = cp.asnumpy(rot_gpu)
    with timed() as t_opt:
        optimizer = LazyGreedySetCover(len(target_points), pos_np, rot_np, V_np)
        opt_result = optimizer.optimize(target_coverage=0.95, max_viewpoints=1000)

    return {
        "num_candidates": int(len(pos_gpu)),
        "num_candidates_requested": num_candidates,
        "num_points": num_points,
        "seed": seed,
        "num_viewpoints": opt_result.num_viewpoints,
        "coverage": float(opt_result.total_coverage),
        "visibility_time": t_vis.elapsed,
        "optimization_time": t_opt.elapsed,
    }


def generate_plots(results: list[dict], output_dir: str):
    setup_thesis_style()
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    candidates = sorted(set(r["num_candidates_requested"] for r in results))
    point_counts = sorted(set(r["num_points"] for r in results))

    # ── Fig 1: Log-log - opt time vs candidates ─────────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3))
    for i, np_ in enumerate(point_counts):
        x, y = [], []
        for nc in candidates:
            vals = [r["optimization_time"] for r in results
                    if r["num_candidates_requested"] == nc and r["num_points"] == np_]
            if vals:
                x.append(nc)
                y.append(np.mean(vals))
        if x:
            log_log_with_fit(ax, x, y, label=f"{np_//1000}K pts",
                             color=CATEGORICAL_COLORS[i])
    ax.set_xlabel("Number of candidates")
    ax.set_ylabel("Optimization time (s)")
    ax.set_title("Set Cover Scaling (candidates)")
    save_figure(fig, os.path.join(fig_dir, "e05_time_vs_candidates"))

    # ── Fig 2: Heatmap - candidates x points -> time ────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3.5))
    vals = np.zeros((len(candidates), len(point_counts)))
    for i, nc in enumerate(candidates):
        for j, np_ in enumerate(point_counts):
            t = [r["optimization_time"] for r in results
                 if r["num_candidates_requested"] == nc and r["num_points"] == np_]
            vals[i, j] = np.mean(t) if t else 0
    heatmap_annotated(ax, [str(c) for c in candidates],
                      [f"{p//1000}K" for p in point_counts],
                      vals, fmt=".2f",
                      title="Optimization Time (s)",
                      xlabel="Surface points", ylabel="Candidates")
    save_figure(fig, os.path.join(fig_dir, "e05_heatmap"))

    # ── Fig 3: Log-log - opt time vs points ──────────────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3))
    for i, nc in enumerate(candidates):
        x, y = [], []
        for np_ in point_counts:
            vals = [r["optimization_time"] for r in results
                    if r["num_candidates_requested"] == nc and r["num_points"] == np_]
            if vals:
                x.append(np_)
                y.append(np.mean(vals))
        if x:
            log_log_with_fit(ax, x, y, label=f"{nc} cands",
                             color=CATEGORICAL_COLORS[i])
    ax.set_xlabel("Number of surface points")
    ax.set_ylabel("Optimization time (s)")
    ax.set_title("Set Cover Scaling (points)")
    save_figure(fig, os.path.join(fig_dir, "e05_time_vs_points"))

    logger.info("E5 figures saved to %s", fig_dir)


def main():
    p = argparse.ArgumentParser(description="E5: Set Cover Scaling")
    p.add_argument("--candidates", type=int, nargs="+", default=E05_CANDIDATE_COUNTS)
    p.add_argument("--point_counts", type=int, nargs="+", default=E05_POINT_COUNTS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_3)
    p.add_argument("--output_dir", default=os.path.join(RESULTS_DIR, "e05_set_cover_scaling"))
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--plots_only", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)-8s %(name)s: %(message)s")

    raw_dir = os.path.join(args.output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    all_results = []

    if not args.plots_only:
        model_cfg = ModelConfig.duke_of_lancaster()
        ctx = PipelineContext(model_cfg)
        ctx.load_mesh()
        ctx.build_sampling_og()

        total = len(args.candidates) * len(args.point_counts) * len(args.seeds)
        idx = 0
        for nc in args.candidates:
            for np_ in args.point_counts:
                for seed in args.seeds:
                    idx += 1
                    rpath = os.path.join(raw_dir, f"cands={nc}_pts={np_}_seed={seed}")
                    if args.skip_existing and os.path.exists(rpath + ".json"):
                        all_results.append(load_run_result(rpath))
                        continue

                    logger.info("[%d/%d] cands=%d pts=%d seed=%d", idx, total, nc, np_, seed)
                    try:
                        result = run_single(ctx, nc, np_, seed)
                        all_results.append(result)
                        save_run_result(result, rpath)
                        logger.info("  VPs=%d time=%.2fs", result["num_viewpoints"],
                                    result["optimization_time"])
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
        generate_plots(all_results, args.output_dir)


if __name__ == "__main__":
    main()
