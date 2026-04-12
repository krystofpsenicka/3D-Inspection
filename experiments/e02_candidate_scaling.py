#!/usr/bin/env python3
"""E2: Candidate Count Scaling

Measures how the number of candidate viewpoints affects solution quality
and computation time. Sweeps num_candidates from 250 to 5000.

Usage:
    conda run -n isaaclab python -m experiments.e02_candidate_scaling
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
    ModelConfig, SEEDS_5, E02_CANDIDATE_COUNTS, RESULTS_DIR,
)
from experiments.common.runner import set_seed, timed
from experiments.common.pipeline_setup import PipelineContext
from experiments.common.persistence import save_run_result, load_run_result
from experiments.common.plotting import (
    setup_thesis_style, save_figure, log_log_with_fit, dual_yaxis,
    THESIS_COL, CATEGORICAL_COLORS,
)
from experiments.common.stats import format_mean_std

from shared.types import Side

logger = logging.getLogger(__name__)


def run_single(ctx: PipelineContext, num_candidates: int, seed: int,
               target_coverage: float = 0.95) -> dict:
    target_points, normals = ctx.sample_surface()  # fixed seed, disk-cached
    set_seed(seed)  # experiment seed for candidate generation
    vis_query = ctx.build_visibility_query("raycast")
    sampler = ctx.build_sampler("targeted")

    n_uniform = int(num_candidates * 0.50)
    n_targeted = num_candidates - n_uniform

    with timed() as t_sample:
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

    with timed() as t_vis:
        V, _ = vis_query.compute_visibility_batch(pos_gpu, rot_gpu)

    from visibility.set_cover import LazyGreedySetCoverCuda
    with timed() as t_opt:
        optimizer = LazyGreedySetCoverCuda(len(target_points), pos_gpu, rot_gpu, V)
        opt_result = optimizer.optimize(target_coverage=target_coverage, max_viewpoints=1000)

    return {
        "model": ctx.model.name,
        "num_candidates": int(len(pos_gpu)),
        "num_candidates_requested": num_candidates,
        "seed": seed,
        "num_viewpoints": opt_result.num_viewpoints,
        "coverage": float(opt_result.total_coverage),
        "redundancy": float(opt_result.redundancy),
        "sampling_time": t_sample.elapsed,
        "visibility_time": t_vis.elapsed,
        "optimization_time": t_opt.elapsed,
        "total_time": t_sample.elapsed + t_vis.elapsed + t_opt.elapsed,
    }


def generate_plots(results: list[dict], output_dir: str):
    setup_thesis_style()
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    candidates = sorted(set(r["num_candidates_requested"] for r in results))

    # Aggregate per candidate count
    def _agg(metric):
        means, stds = [], []
        for nc in candidates:
            vals = [r[metric] for r in results if r["num_candidates_requested"] == nc]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        return np.array(means), np.array(stds)

    vp_m, vp_s = _agg("num_viewpoints")
    cov_m, cov_s = _agg("coverage")
    vis_m, vis_s = _agg("visibility_time")
    tot_m, tot_s = _agg("total_time")

    # ── Fig 1: Line - viewpoints vs candidates ───────────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3))
    ax.errorbar(candidates, vp_m, yerr=vp_s, marker="o", capsize=3,
                color=CATEGORICAL_COLORS[0])
    ax.fill_between(candidates, vp_m - vp_s, vp_m + vp_s, alpha=0.15,
                    color=CATEGORICAL_COLORS[0])
    ax.set_xlabel("Number of candidates")
    ax.set_ylabel("Selected viewpoints")
    ax.set_title("Viewpoints vs. Candidate Count")
    save_figure(fig, os.path.join(fig_dir, "e02_viewpoints_vs_candidates"))

    # ── Fig 2: Log-log - visibility time vs candidates ───────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3))
    log_log_with_fit(ax, candidates, vis_m, label="Visibility time")
    ax.set_xlabel("Number of candidates")
    ax.set_ylabel("Time (s)")
    ax.set_title("Visibility Computation Time (log-log)")
    save_figure(fig, os.path.join(fig_dir, "e02_visibility_time_loglog"))

    # ── Fig 3: Line - coverage vs candidates ─────────────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3))
    ax.errorbar(candidates, cov_m * 100, yerr=cov_s * 100, marker="s",
                capsize=3, color=CATEGORICAL_COLORS[1])
    ax.set_xlabel("Number of candidates")
    ax.set_ylabel("Coverage (%)")
    ax.set_title("Achieved Coverage vs. Candidate Count")
    ax.axhline(95, color="red", linestyle="--", alpha=0.5, label="95% target")
    ax.legend()
    save_figure(fig, os.path.join(fig_dir, "e02_coverage_vs_candidates"))

    # ── Fig 4: Dual y-axis - viewpoints + time vs candidates ────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3))
    dual_yaxis(ax, candidates, vp_m, tot_m,
               "Viewpoints", "Total time",
               ylabel1="Selected viewpoints", ylabel2="Time (s)",
               title="Viewpoints & Time vs. Candidates")
    save_figure(fig, os.path.join(fig_dir, "e02_dual_axis"))

    logger.info("E2 figures saved to %s", fig_dir)


def main():
    p = argparse.ArgumentParser(description="E2: Candidate Count Scaling")
    p.add_argument("--model", default="duke_of_lancaster")
    p.add_argument("--candidates", type=int, nargs="+", default=E02_CANDIDATE_COUNTS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_5)
    p.add_argument("--target_coverage", type=float, default=0.95)
    p.add_argument("--output_dir", default=os.path.join(RESULTS_DIR, "e02_candidate_scaling"))
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

        total = len(args.candidates) * len(args.seeds)
        for i, nc in enumerate(args.candidates):
            for j, seed in enumerate(args.seeds):
                idx = i * len(args.seeds) + j + 1
                rpath = os.path.join(raw_dir, f"candidates={nc}_seed={seed}")

                if args.skip_existing and os.path.exists(rpath + ".json"):
                    logger.info("[%d/%d] SKIP", idx, total)
                    all_results.append(load_run_result(rpath))
                    continue

                logger.info("[%d/%d] candidates=%d seed=%d", idx, total, nc, seed)
                result = run_single(ctx, nc, seed, args.target_coverage)
                all_results.append(result)
                save_run_result(result, rpath)
                logger.info("  VPs=%d cov=%.2f%% time=%.1fs",
                            result["num_viewpoints"], result["coverage"] * 100,
                            result["total_time"])
    else:
        for fname in sorted(os.listdir(raw_dir)):
            if fname.endswith(".json"):
                all_results.append(load_run_result(
                    os.path.join(raw_dir, fname.replace(".json", ""))))

    if all_results:
        generate_plots(all_results, args.output_dir)


if __name__ == "__main__":
    main()
