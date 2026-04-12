#!/usr/bin/env python3
"""E3: Visibility Method Deep Comparison

Compares raycasting (ground truth) vs epsilon-visibility (approximate)
across multiple coverage targets and models. Measures per-viewpoint IoU,
coverage gap, and timing.

Usage:
    conda run -n isaaclab python -m experiments.e03_visibility_comparison
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
    ModelConfig, SEEDS_5, E03_COVERAGE_TARGETS, TOSCA_REPRESENTATIVE, RESULTS_DIR,
)
from experiments.common.runner import set_seed, timed
from experiments.common.pipeline_setup import PipelineContext, DegenerateNormalsError
from experiments.common.persistence import save_run_result, load_run_result
from experiments.common.plotting import (
    setup_thesis_style, save_figure, grouped_bar, violin_with_swarm,
    stacked_bar, THESIS_COL, DOUBLE_COL, CATEGORICAL_COLORS,
)
from experiments.common.stats import format_mean_std

from shared.types import Side

logger = logging.getLogger(__name__)


def run_single(ctx: PipelineContext, method: str, target_coverage: float,
               seed: int) -> dict:
    target_points, normals = ctx.sample_surface()  # fixed seed, disk-cached
    set_seed(seed)  # experiment seed for candidate generation
    sampler = ctx.build_sampler("targeted")

    # Generate candidates
    num_candidates = ctx.model.num_candidates
    n_uniform = int(num_candidates * 0.50)
    n_targeted = num_candidates - n_uniform

    pos_gpu, rot_gpu = sampler.sample(
        cp.arange(len(target_points)), n_uniform,
        side=Side.OUTSIDE, curvature_weighting=False,
    )

    # Use raycast for initial targeted sampling regardless of method
    rc_query = ctx.build_visibility_query("raycast")
    V_init, _ = rc_query.compute_visibility_batch(pos_gpu, rot_gpu)
    coverage_count = V_init.astype(cp.int32).sum(axis=0)
    uncovered = cp.where(coverage_count < 1)[0]
    if len(uncovered) > 0 and n_targeted > 0:
        t_pos, t_rot = sampler.sample(
            uncovered, n_targeted, side=Side.OUTSIDE, curvature_weighting=False,
        )
        pos_gpu = cp.concatenate([pos_gpu, t_pos])
        rot_gpu = cp.concatenate([rot_gpu, t_rot])

    # Build visibility query for the method under test
    vis_query = ctx.build_visibility_query(method)

    with timed() as t_vis:
        V, _ = vis_query.compute_visibility_batch(pos_gpu, rot_gpu)

    from visibility.set_cover import LazyGreedySetCoverCuda
    with timed() as t_opt:
        optimizer = LazyGreedySetCoverCuda(len(target_points), pos_gpu, rot_gpu, V)
        opt_result = optimizer.optimize(target_coverage=target_coverage, max_viewpoints=1000)

    reported_coverage = float(opt_result.total_coverage)

    # Cross-validate with raycast ground truth (if method is epsilon)
    actual_coverage = reported_coverage
    per_vp_ious = []
    if method == "epsilon":
        total_visible_mask = np.zeros(len(target_points), dtype=bool)
        positions_np = opt_result.positions.get()
        rotations_np = opt_result.rotations.get()
        from visibility.visibility.raycast import RaycastingVisibilityQuery
        from visibility.core.types import FrustumParams
        frustum = FrustumParams(
            fov_y=ctx.model.frustum.fov_y_rad,
            aspect=ctx.model.frustum.aspect,
            near=ctx.model.frustum.near,
            far=ctx.model.frustum.far,
        )
        rc_cpu = RaycastingVisibilityQuery(
            mesh=ctx._o3d_mesh,
            target_points=cp.asnumpy(target_points),
            normals=cp.asnumpy(normals),
            frustum_params=frustum,
        )
        for i in range(opt_result.num_viewpoints):
            vis_rc, _ = rc_cpu.compute_visibility(positions_np[i], rotations_np[i])
            total_visible_mask[vis_rc] = True

            # Per-VP IoU
            eps_vis = set(np.where(opt_result.visibility_map[i].get())[0])
            rc_vis = set(vis_rc)
            intersection = len(eps_vis & rc_vis)
            union = len(eps_vis | rc_vis)
            iou = intersection / union if union > 0 else 1.0
            per_vp_ious.append(iou)

        actual_coverage = float(total_visible_mask.sum() / len(target_points))

    return {
        "model": ctx.model.name,
        "method": method,
        "target_coverage": target_coverage,
        "seed": seed,
        "num_viewpoints": opt_result.num_viewpoints,
        "reported_coverage": reported_coverage,
        "actual_coverage": actual_coverage,
        "coverage_gap": reported_coverage - actual_coverage,
        "visibility_time": t_vis.elapsed,
        "optimization_time": t_opt.elapsed,
        "redundancy": float(opt_result.redundancy),
        "mean_vp_iou": float(np.mean(per_vp_ious)) if per_vp_ious else 1.0,
        "num_candidates": int(len(pos_gpu)),
    }


def generate_plots(results: list[dict], output_dir: str):
    setup_thesis_style()
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    models = sorted(set(r["model"] for r in results))
    methods = ["raycast", "epsilon"]
    targets = sorted(set(r["target_coverage"] for r in results))
    target_labels = [f"{t*100:.0f}%" for t in targets]

    for model_name in models:
        mr = [r for r in results if r["model"] == model_name]

        # ── Fig 1: Grouped bar - viewpoints by method per target ─────
        fig, ax = plt.subplots(figsize=(THESIS_COL, 3.5))
        data = {}
        for m in methods:
            data[m] = [np.mean([r["num_viewpoints"] for r in mr
                                if r["method"] == m and r["target_coverage"] == t])
                       for t in targets]
        grouped_bar(ax, data, target_labels, ylabel="Viewpoints",
                    title=f"Viewpoints: Raycast vs Epsilon ({model_name})")
        ax.set_xlabel("Target coverage")
        save_figure(fig, os.path.join(fig_dir, f"{model_name}_e03_viewpoints"))

        # ── Fig 2: Scatter - per-VP IoU ──────────────────────────────
        eps_results = [r for r in mr if r["method"] == "epsilon"]
        if eps_results:
            fig, ax = plt.subplots(figsize=(THESIS_COL, 3))
            ious = [r["mean_vp_iou"] for r in eps_results]
            targets_for_scatter = [r["target_coverage"] * 100 for r in eps_results]
            ax.scatter(targets_for_scatter, ious, alpha=0.6,
                       color=CATEGORICAL_COLORS[1])
            ax.set_xlabel("Target coverage (%)")
            ax.set_ylabel("Mean per-VP IoU")
            ax.set_title(f"Epsilon vs Raycast IoU ({model_name})")
            ax.set_ylim(0, 1.05)
            save_figure(fig, os.path.join(fig_dir, f"{model_name}_e03_iou"))

        # ── Fig 3: Stacked bar - timing ──────────────────────────────
        fig, ax = plt.subplots(figsize=(THESIS_COL, 3.5))
        stacked_bar(ax, target_labels, {
            f"Vis ({m})": [np.mean([r["visibility_time"] for r in mr
                                    if r["method"] == m and r["target_coverage"] == t])
                           for t in targets]
            for m in methods
        }, ylabel="Time (s)", title=f"Timing ({model_name})")
        save_figure(fig, os.path.join(fig_dir, f"{model_name}_e03_timing"))

        # ── Fig 4: Box plot - coverage gap ───────────────────────────
        if eps_results:
            fig, ax = plt.subplots(figsize=(THESIS_COL, 3))
            gap_data = {f"{t*100:.0f}%": [r["coverage_gap"] * 100 for r in eps_results
                                           if r["target_coverage"] == t]
                        for t in targets}
            violin_with_swarm(ax, gap_data,
                              ylabel="Coverage gap (reported - actual) %",
                              title=f"Epsilon Coverage Gap ({model_name})")
            save_figure(fig, os.path.join(fig_dir, f"{model_name}_e03_gap"))

        # ── Fig 5: Line - actual vs target ───────────────────────────
        fig, ax = plt.subplots(figsize=(THESIS_COL, 3))
        for i, m in enumerate(methods):
            actual = [np.mean([r["actual_coverage"] * 100 for r in mr
                               if r["method"] == m and r["target_coverage"] == t])
                      for t in targets]
            ax.plot([t * 100 for t in targets], actual, "o-",
                    color=CATEGORICAL_COLORS[i], label=m)
        ax.plot([t * 100 for t in targets], [t * 100 for t in targets],
                "k--", alpha=0.3, label="Ideal")
        ax.set_xlabel("Target coverage (%)")
        ax.set_ylabel("Actual coverage (%)")
        ax.set_title(f"Actual vs Target ({model_name})")
        ax.legend()
        save_figure(fig, os.path.join(fig_dir, f"{model_name}_e03_actual_vs_target"))

        # ── Fig 6: Grouped bar - redundancy ──────────────────────────
        fig, ax = plt.subplots(figsize=(THESIS_COL, 3.5))
        red_data = {}
        for m in methods:
            red_data[m] = [np.mean([r["redundancy"] for r in mr
                                    if r["method"] == m and r["target_coverage"] == t])
                           for t in targets]
        grouped_bar(ax, red_data, target_labels, ylabel="Redundancy",
                    title=f"Redundancy ({model_name})")
        save_figure(fig, os.path.join(fig_dir, f"{model_name}_e03_redundancy"))

    # Cross-model figures
    if len(models) > 1:
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.5))
        gap_data = {}
        for model_name in models:
            eps_r = [r for r in results
                     if r["model"] == model_name and r["method"] == "epsilon"]
            if eps_r:
                gap_data[model_name] = [r["coverage_gap"] * 100 for r in eps_r]
        if gap_data:
            violin_with_swarm(ax, gap_data,
                              ylabel="Coverage gap (%)",
                              title="Epsilon Coverage Gap Across Models")
            save_figure(fig, os.path.join(fig_dir, "cross_model_e03_gap"))

    logger.info("E3 figures saved to %s", fig_dir)


def main():
    p = argparse.ArgumentParser(description="E3: Visibility Method Comparison")
    p.add_argument("--models", nargs="+",
                   default=["duke_of_lancaster"] + TOSCA_REPRESENTATIVE)
    p.add_argument("--methods", nargs="+", default=["raycast", "epsilon"])
    p.add_argument("--targets", type=float, nargs="+", default=E03_COVERAGE_TARGETS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_5)
    p.add_argument("--output_dir", default=os.path.join(RESULTS_DIR, "e03_visibility_comparison"))
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
        for model_name in args.models:
            model_cfg = (ModelConfig.duke_of_lancaster() if model_name == "duke_of_lancaster"
                         else ModelConfig.tosca(model_name))
            try:
                ctx = PipelineContext(model_cfg)
                ctx.load_mesh()
                ctx.sample_surface(seed=42)
                ctx.build_sampling_og()
            except DegenerateNormalsError:
                logger.warning("Skipping %s: degenerate normals", model_name)
                continue

            for method in args.methods:
                for target in args.targets:
                    for seed in args.seeds:
                        rpath = os.path.join(raw_dir,
                            f"model={model_name}_method={method}_target={target}_seed={seed}")
                        if args.skip_existing and os.path.exists(rpath + ".json"):
                            all_results.append(load_run_result(rpath))
                            continue

                        logger.info("Running: %s %s target=%.2f seed=%d",
                                    model_name, method, target, seed)
                        try:
                            result = run_single(ctx, method, target, seed)
                            all_results.append(result)
                            save_run_result(result, rpath)
                            logger.info("  VPs=%d actual_cov=%.2f%% gap=%.3f%%",
                                        result["num_viewpoints"],
                                        result["actual_coverage"] * 100,
                                        result["coverage_gap"] * 100)
                        except Exception as e:
                            logger.error("  FAILED: %s", e, exc_info=True)
    else:
        for fname in sorted(os.listdir(raw_dir)):
            if fname.endswith(".json"):
                all_results.append(load_run_result(
                    os.path.join(raw_dir, fname.replace(".json", ""))))

    if all_results:
        generate_plots(all_results, args.output_dir)


if __name__ == "__main__":
    main()
