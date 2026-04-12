#!/usr/bin/env python3
"""E15: Cross-Model Generalization

Runs the full inspection pipeline on all TOSCA models + Duke of Lancaster
to evaluate generalization. Uses targeted_50 strategy, target_coverage=0.95,
3 seeds per model.

Models: Duke of Lancaster + TOSCA_ALL (9 models).

Usage:
    conda run -n isaaclab python -m experiments.e15_cross_model
    conda run -n isaaclab python -m experiments.e15_cross_model --plots_only
    conda run -n isaaclab python -m experiments.e15_cross_model --models duke_of_lancaster wolf0 cat0
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
    ModelConfig, SEEDS_3, TOSCA_ALL, RESULTS_DIR,
)
from experiments.common.runner import set_seed, timed
from experiments.common.pipeline_setup import PipelineContext, DegenerateNormalsError
from experiments.common.persistence import save_run_result, load_run_result
from experiments.common.plotting import (
    setup_thesis_style, save_figure, grouped_bar, stacked_bar,
    THESIS_COL, DOUBLE_COL, CATEGORICAL_COLORS,
)

from shared.types import Side
from visibility.set_cover import LazyGreedySetCoverCuda

logger = logging.getLogger(__name__)

ALL_MODELS = ["duke_of_lancaster"] + TOSCA_ALL
TARGET_COVERAGE = 0.95


# ═══════════════════════════════════════════════════════════════════════════
# Single run logic
# ═══════════════════════════════════════════════════════════════════════════

def run_single(model_cfg: ModelConfig, seed: int) -> dict:
    """Run full pipeline on one model with one seed."""
    set_seed(seed)
    ctx = PipelineContext(model_cfg)

    # Mesh loading
    with timed() as t_mesh:
        ctx.load_mesh()
    tm, _ = ctx.load_mesh()
    mesh_vertices = len(tm.vertices)
    mesh_faces = len(tm.faces)

    # Surface sampling
    with timed() as t_surface:
        target_points, normals = ctx.sample_surface()  # fixed seed, disk-cached

    # OG
    with timed() as t_og:
        ctx.build_sampling_og()

    # Sampling (targeted_50)
    with timed() as t_sample:
        sampler = ctx.build_sampler("targeted")
        num_candidates = model_cfg.num_candidates
        n_uniform = int(num_candidates * 0.50)
        n_targeted = num_candidates - n_uniform
        pos_gpu, rot_gpu = sampler.sample(
            cp.arange(len(target_points)), n_uniform,
            side=Side.OUTSIDE, curvature_weighting=False,
        )
        vis_query = ctx.build_visibility_query("raycast")
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

    # Visibility
    with timed() as t_vis:
        V, _ = vis_query.compute_visibility_batch(pos_gpu, rot_gpu)

    # Set cover
    with timed() as t_opt:
        optimizer = LazyGreedySetCoverCuda(
            len(target_points), pos_gpu, rot_gpu, V,
        )
        opt_result = optimizer.optimize(
            target_coverage=TARGET_COVERAGE, max_viewpoints=1000,
        )

    return {
        "model": model_cfg.name,
        "seed": seed,
        "mesh_vertices": mesh_vertices,
        "mesh_faces": mesh_faces,
        "num_viewpoints": opt_result.num_viewpoints,
        "coverage": float(opt_result.total_coverage),
        "redundancy": float(opt_result.redundancy),
        "t_mesh": t_mesh.elapsed,
        "t_surface": t_surface.elapsed,
        "t_og": t_og.elapsed,
        "t_sample": t_sample.elapsed,
        "t_vis": t_vis.elapsed,
        "t_opt": t_opt.elapsed,
        "total_time": (t_mesh.elapsed + t_surface.elapsed + t_og.elapsed +
                       t_sample.elapsed + t_vis.elapsed + t_opt.elapsed),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Plot generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_plots(results: list[dict], output_dir: str):
    """Generate all E15 figures."""
    setup_thesis_style()
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    if not results:
        logger.warning("No results for plotting")
        return

    models = sorted(set(r["model"] for r in results),
                    key=lambda m: np.mean([r["mesh_faces"] for r in results if r["model"] == m]))

    # ── Fig 1: Grouped bar - viewpoints by model ───────────────────
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4))
    vp_data = {"Viewpoints": [np.mean([r["num_viewpoints"] for r in results if r["model"] == m])
                               for m in models]}
    vp_errs = {"Viewpoints": [np.std([r["num_viewpoints"] for r in results if r["model"] == m])
                               for m in models]}
    grouped_bar(ax, vp_data, models, yerr=vp_errs,
                ylabel="Selected viewpoints",
                title="Viewpoints by Model", value_labels=True, fmt="%.0f")
    ax.tick_params(axis="x", rotation=35)
    save_figure(fig, os.path.join(fig_dir, "e15_viewpoints_by_model"))

    # ── Fig 2: Scatter - viewpoints vs mesh faces ───────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3.5))
    for r in results:
        ax.scatter(r["mesh_faces"], r["num_viewpoints"], alpha=0.5,
                   color=CATEGORICAL_COLORS[0], s=20)
    # Model means
    for m in models:
        mr = [r for r in results if r["model"] == m]
        mean_faces = np.mean([r["mesh_faces"] for r in mr])
        mean_vps = np.mean([r["num_viewpoints"] for r in mr])
        ax.annotate(m, (mean_faces, mean_vps), fontsize=6,
                    textcoords="offset points", xytext=(3, 3))
    ax.set_xlabel("Mesh faces")
    ax.set_ylabel("Selected viewpoints")
    ax.set_title("Viewpoints vs Mesh Complexity")
    save_figure(fig, os.path.join(fig_dir, "e15_scatter_vps_vs_faces"))

    # ── Fig 3: Grouped bar - coverage by model ──────────────────────
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4))
    cov_data = {"Coverage (%)": [np.mean([r["coverage"] * 100 for r in results if r["model"] == m])
                                  for m in models]}
    cov_errs = {"Coverage (%)": [np.std([r["coverage"] * 100 for r in results if r["model"] == m])
                                  for m in models]}
    grouped_bar(ax, cov_data, models, yerr=cov_errs,
                ylabel="Coverage (%)",
                title="Coverage by Model", value_labels=True, fmt="%.1f")
    ax.tick_params(axis="x", rotation=35)
    save_figure(fig, os.path.join(fig_dir, "e15_coverage_by_model"))

    # ── Fig 4: Stacked bar - time breakdown ─────────────────────────
    stage_names = ["Mesh", "Surface", "OG", "Sampling", "Visibility", "Set Cover"]
    stage_keys = ["t_mesh", "t_surface", "t_og", "t_sample", "t_vis", "t_opt"]

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4))
    stage_data = {}
    for sname, skey in zip(stage_names, stage_keys):
        stage_data[sname] = [
            np.mean([r[skey] for r in results if r["model"] == m])
            for m in models
        ]
    stacked_bar(ax, models, stage_data,
                ylabel="Time (s)", title="Time Breakdown by Model")
    ax.tick_params(axis="x", rotation=35)
    save_figure(fig, os.path.join(fig_dir, "e15_time_breakdown"))

    # ── Fig 5: Bar - redundancy by model ────────────────────────────
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4))
    red_means = [np.mean([r["redundancy"] for r in results if r["model"] == m])
                 for m in models]
    red_stds = [np.std([r["redundancy"] for r in results if r["model"] == m])
                for m in models]
    x = np.arange(len(models))
    bars = ax.bar(x, red_means, 0.6, yerr=red_stds, capsize=2,
                  color=CATEGORICAL_COLORS[2])
    ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha="right")
    ax.set_ylabel("Redundancy (viewpoints/point)")
    ax.set_title("Coverage Redundancy by Model")
    save_figure(fig, os.path.join(fig_dir, "e15_redundancy"))

    logger.info("E15 figures saved to %s", fig_dir)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="E15: Cross-Model Generalization")
    p.add_argument("--models", nargs="+", default=ALL_MODELS,
                   help="Models to evaluate")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_3)
    p.add_argument("--output_dir",
                   default=os.path.join(RESULTS_DIR, "e15_cross_model"))
    p.add_argument("--skip_existing", action="store_true")
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

        total = len(model_configs) * len(args.seeds)
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

            for seed in args.seeds:
                run_idx += 1
                rpath = os.path.join(
                    raw_dir,
                    f"model={model_cfg.name}_seed={seed}",
                )

                if args.skip_existing and os.path.exists(rpath + ".json"):
                    logger.info("[%d/%d] SKIP: %s seed=%d",
                                run_idx, total, model_cfg.name, seed)
                    all_results.append(load_run_result(rpath))
                    continue

                logger.info("[%d/%d] Running: %s seed=%d",
                            run_idx, total, model_cfg.name, seed)

                try:
                    result = run_single(model_cfg, seed)
                    all_results.append(result)
                    save_run_result(result, rpath)
                    logger.info("  vps=%d cov=%.2f%% time=%.1fs faces=%d",
                                result["num_viewpoints"],
                                result["coverage"] * 100,
                                result["total_time"],
                                result["mesh_faces"])
                except Exception as e:
                    logger.error("  FAILED: %s", e, exc_info=True)
    else:
        for fname in sorted(os.listdir(raw_dir)):
            if fname.endswith(".json"):
                all_results.append(
                    load_run_result(os.path.join(raw_dir, fname.replace(".json", ""))))

    if all_results:
        generate_plots(all_results, args.output_dir)

        # Print summary table
        logger.info("\n" + "=" * 80)
        logger.info("E15 SUMMARY")
        logger.info("=" * 80)
        logger.info("%-20s %8s %8s %10s %10s %10s",
                    "Model", "Faces", "VPs", "Cov%", "Time(s)", "Redund.")
        logger.info("-" * 68)
        models = sorted(set(r["model"] for r in all_results))
        for m in models:
            mr = [r for r in all_results if r["model"] == m]
            if mr:
                logger.info("%-20s %8.0f %8.1f %10.2f %10.1f %10.2f",
                            m,
                            np.mean([r["mesh_faces"] for r in mr]),
                            np.mean([r["num_viewpoints"] for r in mr]),
                            np.mean([r["coverage"] * 100 for r in mr]),
                            np.mean([r["total_time"] for r in mr]),
                            np.mean([r["redundancy"] for r in mr]))


if __name__ == "__main__":
    main()
