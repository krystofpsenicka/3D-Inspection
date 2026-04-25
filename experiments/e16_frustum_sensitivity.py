#!/usr/bin/env python3
"""E16: Frustum Parameter Sensitivity

Sweeps horizontal FoV and near/far plane distances for Duke of Lancaster to
show how the camera frustum specification affects inspection plan quality.

Every experiment in this suite depends heavily on frustum parameters, yet no
prior experiment varies them.  E16 isolates this dependency.

Fixed: Duke of Lancaster, 1500 candidates, targeted_50, 95% target coverage.
Sweep:
  fov_deg    {30, 45, 60, 90}                      degrees
  near/far   {(0.1, 5), (0.2, 10), (0.5, 15)}     metres

The occupancy grid and base sampler are shared across FoV variants for the
same near/far pair (OG depends on far, not FoV).  The visibility query is
rebuilt for each (fov, near, far) combination.

Metrics: viewpoints selected, coverage achieved, visibility + optimization time.
Output:  two heatmaps (viewpoints, coverage), line plots (vs FoV, vs far range),
         grouped timing bar.

Usage:
    conda run -n isaaclab python -m experiments.e16_frustum_sensitivity
    conda run -n isaaclab python -m experiments.e16_frustum_sensitivity --plots_only
"""

from __future__ import annotations

import argparse
import dataclasses
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
    ModelConfig, FrustumConfig, SEEDS_3, RESULTS_DIR,
    E16_FOV_VALUES, E16_NEAR_FAR_PAIRS,
)
from experiments.common.runner import set_seed, timed, free_gpu_memory
from experiments.common.pipeline_setup import PipelineContext, DegenerateNormalsError
from experiments.common.persistence import save_run_result, load_run_result
from experiments.common.sampling_dispatch import sample_strategy
from experiments.common.plotting import (
    setup_thesis_style, save_figure, grouped_bar,
    THESIS_COL, DOUBLE_COL, CATEGORICAL_COLORS, SEQUENTIAL_CMAP,
)
from visibility.set_cover import LazyGreedySetCover

logger = logging.getLogger(__name__)

_STRATEGY = "weighted_curvature"
_NUM_CANDIDATES = 1500
_TARGET_COVERAGE = 0.95


# ═══════════════════════════════════════════════════════════════════════════
# Single run
# ═══════════════════════════════════════════════════════════════════════════

def run_single(
    fov_deg: float,
    near: float,
    far: float,
    seed: int,
    ctx: PipelineContext,
    target_points,
    normals,
) -> dict:
    """Run one (fov, near, far, seed) combination.

    ``ctx`` must already have its OG and sampler built for this (near, far)
    pair.  The visibility query is built here with the full frustum.
    """
    set_seed(seed)

    # Build visibility query for this specific frustum
    _, o3d_mesh = ctx.load_mesh()
    from visibility.core.types import FrustumParams
    from visibility.visibility.raycast_cuda import RaycastingVisibilityQueryCuda

    frustum_params = FrustumParams(
        fov_y=float(np.deg2rad(fov_deg)),
        aspect=1.0,
        near=near,
        far=far,
    )
    vis_query = RaycastingVisibilityQueryCuda(
        mesh=o3d_mesh,
        target_points=target_points,
        normals=normals,
        frustum_params=frustum_params,
    )

    with timed() as t_sample:
        pos_gpu, rot_gpu, n_base, n_iter, base_name, _ = sample_strategy(
            ctx, _STRATEGY, _NUM_CANDIDATES,
            target_points, normals, vis_query, ctx.model,
        )

    with timed() as t_vis:
        V, _ = vis_query.compute_visibility_batch(pos_gpu, rot_gpu)

    V_np = cp.asnumpy(V)
    pos_np = cp.asnumpy(pos_gpu)
    rot_np = cp.asnumpy(rot_gpu)

    with timed() as t_opt:
        optimizer = LazyGreedySetCover(len(target_points), pos_np, rot_np, V_np)
        opt_result = optimizer.optimize(
            target_coverage=_TARGET_COVERAGE, max_viewpoints=1000,
        )

    return {
        "fov_deg": fov_deg,
        "near": near,
        "far": far,
        "seed": seed,
        "num_candidates": int(len(pos_gpu)),
        "n_base_candidates": n_base,
        "n_iterative_candidates": n_iter,
        "num_viewpoints": opt_result.num_viewpoints,
        "coverage": float(opt_result.total_coverage),
        "sampling_time": t_sample.elapsed,
        "visibility_time": t_vis.elapsed,
        "optimization_time": t_opt.elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Plot generation
# ═══════════════════════════════════════════════════════════════════════════

def _nf_label(near: float, far: float) -> str:
    return f"[{near:.1f}, {far:.0f}] m"


def generate_plots(
    results: list[dict],
    fov_values: list[float],
    near_far_pairs: list[tuple[float, float]],
    output_dir: str,
):
    """Generate all E16 figures."""
    setup_thesis_style()
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    if not results:
        logger.warning("No results to plot")
        return

    nf_labels = [_nf_label(n, f) for n, f in near_far_pairs]

    def _mean(fov, near, far, metric):
        vals = [r[metric] for r in results
                if r["fov_deg"] == fov and r["near"] == near and r["far"] == far]
        return float(np.mean(vals)) if vals else float("nan")

    def _std(fov, near, far, metric):
        vals = [r[metric] for r in results
                if r["fov_deg"] == fov and r["near"] == near and r["far"] == far]
        return float(np.std(vals)) if vals else 0.0

    # ── Heatmap helper ────────────────────────────────────────────────────
    def _build_matrix(metric):
        mat = np.full((len(near_far_pairs), len(fov_values)), float("nan"))
        for ri, (near, far) in enumerate(near_far_pairs):
            for ci, fov in enumerate(fov_values):
                mat[ri, ci] = _mean(fov, near, far, metric)
        return mat

    fov_labels = [f"{int(f)}°" for f in fov_values]

    # ── Fig 1: Heatmap — viewpoints selected ────────────────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3.0))
    mat = _build_matrix("num_viewpoints")
    im = ax.imshow(mat, aspect="auto", cmap=SEQUENTIAL_CMAP)
    ax.set_xticks(range(len(fov_values)))
    ax.set_xticklabels(fov_labels)
    ax.set_yticks(range(len(near_far_pairs)))
    ax.set_yticklabels(nf_labels)
    ax.set_xlabel("Horizontal FoV")
    ax.set_ylabel("Near / far plane (m)")
    ax.set_title("Viewpoints selected (Duke, targeted_50, 95% cov)")
    for ri in range(len(near_far_pairs)):
        for ci in range(len(fov_values)):
            v = mat[ri, ci]
            if not np.isnan(v):
                ax.text(ci, ri, f"{v:.0f}", ha="center", va="center",
                        fontsize=9, color="white" if v > mat.max() * 0.6 else "black")
    fig.colorbar(im, ax=ax, label="Viewpoints")
    fig.tight_layout()
    save_figure(fig, os.path.join(fig_dir, "e16_heatmap_viewpoints"))

    # ── Fig 2: Heatmap — coverage achieved ──────────────────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3.0))
    mat_cov = _build_matrix("coverage") * 100
    im = ax.imshow(mat_cov, aspect="auto", cmap=SEQUENTIAL_CMAP,
                   vmin=max(0, np.nanmin(mat_cov) - 2), vmax=100)
    ax.set_xticks(range(len(fov_values)))
    ax.set_xticklabels(fov_labels)
    ax.set_yticks(range(len(near_far_pairs)))
    ax.set_yticklabels(nf_labels)
    ax.set_xlabel("Horizontal FoV")
    ax.set_ylabel("Near / far plane (m)")
    ax.set_title("Coverage achieved (%, target = 95%)")
    for ri in range(len(near_far_pairs)):
        for ci in range(len(fov_values)):
            v = mat_cov[ri, ci]
            if not np.isnan(v):
                ax.text(ci, ri, f"{v:.1f}%", ha="center", va="center",
                        fontsize=9, color="white" if v < 92 else "black")
    fig.colorbar(im, ax=ax, label="Coverage (%)")
    fig.tight_layout()
    save_figure(fig, os.path.join(fig_dir, "e16_heatmap_coverage"))

    # ── Fig 3: Viewpoints vs FoV (one line per near/far pair) ───────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3.5))
    for i, ((near, far), nfl) in enumerate(zip(near_far_pairs, nf_labels)):
        m = [_mean(fov, near, far, "num_viewpoints") for fov in fov_values]
        s = [_std(fov, near, far, "num_viewpoints") for fov in fov_values]
        ax.errorbar(fov_values, m, yerr=s, marker="o", capsize=3,
                    color=CATEGORICAL_COLORS[i], label=nfl, linewidth=1.5)
    ax.set_xlabel("Horizontal FoV (°)")
    ax.set_ylabel("Selected viewpoints")
    ax.set_title("Viewpoints vs FoV (Duke, targeted_50, 95% cov)")
    ax.set_xticks(fov_values)
    ax.legend(fontsize=7)
    save_figure(fig, os.path.join(fig_dir, "e16_viewpoints_vs_fov"))

    # ── Fig 4: Coverage vs FoV ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3.5))
    for i, ((near, far), nfl) in enumerate(zip(near_far_pairs, nf_labels)):
        m = [_mean(fov, near, far, "coverage") * 100 for fov in fov_values]
        ax.plot(fov_values, m, marker="o",
                color=CATEGORICAL_COLORS[i], label=nfl, linewidth=1.5)
    ax.axhline(_TARGET_COVERAGE * 100, color="grey", linestyle="--",
               linewidth=0.9, label=f"Target {_TARGET_COVERAGE*100:.0f}%")
    ax.set_xlabel("Horizontal FoV (°)")
    ax.set_ylabel("Coverage (%)")
    ax.set_title("Coverage vs FoV (Duke)")
    ax.set_xticks(fov_values)
    ax.legend(fontsize=7)
    save_figure(fig, os.path.join(fig_dir, "e16_coverage_vs_fov"))

    # ── Fig 5: Viewpoints vs near/far range (one line per FoV) ──────────
    far_vals = [f for _, f in near_far_pairs]
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3.5))
    for i, fov in enumerate(fov_values):
        m = [_mean(fov, near, far, "num_viewpoints") for near, far in near_far_pairs]
        s = [_std(fov, near, far, "num_viewpoints") for near, far in near_far_pairs]
        ax.errorbar(range(len(near_far_pairs)), m, yerr=s, marker="s", capsize=3,
                    color=CATEGORICAL_COLORS[i], label=f"{int(fov)}°", linewidth=1.5)
    ax.set_xticks(range(len(near_far_pairs)))
    ax.set_xticklabels(nf_labels)
    ax.set_xlabel("Near / far plane (m)")
    ax.set_ylabel("Selected viewpoints")
    ax.set_title("Viewpoints vs Near/Far Range (Duke)")
    ax.legend(fontsize=7)
    save_figure(fig, os.path.join(fig_dir, "e16_viewpoints_vs_nearfar"))

    # ── Fig 6: Timing breakdown — grouped bar (one group per FoV) ───────
    combo_labels = [f"{int(fov)}°\n{_nf_label(n, f)}"
                    for fov in fov_values for n, f in near_far_pairs]
    t_sample_vals, t_vis_vals, t_opt_vals = [], [], []
    for fov in fov_values:
        for near, far in near_far_pairs:
            t_sample_vals.append(_mean(fov, near, far, "sampling_time"))
            t_vis_vals.append(_mean(fov, near, far, "visibility_time"))
            t_opt_vals.append(_mean(fov, near, far, "optimization_time"))

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.5))
    x = np.arange(len(combo_labels))
    width = 0.25
    ax.bar(x - width, t_sample_vals, width, label="Sampling", color=CATEGORICAL_COLORS[0])
    ax.bar(x, t_vis_vals, width, label="Visibility", color=CATEGORICAL_COLORS[1])
    ax.bar(x + width, t_opt_vals, width, label="Set cover", color=CATEGORICAL_COLORS[2])
    ax.set_xticks(x)
    ax.set_xticklabels(combo_labels, fontsize=7)
    ax.set_ylabel("Time (s)")
    ax.set_title("Stage timing per frustum combination (Duke, targeted_50)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, os.path.join(fig_dir, "e16_timing_breakdown"))

    logger.info("E16 figures saved to %s", fig_dir)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="E16: Frustum Parameter Sensitivity")
    p.add_argument("--fov_values", type=float, nargs="+", default=E16_FOV_VALUES)
    p.add_argument("--near_far", type=float, nargs="+",
                   help="Flat list of near/far pairs: near1 far1 near2 far2 ...")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_3)
    p.add_argument("--output_dir",
                   default=os.path.join(RESULTS_DIR, "e16_frustum_sensitivity"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--plots_only", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
    )

    if args.near_far is not None:
        if len(args.near_far) % 2 != 0:
            p.error("--near_far must have an even number of values (pairs)")
        near_far_pairs = list(zip(args.near_far[::2], args.near_far[1::2]))
    else:
        near_far_pairs = E16_NEAR_FAR_PAIRS

    raw_dir = os.path.join(args.output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    all_results: list[dict] = []

    if args.plots_only:
        for fname in sorted(os.listdir(raw_dir)):
            if fname.endswith(".json"):
                all_results.append(load_run_result(
                    os.path.join(raw_dir, fname.replace(".json", ""))))
    else:
        base_model = ModelConfig.duke_of_lancaster()
        total = len(near_far_pairs) * len(args.fov_values) * len(args.seeds)
        run_idx = 0

        # Outer loop: near/far pair — OG and sampler are shared across FoV values.
        for near, far in near_far_pairs:
            logger.info("=" * 60)
            logger.info("Near/far: (%.2f, %.1f) m  — building OG and sampler", near, far)

            # Create a model config with this near/far (FoV default 60° for OG sizing;
            # OG only depends on far, not FoV).
            nf_model = dataclasses.replace(
                base_model,
                frustum=FrustumConfig(fov_deg=60.0, near=near, far=far),
            )
            ctx = PipelineContext(nf_model)
            try:
                ctx.load_mesh()
                target_points, normals = ctx.sample_surface(seed=42)
                ctx.build_sampling_og()   # cached for this (near, far)
            except DegenerateNormalsError as e:
                logger.error("Mesh setup failed: %s", e)
                continue

            # Middle loop: FoV — only visibility query changes.
            for fov_deg in args.fov_values:
                for seed in args.seeds:
                    run_idx += 1
                    rpath = os.path.join(
                        raw_dir,
                        f"fov={fov_deg:.0f}_near={near}_far={far:.0f}_seed={seed}",
                    )
                    if args.resume and os.path.exists(rpath + ".json"):
                        logger.info(
                            "[%d/%d] SKIP fov=%.0f near=%.2f far=%.1f seed=%d",
                            run_idx, total, fov_deg, near, far, seed,
                        )
                        all_results.append(load_run_result(rpath))
                        continue

                    logger.info(
                        "[%d/%d] fov=%.0f° near=%.2f far=%.1f seed=%d",
                        run_idx, total, fov_deg, near, far, seed,
                    )
                    try:
                        result = run_single(
                            fov_deg, near, far, seed,
                            ctx, target_points, normals,
                        )
                        all_results.append(result)
                        save_run_result(result, rpath)
                        logger.info(
                            "  VPs=%d cov=%.2f%% t_vis=%.1fs t_opt=%.1fs",
                            result["num_viewpoints"],
                            result["coverage"] * 100,
                            result["visibility_time"],
                            result["optimization_time"],
                        )
                    except Exception as e:
                        logger.error("  FAILED: %s", e, exc_info=True)
                    finally:
                        free_gpu_memory()

    if all_results:
        generate_plots(all_results, args.fov_values, near_far_pairs, args.output_dir)


if __name__ == "__main__":
    main()
