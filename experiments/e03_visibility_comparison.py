#!/usr/bin/env python3
"""E3: Visibility Method Comparison

Compares four visibility implementations across coverage targets:
  gpu_raycast  — GPU ray-casting (ground truth, always correct)
  gpu_epsilon  — GPU epsilon-visibility (approximate, fast)
  cpu_raycast  — CPU ray-casting (ground truth, baseline speed)
  cpu_epsilon  — CPU epsilon-visibility (approximate)

All methods run on the same candidate viewpoints per (model, target, seed) so
timing and accuracy comparisons are apples-to-apples.

IoU is computed per candidate: |V_method[i] ∩ V_gt[i]| / |V_method[i] ∪ V_gt[i]|,
where V_gt = gpu_raycast.  Mean IoU is reported per run.

Set-cover: LazyGreedySetCover (CPU) — same for all methods to isolate the
visibility comparison from the optimizer.  Actual coverage always cross-validated
against ground-truth gpu_raycast.

Key findings expected:
  - epsilon (GPU & CPU) overestimates coverage (positive gap) and has IoU < 1
  - GPU variants are faster than CPU variants
  - GPU raycast is faster than GPU epsilon for large N (vectorised CUDA raycasting
    vs per-point epsilon angle computation)

Note: CPU methods are slow for large models (O(N) sequential raycast calls).
      Run CPU variants with TOSCA models or a reduced candidate count.

Usage:
    conda run -n isaaclab python -m experiments.e03_visibility_comparison
    conda run -n isaaclab python -m experiments.e03_visibility_comparison --plots_only
    # GPU-only (fast, includes Duke):
    conda run -n isaaclab python -m experiments.e03_visibility_comparison \\
        --methods gpu_raycast gpu_epsilon
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
    ModelConfig, SEEDS_3, E03_COVERAGE_TARGETS, TOSCA_REPRESENTATIVE, RESULTS_DIR,
)
from experiments.common.runner import set_seed, timed, free_gpu_memory
from experiments.common.pipeline_setup import PipelineContext, DegenerateNormalsError
from experiments.common.persistence import save_run_result, load_run_result
from experiments.common.plotting import (
    setup_thesis_style, save_figure, grouped_bar,
    THESIS_COL, DOUBLE_COL, CATEGORICAL_COLORS,
)
from visibility.set_cover import LazyGreedySetCover
from shared.types import Side

logger = logging.getLogger(__name__)

_ALL_METHODS = ["gpu_raycast", "gpu_epsilon", "cpu_raycast", "cpu_epsilon"]
_METHOD_COLORS = {
    "gpu_raycast": CATEGORICAL_COLORS[0],
    "gpu_epsilon": CATEGORICAL_COLORS[1],
    "cpu_raycast": CATEGORICAL_COLORS[2],
    "cpu_epsilon": CATEGORICAL_COLORS[3],
}
_METHOD_LABELS = {
    "gpu_raycast": "GPU Raycast",
    "gpu_epsilon": "GPU Epsilon",
    "cpu_raycast": "CPU Raycast",
    "cpu_epsilon": "CPU Epsilon",
}


# ═══════════════════════════════════════════════════════════════════════════
# Visibility helpers
# ═══════════════════════════════════════════════════════════════════════════

def _build_cpu_query(method: str, ctx: PipelineContext,
                     target_points_np: np.ndarray, normals_np: np.ndarray):
    """Construct a CPU visibility query."""
    from visibility.core.types import FrustumParams
    frustum_params = FrustumParams(
        fov_y=ctx.model.frustum.fov_y_rad,
        aspect=ctx.model.frustum.aspect,
        near=ctx.model.frustum.near,
        far=ctx.model.frustum.far,
    )
    _, o3d_mesh = ctx.load_mesh()
    if method == "cpu_raycast":
        from visibility.visibility.raycast import RaycastingVisibilityQuery
        return RaycastingVisibilityQuery(
            mesh=o3d_mesh,
            target_points=target_points_np,
            normals=normals_np,
            frustum_params=frustum_params,
        )
    elif method == "cpu_epsilon":
        from visibility.visibility.epsilon import EpsilonVisibilityQuery
        return EpsilonVisibilityQuery(
            target_points=target_points_np,
            normals=normals_np,
            frustum_params=frustum_params,
        )
    raise ValueError(f"Not a CPU method: {method}")


def _cpu_batch(vis_cpu, pos_np: np.ndarray, rot_np: np.ndarray) -> np.ndarray:
    """Compute visibility for all candidates via per-viewpoint CPU calls.

    Returns (N, M) boolean numpy array.
    """
    n = len(pos_np)
    m = len(vis_cpu.target_points)
    V = np.zeros((n, m), dtype=bool)
    for i in range(n):
        visible, _ = vis_cpu.compute_visibility(pos_np[i], rot_np[i])
        V[i, visible] = True
    return V


def _compute_iou_all_candidates(V_gt: np.ndarray, V_method: np.ndarray) -> float:
    """Mean per-candidate IoU between method and ground truth."""
    intersection = (V_gt & V_method).sum(axis=1).astype(np.float32)
    union = (V_gt | V_method).sum(axis=1).astype(np.float32)
    iou = np.where(union > 0, intersection / union, 1.0)
    return float(iou.mean())


def _actual_coverage(V_gt_np: np.ndarray, selected_indices) -> float:
    """Compute true coverage from GPU-raycast ground truth for selected viewpoints."""
    if hasattr(selected_indices, "get"):
        selected_indices = selected_indices.get()
    selected_indices = np.asarray(selected_indices)
    covered = V_gt_np[selected_indices].any(axis=0).sum()
    return float(covered / V_gt_np.shape[1])


# ═══════════════════════════════════════════════════════════════════════════
# Per-run logic
# ═══════════════════════════════════════════════════════════════════════════

def run_all_methods(ctx: PipelineContext, methods: list[str],
                    target_coverage: float, seed: int) -> list[dict]:
    """Run all requested methods on the same candidates. Returns list of result dicts."""
    target_points, normals = ctx.sample_surface()
    set_seed(seed)

    # Generate candidates (targeted_50, consistent with e01/e02 default strategy)
    sampler = ctx.build_sampler("targeted")
    num_cands = ctx.model.num_candidates
    n_base = num_cands // 2

    pos_gpu, rot_gpu = sampler.sample(
        cp.arange(len(target_points)), n_base,
        side=Side.OUTSIDE, curvature_weighting=False,
    )
    gt_query = ctx.build_visibility_query("raycast")
    V_init, _ = gt_query.compute_visibility_batch(pos_gpu, rot_gpu)
    coverage_count = V_init.astype(cp.int32).sum(axis=0)
    uncovered = cp.where(coverage_count < 1)[0]
    if len(uncovered) > 0 and n_base < num_cands:
        t_pos, t_rot = sampler.sample(
            uncovered, num_cands - n_base, side=Side.OUTSIDE, curvature_weighting=False)
        pos_gpu = cp.concatenate([pos_gpu, t_pos])
        rot_gpu = cp.concatenate([rot_gpu, t_rot])

    pos_np = cp.asnumpy(pos_gpu)
    rot_np = cp.asnumpy(rot_gpu)
    target_points_np = cp.asnumpy(target_points)
    normals_np = cp.asnumpy(normals)

    # Ground truth visibility (GPU raycast, always computed)
    with timed() as t_gt:
        V_gt_gpu, _ = gt_query.compute_visibility_batch(pos_gpu, rot_gpu)
    V_gt_np = cp.asnumpy(V_gt_gpu)

    results = []
    for method in methods:
        logger.debug("  Computing visibility: %s", method)
        try:
            if method == "gpu_raycast":
                # Already computed above — reuse for timing isolation
                with timed() as t_vis:
                    V_m_gpu, _ = gt_query.compute_visibility_batch(pos_gpu, rot_gpu)
                V_m_np = cp.asnumpy(V_m_gpu)

            elif method == "gpu_epsilon":
                eps_query = ctx.build_visibility_query("epsilon")
                with timed() as t_vis:
                    V_m_gpu, _ = eps_query.compute_visibility_batch(pos_gpu, rot_gpu)
                V_m_np = cp.asnumpy(V_m_gpu)

            elif method in ("cpu_raycast", "cpu_epsilon"):
                vis_cpu = _build_cpu_query(method, ctx, target_points_np, normals_np)
                with timed() as t_vis:
                    V_m_np = _cpu_batch(vis_cpu, pos_np, rot_np)

            else:
                logger.warning("Unknown method %s — skipping", method)
                continue

        except Exception as e:
            logger.error("  Visibility FAILED for %s: %s", method, e, exc_info=True)
            continue

        # IoU vs ground truth (all candidates)
        mean_iou = _compute_iou_all_candidates(V_gt_np, V_m_np)

        # Set cover (CPU LazyGreedy — same for all methods)
        with timed() as t_opt:
            optimizer = LazyGreedySetCover(
                len(target_points), pos_np, rot_np, V_m_np)
            opt_result = optimizer.optimize(
                target_coverage=target_coverage, max_viewpoints=1000)

        reported_cov = float(opt_result.total_coverage)
        actual_cov = _actual_coverage(V_gt_np, opt_result.selected_indices)

        results.append({
            "model": ctx.model.name,
            "method": method,
            "target_coverage": target_coverage,
            "seed": seed,
            "num_candidates": int(len(pos_gpu)),
            "num_viewpoints": opt_result.num_viewpoints,
            "reported_coverage": reported_cov,
            "actual_coverage": actual_cov,
            "coverage_gap": reported_cov - actual_cov,
            "mean_iou": mean_iou,
            "visibility_time": t_vis.elapsed,
            "optimization_time": t_opt.elapsed,
            "redundancy": float(opt_result.redundancy),
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Plot generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_plots(results: list[dict], methods: list[str], output_dir: str):
    """Generate all E3 figures."""
    setup_thesis_style()
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    models = sorted(set(r["model"] for r in results))
    targets = sorted(set(r["target_coverage"] for r in results))
    target_labels = [f"{t*100:.0f}%" for t in targets]

    present_methods = [m for m in methods if any(r["method"] == m for r in results)]
    colors = [_METHOD_COLORS.get(m, "grey") for m in present_methods]

    def _means(method, target, metric):
        vals = [r[metric] for r in results
                if r["method"] == method and r["target_coverage"] == target]
        return float(np.mean(vals)) if vals else float("nan")

    def _vals(method, target, metric):
        return [r[metric] for r in results
                if r["method"] == method and r["target_coverage"] == target]

    for model_name in models:
        mr = [r for r in results if r["model"] == model_name]
        if not mr:
            continue
        model_present = [m for m in present_methods if any(r["method"] == m for r in mr)]

        def _mm(method, target, metric):
            vals = [r[metric] for r in mr
                    if r["method"] == method and r["target_coverage"] == target]
            return float(np.mean(vals)) if vals else float("nan")

        def _mv(method, target, metric):
            return [r[metric] for r in mr
                    if r["method"] == method and r["target_coverage"] == target]

        # ── Fig 1: Timing — grouped bars (one group per target, bars = methods) ─
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4))
        timing_data = {
            _METHOD_LABELS.get(m, m): [_mm(m, t, "visibility_time") for t in targets]
            for m in model_present
        }
        m_colors = [_METHOD_COLORS.get(m, "grey") for m in model_present]
        grouped_bar(ax, timing_data, target_labels,
                    ylabel="Visibility time (s)",
                    title=f"Visibility Timing: GPU vs CPU, Raycast vs Epsilon ({model_name})")
        ax.set_xlabel("Target coverage")
        save_figure(fig, os.path.join(fig_dir, f"{model_name}_e03_timing"))

        # ── Fig 2: IoU box plot — one box per (method, target) ──────────
        approx_methods = [m for m in model_present if m != "gpu_raycast"]
        if approx_methods:
            fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4))
            n_approx = len(approx_methods)
            n_targets = len(targets)
            width = 0.8 / n_approx
            x_base = np.arange(n_targets)
            for ai, m in enumerate(approx_methods):
                iou_per_target = [_mv(m, t, "mean_iou") for t in targets]
                offset = (ai - n_approx / 2 + 0.5) * width
                bp = ax.boxplot(iou_per_target,
                                positions=x_base + offset,
                                widths=width * 0.9,
                                patch_artist=True,
                                medianprops=dict(color="black", linewidth=1.5),
                                boxprops=dict(facecolor=_METHOD_COLORS.get(m, "grey"),
                                              alpha=0.7))
            ax.set_xticks(x_base)
            ax.set_xticklabels(target_labels)
            ax.set_xlabel("Target coverage")
            ax.set_ylabel("Mean per-candidate IoU vs GPU raycast")
            ax.set_ylim(0, 1.05)
            ax.axhline(1.0, color="grey", linestyle="--", alpha=0.4, linewidth=0.8)
            ax.set_title(f"Accuracy vs Ground Truth (GPU Raycast) — {model_name}")
            # Legend
            from matplotlib.patches import Patch
            legend_handles = [Patch(facecolor=_METHOD_COLORS.get(m, "grey"),
                                    alpha=0.7, label=_METHOD_LABELS.get(m, m))
                              for m in approx_methods]
            ax.legend(handles=legend_handles, fontsize=8)
            save_figure(fig, os.path.join(fig_dir, f"{model_name}_e03_iou_boxplot"))

        # ── Fig 3: Coverage gap — grouped bars (epsilon variants only) ───
        epsilon_methods = [m for m in model_present if "epsilon" in m]
        if epsilon_methods:
            fig, ax = plt.subplots(figsize=(THESIS_COL, 3.5))
            gap_data = {
                _METHOD_LABELS.get(m, m): [_mm(m, t, "coverage_gap") * 100 for t in targets]
                for m in epsilon_methods
            }
            gap_colors = [_METHOD_COLORS.get(m, "grey") for m in epsilon_methods]
            grouped_bar(ax, gap_data, target_labels,
                        ylabel="Coverage gap (reported − actual) %",
                        title=f"Epsilon Optimism: Gap vs Ground Truth ({model_name})")
            ax.axhline(0, color="black", linewidth=0.7)
            ax.set_xlabel("Target coverage")
            save_figure(fig, os.path.join(fig_dir, f"{model_name}_e03_gap"))

        # ── Fig 4: Actual vs target coverage (line plot) ─────────────────
        fig, ax = plt.subplots(figsize=(THESIS_COL, 3.5))
        for m in model_present:
            actual = [_mm(m, t, "actual_coverage") * 100 for t in targets]
            ax.plot([t * 100 for t in targets], actual, "o-",
                    color=_METHOD_COLORS.get(m, "grey"),
                    label=_METHOD_LABELS.get(m, m), linewidth=1.5)
        ax.plot([t * 100 for t in targets], [t * 100 for t in targets],
                "k--", alpha=0.3, linewidth=0.8, label="Ideal")
        ax.set_xlabel("Target coverage (%)")
        ax.set_ylabel("Actual coverage (%)")
        ax.set_title(f"Actual vs Target Coverage ({model_name})")
        ax.legend(fontsize=8)
        save_figure(fig, os.path.join(fig_dir, f"{model_name}_e03_actual_vs_target"))

        # ── Fig 5: Viewpoints selected — grouped bars ─────────────────────
        fig, ax = plt.subplots(figsize=(THESIS_COL, 3.5))
        vp_data = {
            _METHOD_LABELS.get(m, m): [_mm(m, t, "num_viewpoints") for t in targets]
            for m in model_present
        }
        vp_colors = [_METHOD_COLORS.get(m, "grey") for m in model_present]
        grouped_bar(ax, vp_data, target_labels,
                    ylabel="Selected viewpoints",
                    title=f"Viewpoints Selected ({model_name})")
        ax.set_xlabel("Target coverage")
        save_figure(fig, os.path.join(fig_dir, f"{model_name}_e03_viewpoints"))

        logger.info("E3 figures saved for %s", model_name)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="E3: Visibility Method Comparison")
    p.add_argument("--models", nargs="+",
                   default=TOSCA_REPRESENTATIVE,
                   help="Models to evaluate (default: TOSCA; add duke_of_lancaster for GPU-only)")
    p.add_argument("--methods", nargs="+", default=_ALL_METHODS,
                   help="Visibility methods: gpu_raycast gpu_epsilon cpu_raycast cpu_epsilon")
    p.add_argument("--targets", type=float, nargs="+", default=E03_COVERAGE_TARGETS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_3)
    p.add_argument("--output_dir",
                   default=os.path.join(RESULTS_DIR, "e03_visibility_comparison"))
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
            cfg = (ModelConfig.duke_of_lancaster() if model_name == "duke_of_lancaster"
                   else ModelConfig.tosca(model_name))
            logger.info("=" * 60)
            logger.info("Model: %s", model_name)
            try:
                ctx = PipelineContext(cfg)
                ctx.load_mesh()
                ctx.sample_surface(seed=42)
                ctx.build_sampling_og()
            except DegenerateNormalsError:
                logger.warning("Skipping %s: degenerate normals", model_name)
                continue

            combos = [(t, seed) for t in args.targets for seed in args.seeds]
            total = len(combos)
            for idx, (target, seed) in enumerate(combos, 1):
                logger.info("[%d/%d] target=%.2f seed=%d", idx, total, target, seed)

                # Check if all methods already have results for this combo
                missing_methods = []
                for method in args.methods:
                    rpath = os.path.join(
                        raw_dir,
                        f"model={model_name}_method={method}_target={target}_seed={seed}")
                    if args.skip_existing and os.path.exists(rpath + ".json"):
                        all_results.append(load_run_result(rpath))
                    else:
                        missing_methods.append(method)

                if not missing_methods:
                    continue

                try:
                    run_results = run_all_methods(ctx, missing_methods, target, seed)
                    for r in run_results:
                        all_results.append(r)
                        rpath = os.path.join(
                            raw_dir,
                            f"model={model_name}_method={r['method']}"
                            f"_target={target}_seed={seed}")
                        save_run_result(r, rpath)
                        logger.info("  [%s] VPs=%d actual=%.2f%% IoU=%.3f t_vis=%.2fs",
                                    r["method"], r["num_viewpoints"],
                                    r["actual_coverage"] * 100,
                                    r["mean_iou"], r["visibility_time"])
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
        generate_plots(all_results, args.methods, args.output_dir)


if __name__ == "__main__":
    main()
