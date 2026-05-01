#!/usr/bin/env python3
"""E3: Visibility Method Comparison

Compares four visibility implementations across coverage targets:
  gpu_raycast   --  GPU ray-casting (ground truth, always correct)
  gpu_epsilon   --  GPU epsilon-visibility (approximate, fast)
  cpu_raycast   --  CPU ray-casting (ground truth, baseline speed)
  cpu_epsilon   --  CPU epsilon-visibility (approximate)

All methods run on the same candidate viewpoints per (model, target, seed) so
timing and accuracy comparisons are apples-to-apples.

IoU is computed per candidate: |V_method[i] ∩ V_gt[i]| / |V_method[i] ∪ V_gt[i]|,
where V_gt = gpu_raycast.  Mean IoU is reported per run.

Set-cover: LazyGreedySetCover (CPU)  --  same for all methods to isolate the
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
    conda run -n isaaclab python -m experiments.e05_visibility_comparison
    conda run -n isaaclab python -m experiments.e05_visibility_comparison --plots_only
    # GPU-only (fast, includes Duke):
    conda run -n isaaclab python -m experiments.e05_visibility_comparison \\
        --methods gpu_raycast gpu_epsilon
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.common.config import (
    E03_COVERAGE_TARGETS,
    RESULTS_DIR,
    SEEDS_3,
    TOSCA_REPRESENTATIVE,
    ModelConfig,
)
from experiments.common.persistence import load_run_result, save_run_result
from experiments.common.plotting import (
    CATEGORICAL_COLORS,
    DOUBLE_COL,
    THESIS_COL,
    grouped_bar,
    save_figure,
    setup_thesis_style,
)

# ── Runtime imports (need isaaclab/CUDA). Plot-only mode skips these. ──────
_RUNTIME_IMPORT_ERROR: ImportError | None = None
try:
    import cupy as cp

    from experiments.common.pipeline_setup import (
        DegenerateNormalsError,
        PipelineContext,
    )
    from experiments.common.runner import free_gpu_memory, set_seed, timed
    from shared.types import Side
    from visibility.set_cover import LazyGreedySetCover

    _RUNTIME_AVAILABLE = True
except ImportError as _e:
    cp = None
    set_seed = timed = free_gpu_memory = None
    PipelineContext = DegenerateNormalsError = None
    LazyGreedySetCover = None
    Side = None
    _RUNTIME_AVAILABLE = False
    _RUNTIME_IMPORT_ERROR = _e

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


def _display_model(name: str) -> str:
    """Map internal model id to a compact display label used in figures."""
    return "duke" if name == "duke_of_lancaster" else name


# ═══════════════════════════════════════════════════════════════════════════
# Visibility helpers
# ═══════════════════════════════════════════════════════════════════════════


def _build_cpu_query(
    method: str, ctx: PipelineContext, target_points_np: np.ndarray, normals_np: np.ndarray
):
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


def _compute_f1_all_candidates(V_gt: np.ndarray, V_method: np.ndarray) -> float:
    """Mean per-candidate F1 score between method and ground truth.

    F1 = 2.TP / (2.TP + FP + FN). F1 weights precision and recall equally,
    which is the right choice here because epsilon's failure mode is
    over-reporting (false positives inflate coverage downstream). When both
    method and GT are empty for a candidate, define F1 = 1 (trivially
    correct "nothing visible" agreement).
    """
    tp = (V_gt & V_method).sum(axis=1).astype(np.float32)
    fp = (~V_gt & V_method).sum(axis=1).astype(np.float32)
    fn = (V_gt & ~V_method).sum(axis=1).astype(np.float32)
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / denom, 1.0)
    return float(f1.mean())


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


def run_all_methods(
    ctx: PipelineContext, methods: list[str], target_coverage: float, seed: int
) -> list[dict]:
    """Run all requested methods on the same candidates. Returns list of result dicts."""
    target_points, normals = ctx.sample_surface()
    set_seed(seed)

    # Generate candidates (targeted_50, consistent with e01/e02 default strategy)
    sampler = ctx.build_sampler("targeted")
    num_cands = ctx.model.num_candidates
    n_base = num_cands // 2

    pos_gpu, rot_gpu = sampler.sample(
        cp.arange(len(target_points)),
        n_base,
        side=Side.OUTSIDE,
        curvature_weighting=False,
    )
    gt_query = ctx.build_visibility_query("raycast")
    V_init, _ = gt_query.compute_visibility_batch(pos_gpu, rot_gpu)
    coverage_count = V_init.astype(cp.int32).sum(axis=0)
    uncovered = cp.where(coverage_count < 1)[0]
    if len(uncovered) > 0 and n_base < num_cands:
        t_pos, t_rot = sampler.sample(
            uncovered, num_cands - n_base, side=Side.OUTSIDE, curvature_weighting=False
        )
        pos_gpu = cp.concatenate([pos_gpu, t_pos])
        rot_gpu = cp.concatenate([rot_gpu, t_rot])

    pos_np = cp.asnumpy(pos_gpu)
    rot_np = cp.asnumpy(rot_gpu)
    target_points_np = cp.asnumpy(target_points)
    normals_np = cp.asnumpy(normals)

    # Ground truth visibility (GPU raycast, always computed)
    with timed():
        V_gt_gpu, _ = gt_query.compute_visibility_batch(pos_gpu, rot_gpu)
    V_gt_np = cp.asnumpy(V_gt_gpu)

    results = []
    for method in methods:
        logger.debug("  Computing visibility: %s", method)
        try:
            if method == "gpu_raycast":
                # Already computed above  --  reuse for timing isolation
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

        # IoU and F1 vs ground truth (all candidates)
        mean_iou = _compute_iou_all_candidates(V_gt_np, V_m_np)
        mean_f1 = _compute_f1_all_candidates(V_gt_np, V_m_np)

        # Set cover (CPU LazyGreedy  --  same for all methods)
        with timed() as t_opt:
            optimizer = LazyGreedySetCover(len(target_points), pos_np, rot_np, V_m_np)
            opt_result = optimizer.optimize(target_coverage=target_coverage, max_viewpoints=1000)

        reported_cov = float(opt_result.total_coverage)
        actual_cov = _actual_coverage(V_gt_np, opt_result.selected_indices)

        results.append(
            {
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
                "mean_f1": mean_f1,
                "visibility_time": t_vis.elapsed,
                "optimization_time": t_opt.elapsed,
                "redundancy": float(opt_result.redundancy),
            }
        )

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
    [f"{t * 100:.0f}%" for t in targets]

    present_methods = [m for m in methods if any(r["method"] == m for r in results)]
    [_METHOD_COLORS.get(m, "grey") for m in present_methods]

    def _means(method, target, metric):
        vals = [
            r[metric] for r in results if r["method"] == method and r["target_coverage"] == target
        ]
        return float(np.mean(vals)) if vals else float("nan")

    def _vals(method, target, metric):
        return [
            r[metric] for r in results if r["method"] == method and r["target_coverage"] == target
        ]

    # ── Per-model: actual-vs-target coverage line, Duke only ────────────
    duke = "duke_of_lancaster"
    if duke in models:
        mr = [r for r in results if r["model"] == duke]
        model_present = [m for m in present_methods if any(r["method"] == m for r in mr)]

        def _mm(method, target, metric):
            vals = [
                r[metric] for r in mr if r["method"] == method and r["target_coverage"] == target
            ]
            return float(np.mean(vals)) if vals else float("nan")

        if model_present:
            fig, ax = plt.subplots(figsize=(THESIS_COL, 3.5))
            for m in model_present:
                actual = [_mm(m, t, "actual_coverage") * 100 for t in targets]
                ax.plot(
                    [t * 100 for t in targets],
                    actual,
                    "o-",
                    color=_METHOD_COLORS.get(m, "grey"),
                    label=_METHOD_LABELS.get(m, m),
                    linewidth=1.5,
                )
            ax.plot(
                [t * 100 for t in targets],
                [t * 100 for t in targets],
                "k--",
                alpha=0.3,
                linewidth=0.8,
                label="Ideal",
            )
            ax.set_xlabel("Target coverage (%)")
            ax.set_ylabel("Actual coverage (%)")
            ax.set_title(f"Actual vs Target Coverage ({_display_model(duke)})")
            ax.legend(fontsize=8)
            save_figure(fig, os.path.join(fig_dir, f"{duke}_e05_actual_vs_target"))
            logger.info("E05 actual-vs-target figure saved for %s", _display_model(duke))

    # ── Cross-model timing bar (one figure, bars grouped by model x method) ─
    if models:
        model_labels = [_display_model(m) for m in models]
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4))
        timing_data = {
            _METHOD_LABELS.get(m, m): [
                float(
                    np.mean(
                        [
                            r["visibility_time"]
                            for r in results
                            if r["method"] == m and r["model"] == md
                        ]
                    )
                    if any(r["method"] == m and r["model"] == md for r in results)
                    else float("nan")
                )
                for md in models
            ]
            for m in present_methods
        }
        grouped_bar(
            ax,
            timing_data,
            model_labels,
            ylabel="Visibility time (s)",
            title="Visibility Timing — Cross-Model Comparison",
        )
        ax.set_xlabel("Model")
        ax.tick_params(axis="x", rotation=20)
        save_figure(fig, os.path.join(fig_dir, "cross_model_e05_timing"))
        logger.info("Cross-model timing figure saved")

    # ── Print per-method mean F1 on Duke (table data for the thesis) ────
    if duke in models:
        f1_rows = [r for r in results if r["model"] == duke and r.get("mean_f1") is not None]
        if f1_rows:
            logger.info("E05 mean F1 vs gpu_raycast on Duke (averaged over targets, seeds):")
            for m in present_methods:
                vals = [r["mean_f1"] for r in f1_rows if r["method"] == m]
                if vals:
                    logger.info("  %-12s F1 = %.3f  (n=%d)", m, float(np.mean(vals)), len(vals))


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def main():
    p = argparse.ArgumentParser(description="E3: Visibility Method Comparison")
    p.add_argument(
        "--models",
        nargs="+",
        default=TOSCA_REPRESENTATIVE + ["duke_of_lancaster"],
        help="Models to evaluate (default: TOSCA and duke_of_lancaster)",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=_ALL_METHODS,
        help="Visibility methods: gpu_raycast gpu_epsilon cpu_raycast cpu_epsilon",
    )
    p.add_argument("--targets", type=float, nargs="+", default=E03_COVERAGE_TARGETS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_3)
    p.add_argument("--output_dir", default=os.path.join(RESULTS_DIR, "e05_visibility_comparison"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--plots_only", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
    )

    if not args.plots_only and not _RUNTIME_AVAILABLE:
        raise SystemExit(
            f"Runtime imports unavailable ({_RUNTIME_IMPORT_ERROR}). "
            "Activate the isaaclab env or pass --plots_only."
        )

    raw_dir = os.path.join(args.output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    all_results: list[dict] = []

    if not args.plots_only:
        for model_name in args.models:
            cfg = (
                ModelConfig.duke_of_lancaster()
                if model_name == "duke_of_lancaster"
                else ModelConfig.tosca(model_name)
            )
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
                        raw_dir, f"model={model_name}_method={method}_target={target}_seed={seed}"
                    )
                    if args.resume and os.path.exists(rpath + ".json"):
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
                            f"model={model_name}_method={r['method']}_target={target}_seed={seed}",
                        )
                        save_run_result(r, rpath)
                        logger.info(
                            "  [%s] VPs=%d actual=%.2f%% IoU=%.3f t_vis=%.2fs",
                            r["method"],
                            r["num_viewpoints"],
                            r["actual_coverage"] * 100,
                            r["mean_iou"],
                            r["visibility_time"],
                        )
                except Exception as e:
                    logger.error("  FAILED: %s", e, exc_info=True)
                finally:
                    free_gpu_memory()
    else:
        for fname in sorted(os.listdir(raw_dir)):
            if fname.endswith(".json"):
                all_results.append(
                    load_run_result(os.path.join(raw_dir, fname.replace(".json", "")))
                )

    if all_results:
        generate_plots(all_results, args.methods, args.output_dir)


if __name__ == "__main__":
    main()
