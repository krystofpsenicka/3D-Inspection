#!/usr/bin/env python3
"""E3: Visibility Method Comparison — gpu/cpu × raycast/epsilon across coverage targets.

All methods run on the same candidates per (model, target, seed). Set-cover: LazyGreedySetCover (CPU; isolates the visibility
comparison). Actual coverage cross-validated against ground-truth gpu_raycast.


    conda run -n isaaclab python -m experiments.e05_visibility_comparison
    conda run -n isaaclab python -m experiments.e05_visibility_comparison --plots_only
    conda run -n isaaclab python -m experiments.e05_visibility_comparison --methods gpu_raycast gpu_epsilon
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import defaultdict

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
    return "duke" if name == "duke_of_lancaster" else name


def _build_cpu_query(
    method: str, ctx: PipelineContext, target_points_np: np.ndarray, normals_np: np.ndarray
):
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
    """(N, M) bool via per-viewpoint CPU calls."""
    n = len(pos_np)
    m = len(vis_cpu.target_points)
    V = np.zeros((n, m), dtype=bool)
    for i in range(n):
        visible, _ = vis_cpu.compute_visibility(pos_np[i], rot_np[i])
        V[i, visible] = True
    return V


def _compute_iou_all_candidates(V_gt: np.ndarray, V_method: np.ndarray) -> float:
    intersection = (V_gt & V_method).sum(axis=1).astype(np.float32)
    union = (V_gt | V_method).sum(axis=1).astype(np.float32)
    iou = np.where(union > 0, intersection / union, 1.0)
    return float(iou.mean())


def _compute_f1_all_candidates(V_gt: np.ndarray, V_method: np.ndarray) -> float:
    """Mean per-candidate F1 = 2·TP / (2·TP + FP + FN). F1 weights precision and recall equally —
    the right choice because epsilon's failure mode is over-reporting (false positives inflate
    coverage downstream). When method and GT are both empty for a candidate, F1 = 1."""
    tp = (V_gt & V_method).sum(axis=1).astype(np.float32)
    fp = (~V_gt & V_method).sum(axis=1).astype(np.float32)
    fn = (V_gt & ~V_method).sum(axis=1).astype(np.float32)
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / denom, 1.0)
    return float(f1.mean())


def _actual_coverage(V_gt_np: np.ndarray, selected_indices) -> float:
    if hasattr(selected_indices, "get"):
        selected_indices = selected_indices.get()
    selected_indices = np.asarray(selected_indices)
    covered = V_gt_np[selected_indices].any(axis=0).sum()
    return float(covered / V_gt_np.shape[1])


def run_all_methods(
    ctx: PipelineContext, methods: list[str], target_coverage: float, seed: int
) -> list[dict]:
    target_points, normals = ctx.sample_surface()
    set_seed(seed)

    # targeted_50 candidates (consistent with e01/e02 default).
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

    with timed():
        V_gt_gpu, _ = gt_query.compute_visibility_batch(pos_gpu, rot_gpu)
    V_gt_np = cp.asnumpy(V_gt_gpu)

    results = []
    for method in methods:
        logger.debug("  Computing visibility: %s", method)
        try:
            if method == "gpu_raycast":
                # Already computed above; recompute for timing isolation.
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

        mean_iou = _compute_iou_all_candidates(V_gt_np, V_m_np)
        mean_f1 = _compute_f1_all_candidates(V_gt_np, V_m_np)

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


def _render_f1_table(
    results: list[dict],
    methods: list[str],
    expected_seeds_per_cell: int,
    fig_dir: str,
) -> None:
    """Aggregate per-(model, method) mean F1 over targets and seeds; log a text table + sanity
    summary; save a matplotlib table figure (mirrors the standalone verify_e05_f1_table.py)."""
    f1_results = [r for r in results if r.get("mean_f1") is not None]
    if not f1_results:
        return

    duke = "duke_of_lancaster"
    models_present = sorted({r["model"] for r in f1_results})
    table_models = ([duke] if duke in models_present else []) + sorted(
        m for m in models_present if m != duke
    )
    table_methods = [m for m in methods if any(r["method"] == m for r in f1_results)]
    targets_present = sorted({round(float(r["target_coverage"]), 4) for r in f1_results})

    by_model_method: dict[tuple[str, str], list[float]] = defaultdict(list)
    cell_counts: dict[tuple[str, str, float], int] = defaultdict(int)
    for r in f1_results:
        target = round(float(r["target_coverage"]), 4)
        by_model_method[(r["model"], r["method"])].append(float(r["mean_f1"]))
        cell_counts[(r["model"], r["method"], target)] += 1

    def fmt_cell(values: list[float]) -> str:
        if not values:
            return "-"
        m = float(np.mean(values))
        s = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        return f"{m:.4f}±{s:.4f} (n={len(values)})"

    col_w = 22
    logger.info("=" * 88)
    logger.info(
        "Per-(model, method) mean F1 over targets %s and seeds",
        targets_present,
    )
    logger.info("=" * 88)
    header = f"{'method':<14} | " + " | ".join(f"{m:^{col_w}}" for m in table_models)
    logger.info(header)
    logger.info("-" * len(header))
    for method in table_methods:
        cells = [
            f"{fmt_cell(by_model_method.get((model, method), [])):^{col_w}}"
            for model in table_models
        ]
        logger.info(f"{method:<14} | " + " | ".join(cells))

    logger.info("")
    logger.info("=" * 60)
    logger.info("Sanity summary")
    logger.info("=" * 60)
    expected_cells = len(table_models) * len(table_methods) * len(targets_present)
    expected_total = expected_cells * expected_seeds_per_cell
    logger.info("Included runs : %d", len(f1_results))
    logger.info(
        "Expected      : %d (%d models x %d methods x %d targets x %d seeds)",
        expected_total,
        len(table_models),
        len(table_methods),
        len(targets_present),
        expected_seeds_per_cell,
    )

    off_cells = [
        (model, method, target, cell_counts.get((model, method, target), 0))
        for model in table_models
        for method in table_methods
        for target in targets_present
        if cell_counts.get((model, method, target), 0) != expected_seeds_per_cell
    ]
    if off_cells:
        logger.warning("Cells with seed count != %d:", expected_seeds_per_cell)
        for model, method, target, n in off_cells:
            logger.warning("  (%s, %s, target=%s): n=%d", model, method, target, n)
    else:
        logger.info(
            "All %d cells have exactly %d seeds.", expected_cells, expected_seeds_per_cell
        )

    if not table_methods or not table_models:
        return

    col_labels = [_display_model(m) for m in table_models]
    row_labels = [_METHOD_LABELS.get(m, m) for m in table_methods]
    cell_text = []
    for method in table_methods:
        row = []
        for model in table_models:
            vals = by_model_method.get((model, method), [])
            if not vals:
                row.append("—")
            else:
                m = float(np.mean(vals))
                s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                row.append(f"{m:.4f}±{s:.4f}\n(n={len(vals)})")
        cell_text.append(row)

    height = max(2.0, 0.7 * (len(table_methods) + 1))
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, height))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.6)
    save_figure(fig, os.path.join(fig_dir, "cross_model_e05_f1_table"))
    logger.info("E05 F1 table figure saved")


def generate_plots(
    results: list[dict],
    methods: list[str],
    output_dir: str,
    expected_seeds_per_cell: int,
):
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

    # Per-model: actual-vs-target coverage line, Duke only
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

    # Cross-model timing bar (one figure, bars grouped by model × method)
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

    _render_f1_table(results, methods, expected_seeds_per_cell, fig_dir)


def main():
    p = argparse.ArgumentParser(description="E3: Visibility Method Comparison")
    p.add_argument(
        "--models",
        nargs="+",
        default=TOSCA_REPRESENTATIVE + ["duke_of_lancaster"],
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=_ALL_METHODS,
        help="gpu_raycast gpu_epsilon cpu_raycast cpu_epsilon",
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
        generate_plots(all_results, args.methods, args.output_dir, len(args.seeds))


if __name__ == "__main__":
    main()
