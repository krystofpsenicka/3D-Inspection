#!/usr/bin/env python3
"""E03: Iterative Sampler Parameter Sweep

Runs on two model groups: 3 TOSCA models (wolf0, cat0, david0) and the larger
Duke of Lancaster model. Candidate budget scales per model via
`cfg.num_candidates x k_coverage` (TOSCA -> 500.k, Duke -> 1500.k) and the
set-cover cap is 1000 for TOSCA, 2000 for Duke.

Section 1  --  Targeted sampler:
  Sub-B only: samples_per_iteration sweep at k=4, fraction=100%.
  The goal is to confirm the targeted sampler does not improve over
  weighted_curvature; a full k x fraction sweep is not needed.
  spi=None is the pre-fix baseline: all n_iter candidates sampled in one shot
  from the frozen initial uncovered set (no coverage updates).

Section 2  --  CMA-ES sampler (travel_weight is per-model-group):
  Sub-B: joint k x travel_weight sweep at fraction=100%  --  2-D heatmap,
         because k and travel_weight interact meaningfully.
         TOSCA tw list: [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3];
         Duke tw list:  [0.01, 0.02, 0.03, 0.06, 0.1].
  Sub-C: popsize x maxiter heatmap at k=4, fraction=100%, tw fixed per group
         (TOSCA tw=0.1, Duke tw=0.0).

Set-cover: LazyGreedySetCover (CPU)  --  fastest per e06 results.

Usage:
    # Default: run TOSCA + Duke
    conda run -n isaaclab python -m experiments.e03_iterative_sampler_params

    # TOSCA only / Duke only
    conda run -n isaaclab python -m experiments.e03_iterative_sampler_params --model_group tosca
    conda run -n isaaclab python -m experiments.e03_iterative_sampler_params --model_group duke

    # Restrict to one sampler section
    conda run -n isaaclab python -m experiments.e03_iterative_sampler_params --section 1

    # Explicit model subset (overrides --model_group)
    conda run -n isaaclab python -m experiments.e03_iterative_sampler_params --models wolf0 duke_of_lancaster

    # Regenerate plots from saved results
    conda run -n isaaclab python -m experiments.e03_iterative_sampler_params --plots_only
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
    E03_C_K_VALUES,
    E03_C_MAXITER_VALUES,
    E03_C_POPSIZE_VALUES,
    E03_C_TRAVEL_WEIGHTS_DUKE,
    E03_C_TRAVEL_WEIGHTS_TOSCA,
    E03_T_SPI_VALUES,
    RESULTS_DIR,
    SEEDS_3,
    TOSCA_REPRESENTATIVE,
    ModelConfig,
)
from experiments.common.persistence import load_run_result, save_run_result
from experiments.common.plotting import (
    CATEGORICAL_COLORS,
    DOUBLE_COL,
    heatmap_annotated,
    panel_title,
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
    from visibility.core.constants import OPT_SAMPLER_MAXITER, OPT_SAMPLER_POPSIZE
    from visibility.set_cover import LazyGreedySetCover

    _RUNTIME_AVAILABLE = True
except ImportError as _e:
    cp = None
    OPT_SAMPLER_POPSIZE = OPT_SAMPLER_MAXITER = None
    set_seed = timed = free_gpu_memory = None
    PipelineContext = DegenerateNormalsError = None
    Side = None
    LazyGreedySetCover = None
    _RUNTIME_AVAILABLE = False
    _RUNTIME_IMPORT_ERROR = _e

logger = logging.getLogger(__name__)

# Fixed sub-section parameters
_S1B_K = 4
_S1B_FRACTION = 100
_S2B_FRACTION = 100
_S2C_K = 4
_S2C_FRACTION = 100
_S2C_TRAVEL_WEIGHT_TOSCA = 0.1
_S2C_TRAVEL_WEIGHT_DUKE = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════


def _max_viewpoints_for(model) -> int:
    """Set-cover cap: 2000 for the larger Duke model, 1000 for TOSCA."""
    return 2000 if model.name == "duke_of_lancaster" else 1000


def _is_duke(model) -> bool:
    return model.name == "duke_of_lancaster"


def _s2c_travel_weight_for(model) -> float:
    return _S2C_TRAVEL_WEIGHT_DUKE if _is_duke(model) else _S2C_TRAVEL_WEIGHT_TOSCA


def _s2b_travel_weights_for(model) -> list:
    return E03_C_TRAVEL_WEIGHTS_DUKE if _is_duke(model) else E03_C_TRAVEL_WEIGHTS_TOSCA


def _make_cfg(name: str):
    """Resolve a model name to a ModelConfig. Returns None if not found."""
    if name == "duke_of_lancaster":
        return ModelConfig.duke_of_lancaster()
    try:
        return ModelConfig.tosca(name)
    except FileNotFoundError:
        logger.warning("Model not found: %s", name)
        return None


def _run_set_cover(target_points, pos_gpu, rot_gpu, V, target_coverage, max_viewpoints=1000):
    """Run LazyGreedySetCover (CPU). Returns (opt_result, V_np)."""
    V_np = cp.asnumpy(V)
    pos_np = cp.asnumpy(pos_gpu)
    rot_np = cp.asnumpy(rot_gpu)
    optimizer = LazyGreedySetCover(len(target_points), pos_np, rot_np, V_np)
    return optimizer.optimize(target_coverage=target_coverage, max_viewpoints=max_viewpoints), V_np


def _base_phase(sampler, vis_query, target_points, n_base):
    """Run weighted base sampling phase.

    Returns (pos, rot, coverage_count, uncovered_indices) on GPU.
    If n_base == 0, returns empty arrays and zero coverage.
    """
    M = len(target_points)
    if n_base > 0:
        pos, rot = sampler.sample(
            cp.arange(M),
            n_base,
            side=Side.OUTSIDE,
            curvature_weighting=False,
        )
        V_init, _ = vis_query.compute_visibility_batch(pos, rot)
        coverage_count = V_init.astype(cp.int32).sum(axis=0)
    else:
        pos = cp.empty((0, 3), dtype=cp.float32)
        rot = cp.empty((0, 3, 3), dtype=cp.float32)
        coverage_count = cp.zeros(M, dtype=cp.int32)
    return pos, rot, coverage_count


# ═══════════════════════════════════════════════════════════════════════════
# Section 1  --  Targeted sampler
# ═══════════════════════════════════════════════════════════════════════════


def run_single_targeted(
    ctx: PipelineContext,
    k_coverage: int,
    fraction: int,
    spi: int | None,
    seed: int,
    section_tag: str,
    target_coverage: float = 0.95,
) -> dict:
    """One targeted-sampler run.

    Args:
        spi: samples_per_iteration. None = all-at-once baseline (no iterative
             coverage updates); int = iterative with batch size spi.
    """
    target_points, normals = ctx.sample_surface()
    set_seed(seed)
    vis_query = ctx.build_visibility_query("raycast")
    sampler = ctx.build_sampler("targeted")
    model = ctx.model
    max_vps = _max_viewpoints_for(model)

    N = model.num_candidates * k_coverage
    n_iter = int(N * fraction / 100)
    n_base = N - n_iter

    with timed() as t_sample:
        pos, rot, coverage_count = _base_phase(sampler, vis_query, target_points, n_base)
        uncovered = cp.where(coverage_count < k_coverage)[0]

        if n_iter > 0 and len(uncovered) > 0:
            if spi is None:
                # Baseline: single-shot sampling, no iterative coverage updates
                t_pos, t_rot = sampler.sample(
                    uncovered,
                    n_iter,
                    side=Side.OUTSIDE,
                    curvature_weighting=False,
                    k_coverage=k_coverage,
                    coverage_count_gpu=coverage_count,
                    visibility_query=None,
                )
            else:
                # Proper iterative mode: update coverage after each batch
                t_pos, t_rot = sampler.sample(
                    uncovered,
                    n_iter,
                    side=Side.OUTSIDE,
                    curvature_weighting=False,
                    k_coverage=k_coverage,
                    coverage_count_gpu=coverage_count,
                    visibility_query=vis_query,
                    samples_per_iteration=spi,
                )
        else:
            t_pos = cp.empty((0, 3), dtype=cp.float32)
            t_rot = cp.empty((0, 3, 3), dtype=cp.float32)

        if len(t_pos) > 0:
            pos = cp.concatenate([pos, t_pos]) if len(pos) > 0 else t_pos
            rot = cp.concatenate([rot, t_rot]) if len(rot) > 0 else t_rot

    with timed() as t_vis:
        V, _ = vis_query.compute_visibility_batch(pos, rot)

    V_np = cp.asnumpy(V)
    pool_redundancy = float(V_np.sum() / len(target_points))

    with timed() as t_opt:
        opt_result, _ = _run_set_cover(
            target_points, pos, rot, V, target_coverage, max_viewpoints=max_vps
        )

    return {
        "section": section_tag,
        "model": model.name,
        "k_coverage": k_coverage,
        "fraction": fraction,
        "spi": spi,
        "seed": seed,
        "base_n": model.num_candidates,
        "max_viewpoints": max_vps,
        "n_base": n_base,
        "n_iter_requested": n_iter,
        "n_iter_actual": int(len(t_pos)),
        "num_candidates": int(len(pos)),
        "num_viewpoints": opt_result.num_viewpoints,
        "coverage": float(opt_result.total_coverage),
        "pool_redundancy": pool_redundancy,
        "redundancy": float(opt_result.redundancy),
        "sampling_time": t_sample.elapsed,
        "visibility_time": t_vis.elapsed,
        "optimization_time": t_opt.elapsed,
        "total_time": t_sample.elapsed + t_vis.elapsed + t_opt.elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Section 2  --  CMA-ES sampler
# ═══════════════════════════════════════════════════════════════════════════


def run_single_cmaes(
    ctx: PipelineContext,
    k_coverage: int,
    fraction: int,
    travel_weight: float,
    seed: int,
    section_tag: str,
    target_coverage: float = 0.95,
    popsize: int = OPT_SAMPLER_POPSIZE,
    maxiter: int = OPT_SAMPLER_MAXITER,
) -> dict:
    """One CMA-ES sampler run."""
    from visibility.sampling import CMAESBackend, OptimizingSampler

    target_points, normals = ctx.sample_surface()
    set_seed(seed)
    vis_query = ctx.build_visibility_query("raycast")
    sampler = ctx.build_sampler("targeted")  # random_sampler for warm-start
    og = ctx.build_sampling_og()
    model = ctx.model
    max_vps = _max_viewpoints_for(model)

    N = model.num_candidates * k_coverage
    n_iter = int(N * fraction / 100)
    n_base = N - n_iter

    with timed() as t_sample:
        pos, rot, coverage_count = _base_phase(sampler, vis_query, target_points, n_base)
        uncovered = cp.where(coverage_count < k_coverage)[0]

        if n_iter > 0 and len(uncovered) > 0:
            opt_sampler = OptimizingSampler(
                mesh=ctx._o3d_mesh,
                target_points=target_points,
                normals=normals,
                frustum_far=model.frustum.far,
                collision_radius=model.collision_radius,
                occupancy_grid=og,
                backend=CMAESBackend(),
                random_sampler=sampler,
            )
            opt_pos, opt_rot, n_warmstart_fallbacks = opt_sampler.sample_optimized(
                n_iter,
                coverage_count,
                vis_query,
                existing_pos_gpu=pos if len(pos) > 0 else None,
                existing_rot_gpu=rot if len(rot) > 0 else None,
                k_coverage=k_coverage,
                travel_weight=travel_weight,
                popsize=popsize,
                maxiter=maxiter,
            )
        else:
            opt_pos = cp.empty((0, 3), dtype=cp.float32)
            opt_rot = cp.empty((0, 3, 3), dtype=cp.float32)
            n_warmstart_fallbacks = 0

        if len(opt_pos) > 0:
            pos = cp.concatenate([pos, opt_pos]) if len(pos) > 0 else opt_pos
            rot = cp.concatenate([rot, opt_rot]) if len(rot) > 0 else opt_rot

    with timed() as t_vis:
        V, _ = vis_query.compute_visibility_batch(pos, rot)

    V_np = cp.asnumpy(V)
    pool_redundancy = float(V_np.sum() / len(target_points))

    with timed() as t_opt:
        opt_result, _ = _run_set_cover(
            target_points, pos, rot, V, target_coverage, max_viewpoints=max_vps
        )

    return {
        "section": section_tag,
        "model": model.name,
        "k_coverage": k_coverage,
        "fraction": fraction,
        "travel_weight": travel_weight,
        "popsize": popsize,
        "maxiter": maxiter,
        "seed": seed,
        "base_n": model.num_candidates,
        "max_viewpoints": max_vps,
        "n_base": n_base,
        "n_iter_requested": n_iter,
        "n_iter_actual": int(len(opt_pos)),
        "n_warmstart_fallbacks": int(n_warmstart_fallbacks),
        "num_candidates": int(len(pos)),
        "num_viewpoints": opt_result.num_viewpoints,
        "coverage": float(opt_result.total_coverage),
        "pool_redundancy": pool_redundancy,
        "redundancy": float(opt_result.redundancy),
        "sampling_time": t_sample.elapsed,
        "visibility_time": t_vis.elapsed,
        "optimization_time": t_opt.elapsed,
        "total_time": t_sample.elapsed + t_vis.elapsed + t_opt.elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Plotting  --  shared helpers
# ═══════════════════════════════════════════════════════════════════════════


def _mean(results, metric, **filters):
    """Mean of metric over results matching all filter key=value pairs."""
    rows = results
    for k, v in filters.items():
        rows = [r for r in rows if r.get(k) == v]
    vals = [r[metric] for r in rows if metric in r]
    return float(np.mean(vals)) if vals else float("nan")


def _std(results, metric, **filters):
    rows = results
    for k, v in filters.items():
        rows = [r for r in rows if r.get(k) == v]
    vals = [r[metric] for r in rows if metric in r]
    return float(np.std(vals)) if len(vals) > 1 else 0.0


def _heatmap_2d(
    ax,
    results,
    row_key,
    row_vals,
    col_key,
    col_vals,
    metric,
    fmt,
    title,
    mult=1.0,
    xlabel="",
    ylabel="",
    cbar_label: str = "",
    overlay_mask: np.ndarray | None = None,
):
    """General annotated heatmap for any two parameter axes."""
    vals = np.zeros((len(row_vals), len(col_vals)))
    for i, rv in enumerate(row_vals):
        for j, cv in enumerate(col_vals):
            vals[i, j] = _mean(results, metric, **{row_key: rv, col_key: cv}) * mult
    heatmap_annotated(
        ax,
        [str(v) for v in row_vals],
        [str(v) for v in col_vals],
        vals,
        fmt=fmt,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        cbar_label=cbar_label,
        overlay_mask=overlay_mask,
    )


def _coverage_below_target_mask(
    results, row_key, row_vals, col_key, col_vals, target: float
) -> np.ndarray:
    """Boolean mask of (row, col) cells whose mean coverage < target."""
    mask = np.zeros((len(row_vals), len(col_vals)), dtype=bool)
    for i, rv in enumerate(row_vals):
        for j, cv in enumerate(col_vals):
            cov = _mean(results, "coverage", **{row_key: rv, col_key: cv})
            mask[i, j] = (cov > 0) and (cov < target)
    return mask


# ═══════════════════════════════════════════════════════════════════════════
# Plotting  --  combined Duke + TOSCA figures
# ═══════════════════════════════════════════════════════════════════════════


def generate_plots_section1_combined(
    groups: list[tuple[str, str, list[dict]]], spi_values: list, fig_dir: str
) -> None:
    """One combined figure for Section 1B with Duke + TOSCA side-by-side.

    Drops the coverage panel (always >= target) and shows two metrics
    (selected viewpoints, sampling time) for each group. Each row is one
    mesh group (panel-titled with the group label).
    """
    spi_labels = ["all-at-once" if s is None else str(s) for s in spi_values]
    metrics = [
        ("num_viewpoints", 1.0, "Selected viewpoints"),
        ("sampling_time", 1.0, "Sampling time (s)"),
    ]
    fig, axes = plt.subplots(
        len(groups), len(metrics), figsize=(DOUBLE_COL, 3.4 * len(groups)), squeeze=False
    )
    for row, (_tag, label, results) in enumerate(groups):
        r1B = [r for r in results if r.get("section") == "1B"]
        if not r1B:
            continue
        for col, (metric, mult, ylabel) in enumerate(metrics):
            ax = axes[row][col]
            means = [_mean(r1B, metric, spi=s) * mult for s in spi_values]
            stds = [_std(r1B, metric, spi=s) * mult for s in spi_values]
            colors = [
                CATEGORICAL_COLORS[0] if s is None else CATEGORICAL_COLORS[2] for s in spi_values
            ]
            ax.bar(spi_labels, means, yerr=stds, color=colors, capsize=3, alpha=0.85)
            ax.set_ylabel(ylabel)
            ax.tick_params(axis="x", rotation=20)
            panel_title(ax, f"{label} — {ylabel}")
    fig.tight_layout()
    save_figure(fig, os.path.join(fig_dir, "e03_combined_s1B_spi_sweep"))
    logger.info("Section 1B combined figure saved.")


def generate_plots_section2_combined(
    groups: list[tuple[str, str, list[dict]]],
    k_values: list[int],
    popsize_values: list[int],
    maxiter_values: list[int],
    fig_dir: str,
    target_coverage: float = 0.95,
) -> None:
    """Combined Duke + TOSCA figures for Section 2B (k x tw) and 2C (pop x mi).

    Cells whose mean coverage falls below ``target_coverage`` are flagged
    with an overlay marker so the reader can see where CMA-ES failed to
    meet the target while inspecting the viewpoint-count heatmap.
    """
    # ── Sub-B: viewpoints heatmap (k x travel_weight, frac=100%) ────────
    fig, axes = plt.subplots(1, len(groups), figsize=(DOUBLE_COL, 3.5), squeeze=False)
    for col, (_tag, label, results) in enumerate(groups):
        r2B = [r for r in results if r.get("section") == "2B"]
        if not r2B:
            continue
        tws = sorted(set(r["travel_weight"] for r in r2B))
        below = _coverage_below_target_mask(
            r2B, "k_coverage", k_values, "travel_weight", tws, target_coverage
        )
        ax = axes[0][col]
        _heatmap_2d(
            ax,
            r2B,
            "k_coverage",
            k_values,
            "travel_weight",
            tws,
            "num_viewpoints",
            ".0f",
            "",
            xlabel="travel_weight",
            ylabel="k-coverage",
            cbar_label="Selected viewpoints",
            overlay_mask=below,
        )
        panel_title(ax, label)
    fig.tight_layout()
    save_figure(fig, os.path.join(fig_dir, "e03_combined_s2B_k_tw_heatmap"))
    logger.info("Section 2B combined figure saved.")

    # ── Sub-C: viewpoints heatmap (popsize x maxiter) ────────────────────
    fig, axes = plt.subplots(1, len(groups), figsize=(DOUBLE_COL, 3.5), squeeze=False)
    for col, (_tag, label, results) in enumerate(groups):
        r2C = [r for r in results if r.get("section") == "2C"]
        if not r2C:
            continue
        below = _coverage_below_target_mask(
            r2C, "popsize", popsize_values, "maxiter", maxiter_values, target_coverage
        )
        ax = axes[0][col]
        _heatmap_2d(
            ax,
            r2C,
            "popsize",
            popsize_values,
            "maxiter",
            maxiter_values,
            "num_viewpoints",
            ".0f",
            "",
            xlabel="maxiter",
            ylabel="popsize",
            cbar_label="Selected viewpoints",
            overlay_mask=below,
        )
        panel_title(ax, label)
    fig.tight_layout()
    save_figure(fig, os.path.join(fig_dir, "e03_combined_s2C_popsize_maxiter"))
    logger.info("Section 2C combined figure saved.")


# ═══════════════════════════════════════════════════════════════════════════
# Post-processing: add derived fields to loaded results
# ═══════════════════════════════════════════════════════════════════════════


def _add_derived(results: list[dict]) -> list[dict]:
    """Add early_stop_ratio field if not already present."""
    for r in results:
        if "early_stop_ratio" not in r:
            req = r.get("n_iter_requested", 0)
            act = r.get("n_iter_actual", 0)
            r["early_stop_ratio"] = (act / req) if req > 0 else 1.0
    return results


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def main():
    p = argparse.ArgumentParser(description="E03: Iterative Sampler Parameter Sweep")
    p.add_argument(
        "--section",
        choices=["1", "2", "both"],
        default="both",
        help="Which sampler section to run (1=Targeted, 2=CMA-ES, default: both)",
    )
    p.add_argument(
        "--model_group",
        choices=["tosca", "duke", "both"],
        default="both",
        help="Which model family to run. Ignored if --models is given.",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Explicit model list (TOSCA names and/or 'duke_of_lancaster'). "
        "Overrides --model_group when set.",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_3)
    p.add_argument("--target_coverage", type=float, default=0.95)
    p.add_argument(
        "--output_dir", default=os.path.join(RESULTS_DIR, "e03_iterative_sampler_params")
    )
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--plots_only", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    if args.models is None:
        if args.model_group == "tosca":
            args.models = list(TOSCA_REPRESENTATIVE)
        elif args.model_group == "duke":
            args.models = ["duke_of_lancaster"]
        else:  # "both"
            args.models = list(TOSCA_REPRESENTATIVE) + ["duke_of_lancaster"]

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

    run_s1 = args.section in ("1", "both")
    run_s2 = args.section in ("2", "both")

    if not args.plots_only:
        cfgs = [c for name in args.models if (c := _make_cfg(name)) is not None]

        # ── Section 1: Targeted ──────────────────────────────────────────
        if run_s1:
            for cfg in cfgs:
                logger.info("=" * 60)
                logger.info("Section 1 — Targeted — Model: %s", cfg.name)
                try:
                    ctx = PipelineContext(cfg)
                    ctx.load_mesh()
                    ctx.sample_surface(seed=42)
                    ctx.build_sampling_og()
                except DegenerateNormalsError as e:
                    logger.warning("Skipping %s: %s", cfg.name, e)
                    continue

                # Sub-B: spi sweep (k=_S1B_K, fraction=_S1B_FRACTION)
                combos_1B = [(spi, seed) for spi in E03_T_SPI_VALUES for seed in args.seeds]
                logger.info(
                    "  Sub-B: %d combos (spi sweep, k=%d frac=%d%%)",
                    len(combos_1B),
                    _S1B_K,
                    _S1B_FRACTION,
                )
                for idx, (spi, seed) in enumerate(combos_1B, 1):
                    spi_tag = "all" if spi is None else str(spi)
                    rpath = os.path.join(raw_dir, f"1B_model={cfg.name}_spi={spi_tag}_seed={seed}")
                    if args.skip_existing and os.path.exists(rpath + ".json"):
                        logger.info(
                            "[1B %d/%d] SKIP %s spi=%s seed=%d",
                            idx,
                            len(combos_1B),
                            cfg.name,
                            spi_tag,
                            seed,
                        )
                        all_results.append(load_run_result(rpath))
                        continue
                    logger.info(
                        "[1B %d/%d] model=%s k=%d frac=%d%% spi=%s seed=%d",
                        idx,
                        len(combos_1B),
                        cfg.name,
                        _S1B_K,
                        _S1B_FRACTION,
                        spi_tag,
                        seed,
                    )
                    try:
                        result = run_single_targeted(
                            ctx,
                            _S1B_K,
                            _S1B_FRACTION,
                            spi=spi,
                            seed=seed,
                            section_tag="1B",
                            target_coverage=args.target_coverage,
                        )
                        result = _add_derived([result])[0]
                        all_results.append(result)
                        save_run_result(result, rpath)
                        logger.info(
                            "  VPs=%d cov=%.1f%% iter=%d/%d t=%.1fs",
                            result["num_viewpoints"],
                            result["coverage"] * 100,
                            result["n_iter_actual"],
                            result["n_iter_requested"],
                            result["total_time"],
                        )
                    except Exception as e:
                        logger.error("  FAILED: %s", e, exc_info=True)
                    finally:
                        free_gpu_memory()

        # ── Section 2: CMA-ES ────────────────────────────────────────────
        if run_s2:
            for cfg in cfgs:
                logger.info("=" * 60)
                logger.info("Section 2 — CMA-ES — Model: %s", cfg.name)
                try:
                    ctx = PipelineContext(cfg)
                    ctx.load_mesh()
                    ctx.sample_surface(seed=42)
                    ctx.build_sampling_og()
                except DegenerateNormalsError as e:
                    logger.warning("Skipping %s: %s", cfg.name, e)
                    continue

                # Sub-B: joint k x travel_weight sweep at fraction=_S2B_FRACTION
                s2b_tws = _s2b_travel_weights_for(cfg)
                combos_2B = [
                    (k, tw, seed) for k in E03_C_K_VALUES for tw in s2b_tws for seed in args.seeds
                ]
                logger.info(
                    "  Sub-B: %d combos (k × travel_weight, k=%s, tw=%s, frac=%d%%)",
                    len(combos_2B),
                    E03_C_K_VALUES,
                    s2b_tws,
                    _S2B_FRACTION,
                )
                for idx, (k, tw, seed) in enumerate(combos_2B, 1):
                    rpath = os.path.join(raw_dir, f"2B_model={cfg.name}_k={k}_tw={tw}_seed={seed}")
                    if args.skip_existing and os.path.exists(rpath + ".json"):
                        logger.info(
                            "[2B %d/%d] SKIP %s k=%d tw=%s seed=%d",
                            idx,
                            len(combos_2B),
                            cfg.name,
                            k,
                            tw,
                            seed,
                        )
                        all_results.append(load_run_result(rpath))
                        continue
                    logger.info(
                        "[2B %d/%d] model=%s k=%d frac=%d%% tw=%s seed=%d",
                        idx,
                        len(combos_2B),
                        cfg.name,
                        k,
                        _S2B_FRACTION,
                        tw,
                        seed,
                    )
                    try:
                        result = run_single_cmaes(
                            ctx,
                            k,
                            _S2B_FRACTION,
                            travel_weight=tw,
                            seed=seed,
                            section_tag="2B",
                            target_coverage=args.target_coverage,
                        )
                        result = _add_derived([result])[0]
                        all_results.append(result)
                        save_run_result(result, rpath)
                        logger.info(
                            "  VPs=%d cov=%.1f%% iter=%d/%d t=%.1fs",
                            result["num_viewpoints"],
                            result["coverage"] * 100,
                            result["n_iter_actual"],
                            result["n_iter_requested"],
                            result["total_time"],
                        )
                    except Exception as e:
                        logger.error("  FAILED: %s", e, exc_info=True)
                    finally:
                        free_gpu_memory()

                # Sub-C: popsize x maxiter (k=_S2C_K, fraction=_S2C_FRACTION;
                # tw fixed per model group)
                s2c_tw = _s2c_travel_weight_for(cfg)
                combos_2C = [
                    (pop, mi, seed)
                    for pop in E03_C_POPSIZE_VALUES
                    for mi in E03_C_MAXITER_VALUES
                    for seed in args.seeds
                ]
                logger.info(
                    "  Sub-C: %d combos (popsize × maxiter, k=%d frac=%d%% tw=%.2f)",
                    len(combos_2C),
                    _S2C_K,
                    _S2C_FRACTION,
                    s2c_tw,
                )
                for idx, (pop, mi, seed) in enumerate(combos_2C, 1):
                    rpath = os.path.join(
                        raw_dir, f"2C_model={cfg.name}_pop={pop}_mi={mi}_seed={seed}"
                    )
                    if args.skip_existing and os.path.exists(rpath + ".json"):
                        logger.info(
                            "[2C %d/%d] SKIP %s pop=%d mi=%d seed=%d",
                            idx,
                            len(combos_2C),
                            cfg.name,
                            pop,
                            mi,
                            seed,
                        )
                        all_results.append(load_run_result(rpath))
                        continue
                    logger.info(
                        "[2C %d/%d] model=%s k=%d frac=%d%% tw=%.2f pop=%d mi=%d seed=%d",
                        idx,
                        len(combos_2C),
                        cfg.name,
                        _S2C_K,
                        _S2C_FRACTION,
                        s2c_tw,
                        pop,
                        mi,
                        seed,
                    )
                    try:
                        result = run_single_cmaes(
                            ctx,
                            _S2C_K,
                            _S2C_FRACTION,
                            travel_weight=s2c_tw,
                            seed=seed,
                            section_tag="2C",
                            target_coverage=args.target_coverage,
                            popsize=pop,
                            maxiter=mi,
                        )
                        result = _add_derived([result])[0]
                        all_results.append(result)
                        save_run_result(result, rpath)
                        logger.info(
                            "  VPs=%d cov=%.1f%% iter=%d/%d t=%.1fs",
                            result["num_viewpoints"],
                            result["coverage"] * 100,
                            result["n_iter_actual"],
                            result["n_iter_requested"],
                            result["total_time"],
                        )
                    except Exception as e:
                        logger.error("  FAILED: %s", e, exc_info=True)
                    finally:
                        free_gpu_memory()

    else:
        # Load all saved results
        for fname in sorted(os.listdir(raw_dir)):
            if fname.endswith(".json"):
                r = load_run_result(os.path.join(raw_dir, fname.replace(".json", "")))
                all_results.append(r)
        _add_derived(all_results)

    if all_results:
        setup_thesis_style()
        fig_dir = os.path.join(args.output_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)

        tosca_names = set(TOSCA_REPRESENTATIVE)
        groups = [
            ("tosca", "TOSCA avg", [r for r in all_results if r.get("model") in tosca_names]),
            ("duke", "Duke", [r for r in all_results if r.get("model") == "duke_of_lancaster"]),
        ]

        # Combined Duke + TOSCA figures (used by the thesis chapter).
        non_empty = [(t, l, r) for (t, l, r) in groups if r]
        if non_empty:
            if run_s1 or args.plots_only:
                generate_plots_section1_combined(non_empty, E03_T_SPI_VALUES, fig_dir)
            if run_s2 or args.plots_only:
                generate_plots_section2_combined(
                    non_empty,
                    E03_C_K_VALUES,
                    E03_C_POPSIZE_VALUES,
                    E03_C_MAXITER_VALUES,
                    fig_dir,
                    target_coverage=args.target_coverage,
                )

        for _tag, label, group_results in groups:
            if not group_results:
                continue

            # ── Summary tables (per group) ────────────────────────────────
            r2B = [r for r in group_results if r.get("section") == "2B"]
            logger.info(
                "\n%s\nE03 SECTION 2B SUMMARY — %s (CMA-ES: k × travel_weight, frac=100%%)\n%s",
                "=" * 80,
                label,
                "=" * 80,
            )
            if r2B:
                tws = sorted(set(r["travel_weight"] for r in r2B))
                logger.info("%-4s %-6s %8s %10s %10s", "k", "tw", "VPs", "Coverage%", "Time(s)")
                logger.info("-" * 50)
                for k in E03_C_K_VALUES:
                    for tw in tws:
                        rows = [r for r in r2B if r["k_coverage"] == k and r["travel_weight"] == tw]
                        if rows:
                            logger.info(
                                "%-4d %-6.3f %8.0f %10.2f %10.1f",
                                k,
                                tw,
                                np.mean([r["num_viewpoints"] for r in rows]),
                                np.mean([r["coverage"] * 100 for r in rows]),
                                np.mean([r["total_time"] for r in rows]),
                            )

            r2C = [r for r in group_results if r.get("section") == "2C"]
            s2c_tw = r2C[0]["travel_weight"] if r2C else None
            logger.info(
                "\n%s\nE03 SECTION 2C SUMMARY — %s (CMA-ES: popsize × maxiter, tw=%s)\n%s",
                "=" * 80,
                label,
                s2c_tw,
                "=" * 80,
            )
            if r2C:
                logger.info("%-8s %-8s %8s %10s %10s", "pop", "mi", "VPs", "Coverage%", "Time(s)")
                logger.info("-" * 50)
                for pop in E03_C_POPSIZE_VALUES:
                    for mi in E03_C_MAXITER_VALUES:
                        rows = [r for r in r2C if r["popsize"] == pop and r["maxiter"] == mi]
                        if rows:
                            logger.info(
                                "%-8d %-8d %8.0f %10.2f %10.1f",
                                pop,
                                mi,
                                np.mean([r["num_viewpoints"] for r in rows]),
                                np.mean([r["coverage"] * 100 for r in rows]),
                                np.mean([r["total_time"] for r in rows]),
                            )


if __name__ == "__main__":
    main()
