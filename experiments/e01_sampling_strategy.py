#!/usr/bin/env python3
"""E1: Sampling Strategy Comparison.

Compares 3 strategies on Duke + TOSCA_REPRESENTATIVE × 3 seeds:
  weighted             – SDF^2 uniform
  weighted_curvature   – SDF^2 + curvature bias
  cmaes_100            – 100% CMA-ES (k_coverage=3, popsize=40, maxiter=40,
                         travel_weight=0.1 TOSCA / 0.0 Duke; e03 §2B)

The k_coverage and k×travel_weight sweeps live in e03; this experiment fixes the CMA-ES
hyperparameters and asks how the three samplers compare. Targeted sampler excluded — it
does not outperform weighted_curvature (e03). Set-cover: LazyGreedySetCover (CPU, fastest per e06).

    conda run -n isaaclab python -m experiments.e01_sampling_strategy
    conda run -n isaaclab python -m experiments.e01_sampling_strategy --plots_only
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
    RESULTS_DIR,
    SEEDS_3,
    TOSCA_REPRESENTATIVE,
    ModelConfig,
)

_RUNTIME_IMPORT_ERROR: ImportError | None = None
try:
    import cupy as cp

    from experiments.common.pipeline_setup import (
        DegenerateNormalsError,
        PipelineContext,
    )
    from experiments.common.runner import free_gpu_memory, set_seed, timed
    from experiments.common.sampling_dispatch import sample_strategy
    from visibility.set_cover import LazyGreedySetCover

    _RUNTIME_AVAILABLE = True
except ImportError as _e:
    cp = None
    set_seed = timed = free_gpu_memory = None
    PipelineContext = DegenerateNormalsError = None
    sample_strategy = None
    LazyGreedySetCover = None
    _RUNTIME_AVAILABLE = False
    _RUNTIME_IMPORT_ERROR = _e

# Targeted sampler intentionally excluded: never outperforms weighted_curvature at k=1 (e03 §1).
_E01_STRATEGIES = ["weighted", "weighted_curvature", "cmaes_100"]

# Thesis-final CMA-ES configuration (e03 §2B).
_E01_K_COVERAGE = 3
_E01_CMAES_POPSIZE = 40
_E01_CMAES_MAXITER = 40
_E01_CMAES_TRAVEL_WEIGHT_DUKE = 0.0
_E01_CMAES_TRAVEL_WEIGHT_TOSCA = 0.1


def _strategy_kwargs(strategy: str, model_name: str) -> dict:
    """``weighted`` / ``weighted_curvature`` are one-shot — kwargs ignored in dispatch's one-shot
    branch. ``cmaes_100`` pins thesis-final hyperparameters; travel_weight per model group (e03 §2B)."""
    if strategy == "cmaes_100":
        tw = (
            _E01_CMAES_TRAVEL_WEIGHT_DUKE
            if model_name == "duke_of_lancaster"
            else _E01_CMAES_TRAVEL_WEIGHT_TOSCA
        )
        return {
            "k_coverage": _E01_K_COVERAGE,
            "popsize": _E01_CMAES_POPSIZE,
            "maxiter": _E01_CMAES_MAXITER,
            "travel_weight": tw,
        }
    return {}


from experiments.common.persistence import load_run_result, save_run_result
from experiments.common.plotting import (
    DOUBLE_COL,
    display_strategy,
    grouped_bar,
    panel_title,
    save_figure,
    setup_thesis_style,
    stacked_bar,
)


def _display_model(name: str) -> str:
    return "duke" if name == "duke_of_lancaster" else name


logger = logging.getLogger(__name__)


def _set_cover(target_points, pos_gpu, rot_gpu, V, target_coverage):
    V_np = cp.asnumpy(V)
    pos_np = cp.asnumpy(pos_gpu)
    rot_np = cp.asnumpy(rot_gpu)
    optimizer = LazyGreedySetCover(len(target_points), pos_np, rot_np, V_np)
    return optimizer.optimize(target_coverage=target_coverage, max_viewpoints=1000), V_np


def _save_viz(opt_result, target_points, normals, viz_path, meta):
    data = {
        "positions": opt_result.positions,
        "rotations": opt_result.rotations,
        "visibility_map": opt_result.visibility_map,
        "target_points": np.asarray(target_points.get(), dtype=np.float32),
        "normals": np.asarray(normals.get(), dtype=np.float32),
        **meta,
    }
    save_run_result(data, viz_path)


def run_single_A(
    ctx: PipelineContext,
    strategy: str,
    seed: int,
    target_coverage: float = 0.95,
    viz_path: str | None = None,
) -> dict:
    """Fixed candidate budget (model.num_candidates), thesis-final CMA-ES hyperparameters."""
    target_points, normals = ctx.sample_surface()
    set_seed(seed)
    vis_query = ctx.build_visibility_query("raycast")
    model = ctx.model

    strat_kw = _strategy_kwargs(strategy, model.name)
    k_recorded = strat_kw.get("k_coverage", 1)

    with timed() as t_sample:
        pos_gpu, rot_gpu, n_base, n_iter, base_name, n_warmstart_fallbacks = sample_strategy(
            ctx,
            strategy,
            model.num_candidates,
            target_points,
            normals,
            vis_query,
            model,
            **strat_kw,
        )

    with timed() as t_vis:
        V, _ = vis_query.compute_visibility_batch(pos_gpu, rot_gpu)

    V_np = cp.asnumpy(V)
    pool_redundancy = float(V_np.sum() / len(target_points))

    with timed() as t_opt:
        opt_result, _ = _set_cover(target_points, pos_gpu, rot_gpu, V, target_coverage)

    if viz_path is not None:
        _save_viz(
            opt_result,
            target_points,
            normals,
            viz_path,
            {
                "strategy": strategy,
                "model": model.name,
                "seed": seed,
                "num_candidates": int(len(pos_gpu)),
                "num_viewpoints": opt_result.num_viewpoints,
                "coverage": float(opt_result.total_coverage),
            },
        )

    return {
        "section": "A",
        "model": model.name,
        "strategy": strategy,
        "seed": seed,
        "k_coverage": k_recorded,
        "n_base_candidates": n_base,
        "n_iterative_candidates": n_iter,
        "n_warmstart_fallbacks": int(n_warmstart_fallbacks),
        "base_sampler": base_name,
        "num_candidates_requested": model.num_candidates,
        "num_candidates": int(len(pos_gpu)),
        "num_viewpoints": opt_result.num_viewpoints,
        "coverage": float(opt_result.total_coverage),
        "pool_redundancy": pool_redundancy,
        "redundancy": float(opt_result.redundancy),
        "sampling_time": t_sample.elapsed,
        "visibility_time": t_vis.elapsed,
        "optimization_time": t_opt.elapsed,
        "total_time": t_sample.elapsed + t_vis.elapsed + t_opt.elapsed,
    }


def _per_strategy_mean(rows, strategies, metric, mult=1.0):
    means, stds = [], []
    for s in strategies:
        vals = [r[metric] * mult for r in rows if r["strategy"] == s]
        means.append(float(np.mean(vals)) if vals else float("nan"))
        stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
    return means, stds


def generate_plots_A(results: list[dict], strategies: list[str], fig_dir: str):
    mr = [r for r in results if r["section"] == "A"]
    if not mr:
        return

    short_labels = [display_strategy(s).replace("weighted_curvature", "w_curv") for s in strategies]
    models = sorted(set(r["model"] for r in mr))
    model_labels = [_display_model(m) for m in models]

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4))
    vp_data: dict = {}
    vp_err: dict = {}
    for s in strategies:
        per_model_mean = []
        per_model_std = []
        for m in models:
            vals = [r["num_viewpoints"] for r in mr if r["strategy"] == s and r["model"] == m]
            per_model_mean.append(float(np.mean(vals)) if vals else float("nan"))
            per_model_std.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
        label = display_strategy(s).replace("weighted_curvature", "w_curv")
        vp_data[label] = per_model_mean
        vp_err[label] = per_model_std
    grouped_bar(
        ax,
        vp_data,
        model_labels,
        yerr=vp_err,
        ylabel="Selected viewpoints",
        title="Viewpoints by Model (absolute, per strategy)",
    )
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=30)
    save_figure(fig, os.path.join(fig_dir, "e01_A_by_model_viewpoints"))

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(DOUBLE_COL, 3.8), sharey=True)
    if n_models == 1:
        axes = [axes]
    for ax_m, m, m_label in zip(axes, models, model_labels, strict=False):
        m_rows = [r for r in mr if r["model"] == m]
        stacked_bar(
            ax_m,
            short_labels,
            {
                "Sampling": _per_strategy_mean(m_rows, strategies, "sampling_time")[0],
                "Visibility": _per_strategy_mean(m_rows, strategies, "visibility_time")[0],
                "Optimization": _per_strategy_mean(m_rows, strategies, "optimization_time")[0],
            },
            ylabel="Time (s)" if ax_m is axes[0] else "",
            title="",
        )
        ax_m.set_yscale("log")
        ax_m.tick_params(axis="x", rotation=30)
        panel_title(ax_m, m_label)
    fig.tight_layout()
    save_figure(fig, os.path.join(fig_dir, "e01_A_by_model_timing"))

    logger.info("Section A aggregated figures saved")


def main():
    p = argparse.ArgumentParser(description="E1: Sampling Strategy Comparison")
    p.add_argument(
        "--models",
        nargs="+",
        default=["duke_of_lancaster"] + TOSCA_REPRESENTATIVE,
    )
    p.add_argument("--strategies", nargs="+", default=_E01_STRATEGIES)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_3)
    p.add_argument("--target_coverage", type=float, default=0.95)
    p.add_argument("--output_dir", default=os.path.join(RESULTS_DIR, "e01_sampling_strategy"))
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

    def _make_cfg(name: str):
        if name == "duke_of_lancaster":
            return ModelConfig.duke_of_lancaster()
        try:
            return ModelConfig.tosca(name)
        except FileNotFoundError:
            logger.warning("Model not found: %s", name)
            return None

    if not args.plots_only:
        cfgs = [c for name in args.models if (c := _make_cfg(name)) is not None]
        total = len(cfgs) * len(args.strategies) * len(args.seeds)
        logger.info(
            "E1: %d runs (%d models × %d strategies × %d seeds)",
            total,
            len(cfgs),
            len(args.strategies),
            len(args.seeds),
        )

        for cfg in cfgs:
            logger.info("=" * 60)
            logger.info("Model: %s", cfg.name)
            try:
                ctx = PipelineContext(cfg)
                ctx.load_mesh()
                ctx.sample_surface(seed=42)
                ctx.build_sampling_og()
            except DegenerateNormalsError as e:
                logger.warning("Skipping %s: %s", cfg.name, e)
                continue

            model_combos = [(s, seed) for s in args.strategies for seed in args.seeds]
            for idx, (strategy, seed) in enumerate(model_combos, 1):
                rpath = os.path.join(raw_dir, f"A_model={cfg.name}_strategy={strategy}_seed={seed}")
                if args.resume and os.path.exists(rpath + ".json"):
                    logger.info(
                        "[%d/%d] SKIP %s %s seed=%d",
                        idx,
                        len(model_combos),
                        cfg.name,
                        strategy,
                        seed,
                    )
                    all_results.append(load_run_result(rpath))
                    continue

                logger.info(
                    "[%d/%d] model=%s strategy=%s seed=%d",
                    idx,
                    len(model_combos),
                    cfg.name,
                    strategy,
                    seed,
                )
                try:
                    viz_path = None
                    if cfg.name == "duke_of_lancaster" and seed == SEEDS_3[0]:
                        viz_dir = os.path.join(args.output_dir, "viz")
                        os.makedirs(viz_dir, exist_ok=True)
                        viz_path = os.path.join(viz_dir, f"strategy={strategy}_seed={seed}")
                    result = run_single_A(
                        ctx, strategy, seed, args.target_coverage, viz_path=viz_path
                    )
                    all_results.append(result)
                    save_run_result(result, rpath)
                    logger.info(
                        "  VPs=%d cov=%.2f%% actual=%d time=%.1fs",
                        result["num_viewpoints"],
                        result["coverage"] * 100,
                        result["num_candidates"],
                        result["total_time"],
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
        setup_thesis_style()
        fig_dir = os.path.join(args.output_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)

        a_results = [r for r in all_results if r.get("section") == "A"]
        if a_results:
            generate_plots_A(a_results, args.strategies, fig_dir)

        models = sorted(set(r["model"] for r in a_results))
        logger.info("\n%s\nE1 SUMMARY\n%s", "=" * 80, "=" * 80)
        for model_name in models:
            logger.info("\nModel: %s", model_name)
            logger.info(
                "%-22s %8s %10s %10s %8s %8s",
                "Strategy",
                "VPs",
                "Coverage%",
                "Time(s)",
                "PoolRed",
                "SelRed",
            )
            logger.info("-" * 72)
            mr = [r for r in a_results if r["model"] == model_name]
            for s in args.strategies:
                sr = [r for r in mr if r["strategy"] == s]
                if sr:
                    logger.info(
                        "%-22s %8.0f %10.2f %10.1f %8.2f %8.2f",
                        s,
                        np.mean([r["num_viewpoints"] for r in sr]),
                        np.mean([r["coverage"] * 100 for r in sr]),
                        np.mean([r["total_time"] for r in sr]),
                        np.mean([r["pool_redundancy"] for r in sr]),
                        np.mean([r["redundancy"] for r in sr]),
                    )


if __name__ == "__main__":
    main()
