#!/usr/bin/env python3
"""E4: Set Cover Optimizer Comparison

Two sections:

Section A — Optimizer comparison (fixed input strategy: targeted_50):
  Optimizers compared:
    GreedySetCover              — CPU greedy, O(N·M) per step
    GreedySetCoverCuda          — GPU greedy, ~30× faster than CPU for large N
    LazyGreedySetCover          — CPU lazy greedy, O(log N) heap — fastest for N~1500
    LazyGreedySetCoverCuda      — GPU lazy greedy, O(N) argmax scan — slower than CPU
    ExpansionIterative_weighted         — LazyGreedyCuda + weighted expansion
    ExpansionIterative_weighted_curvature — LazyGreedyCuda + curvature-weighted expansion
    ExpansionIterative_cmaes            — LazyGreedyCuda + CMA-ES expansion

  Note: GPU LazyGreedy uses O(N) argmax scan (no heap) — slower than CPU O(log N)
  for N~1500. GreedyCuda is fast because it evaluates all N candidates in one CUDA
  kernel per step. Reference: e04 results confirm LazyGreedy-CPU is fastest for N≤2K.

Section B — Input strategy robustness (all 10 sampling strategies × 4 key optimizers):
  Shows whether optimizer ranking (timing and solution quality) holds regardless of
  how the candidate pool was generated. Each strategy generates different candidates
  (different N, quality, spatial distribution) — this tests optimizer robustness.

Lower bound: LP relaxation (tightest valid lower bound) enforcing coverage on only
  ceil(M × target_coverage) hardest-to-cover points. See common/lower_bounds.py.

Usage:
    conda run -n isaaclab python -m experiments.e04_set_cover_optimizers
    conda run -n isaaclab python -m experiments.e04_set_cover_optimizers --section A
    conda run -n isaaclab python -m experiments.e04_set_cover_optimizers --section B
    conda run -n isaaclab python -m experiments.e04_set_cover_optimizers --plots_only
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
    ModelConfig, SEEDS_3, SEEDS_5, E04_COVERAGE_TARGETS,
    E04_OPTIMIZERS_A, E04_OPTIMIZERS_B, E04_INPUT_STRATEGIES,
    TOSCA_REPRESENTATIVE, RESULTS_DIR,
)
from experiments.common.runner import set_seed, timed, free_gpu_memory
from experiments.common.pipeline_setup import PipelineContext, DegenerateNormalsError
from experiments.common.persistence import save_run_result, load_run_result
from experiments.common.lower_bounds import lp_relaxation_set_cover, information_theoretic_lb
from experiments.common.sampling_dispatch import sample_strategy
from experiments.common.plotting import (
    setup_thesis_style, save_figure, grouped_bar,
    heatmap_annotated, THESIS_COL, DOUBLE_COL, CATEGORICAL_COLORS,
)
from shared.types import Side

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Expansion adapter (adapts TargetedViewpointSampler to ProbabilisticSampler
# interface for ProbabilisticExpansionSampler)
# ═══════════════════════════════════════════════════════════════════════════

class _WeightedExpansionAdapter:
    """Wraps a TargetedViewpointSampler and exposes ProbabilisticSampler.sample(n).

    ProbabilisticExpansionSampler.refine() calls self.sampler.sample(n_samples)
    with no other arguments.  TargetedViewpointSampler.sample() requires
    uncovered_indices as the first arg, so we call the parent class method.
    """

    def __init__(self, sampler, curvature_weighting: bool = False):
        self._s = sampler
        self._cw = curvature_weighting

    def restrict_to_sphere(self, center, radius):
        self._s.restrict_to_sphere(center, radius)

    def clear_restriction(self):
        self._s.clear_restriction()

    def sample(self, num_candidates: int):
        from visibility.sampling.samplers.weighted import WeightedViewpointSampler
        return WeightedViewpointSampler.sample(
            self._s, num_candidates,
            side=Side.OUTSIDE,
            curvature_weighting=self._cw,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Optimizer factory
# ═══════════════════════════════════════════════════════════════════════════

def _make_optimizer(name: str, ctx: PipelineContext, vis_query,
                    num_points: int, pos_gpu, rot_gpu, V_gpu, V_np):
    """Instantiate a set cover optimizer.

    Args:
        name:       optimizer identifier (see E04_OPTIMIZERS_A / _B)
        ctx:        PipelineContext (needed for expansion samplers)
        vis_query:  GPU visibility query (needed for expansion samplers)
        V_gpu:      CuPy (N, M) visibility matrix
        V_np:       numpy (N, M) visibility matrix
        pos_gpu / rot_gpu: CuPy candidate positions/rotations
    """
    from visibility.set_cover import (
        GreedySetCover, GreedySetCoverCuda,
        LazyGreedySetCover, LazyGreedySetCoverCuda,
        ExpansionIterativeSetCover,
    )

    pos_np = cp.asnumpy(pos_gpu)
    rot_np = cp.asnumpy(rot_gpu)

    if name == "GreedySetCoverCuda":
        return GreedySetCoverCuda(num_points, pos_gpu, rot_gpu, V_gpu)

    elif name == "LazyGreedySetCoverCuda":
        return LazyGreedySetCoverCuda(num_points, pos_gpu, rot_gpu, V_gpu)

    elif name == "GreedySetCover":
        return GreedySetCover(num_points, pos_np, rot_np, V_np)

    elif name == "LazyGreedySetCover":
        return LazyGreedySetCover(num_points, pos_np, rot_np, V_np)

    elif name.startswith("ExpansionIterative_"):
        inner_sampler_type = name.split("_", 1)[1]  # "weighted", "weighted_curvature", "cmaes"
        inner = LazyGreedySetCoverCuda(num_points, pos_gpu, rot_gpu, V_gpu)

        if inner_sampler_type in ("weighted", "weighted_curvature"):
            from visibility.sampling.samplers.expansion import ProbabilisticExpansionSampler
            curvature = (inner_sampler_type == "weighted_curvature")
            raw_sampler = ctx.build_sampler("targeted")
            adapter = _WeightedExpansionAdapter(raw_sampler, curvature_weighting=curvature)
            exp_sampler = ProbabilisticExpansionSampler(
                adapter, vis_query, n_samples=20, radius=0.5)

        elif inner_sampler_type == "cmaes":
            from visibility.sampling.samplers.expansion import OptimizingExpansionSampler
            opt_sampler = ctx.build_sampler("optimizing")
            exp_sampler = OptimizingExpansionSampler(opt_sampler, vis_query, radius=0.5)

        else:
            raise ValueError(f"Unknown expansion inner sampler: {inner_sampler_type}")

        return ExpansionIterativeSetCover(inner, exp_sampler)

    else:
        raise ValueError(f"Unknown optimizer: {name!r}")


# ═══════════════════════════════════════════════════════════════════════════
# Single run
# ═══════════════════════════════════════════════════════════════════════════

def _save_viz(opt_result, target_points, normals, viz_path, meta):
    """Save set-cover viz data (positions, rotations, visibility_map, points) for replay."""
    data = {
        "positions": opt_result.positions,
        "rotations": opt_result.rotations,
        "visibility_map": opt_result.visibility_map,
        "target_points": np.asarray(target_points, dtype=np.float32),
        "normals": np.asarray(normals, dtype=np.float32),
        **meta,
    }
    save_run_result(data, viz_path)


def run_single(ctx: PipelineContext, optimizer_name: str, input_strategy: str,
               target_coverage: float, seed: int,
               pos_gpu, rot_gpu, V_gpu, V_np, num_points: int,
               viz_path: str | None = None,
               target_points_viz=None, normals_viz=None) -> dict:
    """Run one optimizer on pre-generated candidates.

    Candidates are passed in to avoid recomputing across optimizers.
    """
    vis_query = ctx.build_visibility_query("raycast")
    optimizer = _make_optimizer(
        optimizer_name, ctx, vis_query,
        num_points, pos_gpu, rot_gpu, V_gpu, V_np,
    )
    with timed() as t_opt:
        opt_result = optimizer.optimize(
            target_coverage=target_coverage, max_viewpoints=1000)

    if viz_path is not None and target_points_viz is not None:
        _save_viz(opt_result, target_points_viz, normals_viz, viz_path, {
            "optimizer": optimizer_name,
            "target_coverage": target_coverage,
            "model": ctx.model.name,
            "seed": seed,
            "num_candidates": int(len(pos_gpu)),
            "num_viewpoints": opt_result.num_viewpoints,
            "coverage": float(opt_result.total_coverage),
        })

    return {
        "model": ctx.model.name,
        "optimizer": optimizer_name,
        "input_strategy": input_strategy,
        "target_coverage": target_coverage,
        "seed": seed,
        "num_candidates": int(len(pos_gpu)),
        "num_viewpoints": opt_result.num_viewpoints,
        "coverage": float(opt_result.total_coverage),
        "optimization_time": t_opt.elapsed,
        "redundancy": float(opt_result.redundancy),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Lower bound computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_lower_bounds(V_np: np.ndarray, num_points: int,
                         coverage_targets: list) -> dict:
    """Compute LP relaxation and info-theoretic lower bounds."""
    V_bool = V_np.astype(np.bool_)
    max_single_cov = int(V_bool.sum(axis=1).max())
    bounds = {}
    for target in coverage_targets:
        lp_lb = lp_relaxation_set_cover(V_bool, target)
        info_lb = information_theoretic_lb(num_points, target, max_single_cov)
        best = max(lp_lb, info_lb)
        bounds[target] = {"lp_relaxation": lp_lb, "info_theoretic": info_lb, "best": best}
        logger.info("  LB at %.0f%%: LP=%d, Info=%d, Best=%d",
                    target * 100, lp_lb, info_lb, best)
    return bounds


# ═══════════════════════════════════════════════════════════════════════════
# Plot generation
# ═══════════════════════════════════════════════════════════════════════════

def _short(optimizer_name: str) -> str:
    """Short label for plots."""
    return (optimizer_name
            .replace("SetCoverCuda", "Cuda")
            .replace("SetCover", "")
            .replace("ExpansionIterative_", "Exp_"))


def generate_plots_A(results: list[dict], lower_bounds: dict,
                     optimizers: list[str], fig_dir: str):
    """Section A plots: optimizer comparison on targeted_50 input."""
    mr = [r for r in results if r.get("section") == "A"]
    if not mr:
        return

    models = sorted(set(r["model"] for r in mr))
    targets = sorted(set(r["target_coverage"] for r in mr))
    target_labels = [f"{t*100:.0f}%" for t in targets]
    opt_labels = [_short(o) for o in optimizers]

    def _mean(opt, target, metric, model=None):
        rows = [r for r in mr if r["optimizer"] == opt and r["target_coverage"] == target]
        if model:
            rows = [r for r in rows if r["model"] == model]
        vals = [r[metric] for r in rows]
        return float(np.mean(vals)) if vals else float("nan")

    for model_name in models:
        model_mr = [r for r in mr if r["model"] == model_name]
        present_opts = [o for o in optimizers
                        if any(r["optimizer"] == o for r in model_mr)]
        present_labels = [_short(o) for o in present_opts]

        # ── Fig 1: Timing — grouped bars per target ──────────────────────
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4))
        timing_data = {
            _short(o): [_mean(o, t, "optimization_time", model_name) for t in targets]
            for o in present_opts
        }
        grouped_bar(ax, timing_data, target_labels,
                    ylabel="Optimization time (s)",
                    title=f"Optimizer Timing ({model_name})\n"
                          "Note: LazyGreedy-CPU beats LazyGreedy-GPU (O(log N) heap vs O(N) scan)")
        ax.set_xlabel("Target coverage")
        save_figure(fig, os.path.join(fig_dir, f"{model_name}_e04_A_timing"))

        # ── Fig 2: Viewpoints at 95% target ──────────────────────────────
        fig, ax = plt.subplots(figsize=(THESIS_COL, 3.5))
        target_95 = min(targets, key=lambda t: abs(t - 0.95))
        vp_means = [_mean(o, target_95, "num_viewpoints", model_name) for o in present_opts]
        vp_stds = []
        for o in present_opts:
            vals = [r["num_viewpoints"] for r in model_mr
                    if r["optimizer"] == o and r["target_coverage"] == target_95]
            vp_stds.append(float(np.std(vals)) if vals else 0.0)

        x = np.arange(len(present_opts))
        ax.bar(x, vp_means, yerr=vp_stds, capsize=3, alpha=0.85,
               color=[CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)]
                      for i in range(len(present_opts))])

        # LP lower bound line
        lb = lower_bounds.get(target_95, {}).get("best", 0)
        if lb > 0:
            ax.axhline(lb, color="red", linestyle="--", alpha=0.7,
                       label=f"LP LB = {lb}")
            ax.legend(fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(present_labels, rotation=30, ha="right")
        ax.set_ylabel("Selected viewpoints")
        ax.set_title(f"Viewpoints at {target_95*100:.0f}% ({model_name})")
        save_figure(fig, os.path.join(fig_dir, f"{model_name}_e04_A_viewpoints"))

        # ── Fig 3: Optimality ratio (viewpoints / LP lower bound) ────────
        if lower_bounds:
            fig, ax = plt.subplots(figsize=(THESIS_COL, 3))
            for ti, target in enumerate(targets):
                lb = lower_bounds.get(target, {}).get("best", 0)
                if lb <= 0:
                    continue
                ratios = [_mean(o, target, "num_viewpoints", model_name) / lb
                          for o in present_opts]
                if ti == 0:
                    ax.bar(x - 0.2, ratios, width=0.35,
                           color=CATEGORICAL_COLORS[ti], alpha=0.7,
                           label=f"{target*100:.0f}%")
                else:
                    ax.bar(x + 0.2, ratios, width=0.35,
                           color=CATEGORICAL_COLORS[ti], alpha=0.7,
                           label=f"{target*100:.0f}%")
            ax.axhline(1.0, color="black", linestyle="--", alpha=0.4, linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(present_labels, rotation=30, ha="right")
            ax.set_ylabel("Viewpoints / LP lower bound")
            ax.set_title(f"Optimality Ratio ({model_name})")
            ax.legend(fontsize=8)
            save_figure(fig, os.path.join(fig_dir, f"{model_name}_e04_A_optgap"))

        # ── Fig 4: GPU speedup (CPU/GPU time ratio) ───────────────────────
        fig, ax = plt.subplots(figsize=(THESIS_COL, 3))
        pairs = [("GreedySetCover", "GreedySetCoverCuda"),
                 ("LazyGreedySetCover", "LazyGreedySetCoverCuda")]
        pair_labels, speedups = [], []
        for cpu_name, gpu_name in pairs:
            cpu_t = [r["optimization_time"] for r in model_mr if r["optimizer"] == cpu_name]
            gpu_t = [r["optimization_time"] for r in model_mr if r["optimizer"] == gpu_name]
            if cpu_t and gpu_t:
                speedups.append(np.mean(cpu_t) / np.mean(gpu_t))
                pair_labels.append(cpu_name.replace("SetCover", ""))
        if speedups:
            bars = ax.bar(pair_labels, speedups,
                          color=[CATEGORICAL_COLORS[0], CATEGORICAL_COLORS[2]])
            ax.bar_label(bars, fmt="%.2fx", fontsize=9, padding=3)
            ax.axhline(1.0, color="gray", linestyle="--", alpha=0.4)
            ax.set_ylabel("Speedup (CPU time / GPU time)")
            ax.set_title(f"GPU Speedup ({model_name})\n"
                         "Greedy: GPU wins (~30×).  LazyGreedy: CPU wins (heap vs scan).")
            save_figure(fig, os.path.join(fig_dir, f"{model_name}_e04_A_speedup"))

    logger.info("Section A figures saved to %s", fig_dir)


def generate_plots_B(results: list[dict], optimizers: list[str],
                     strategies: list[str], fig_dir: str):
    """Section B plots: optimizer robustness across input strategies."""
    br = [r for r in results if r.get("section") == "B"]
    if not br:
        return

    models = sorted(set(r["model"] for r in br))
    target_95 = 0.95
    close_targets = sorted(set(r["target_coverage"] for r in br))
    target_main = min(close_targets, key=lambda t: abs(t - target_95))

    opt_labels = [_short(o) for o in optimizers]
    strat_labels = strategies

    def _mean_vp(model, opt, strat):
        vals = [r["num_viewpoints"] for r in br
                if r["model"] == model and r["optimizer"] == opt
                and r["input_strategy"] == strat
                and abs(r["target_coverage"] - target_main) < 0.01]
        return float(np.mean(vals)) if vals else float("nan")

    def _mean_time(model, opt, strat):
        vals = [r["optimization_time"] for r in br
                if r["model"] == model and r["optimizer"] == opt
                and r["input_strategy"] == strat
                and abs(r["target_coverage"] - target_main) < 0.01]
        return float(np.mean(vals)) if vals else float("nan")

    for model_name in models:
        present_opts = [o for o in optimizers
                        if any(r["optimizer"] == o and r["model"] == model_name for r in br)]
        present_strats = [s for s in strategies
                          if any(r["input_strategy"] == s and r["model"] == model_name
                                 for r in br)]
        if not present_opts or not present_strats:
            continue

        # ── Fig 5: Heatmap — input strategy × optimizer → viewpoints ────
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, max(3, len(present_strats) * 0.4 + 1)))
        vals = np.array([[_mean_vp(model_name, o, s)
                          for o in present_opts]
                         for s in present_strats])
        heatmap_annotated(
            ax,
            present_strats,
            [_short(o) for o in present_opts],
            vals, fmt=".0f",
            title=f"Viewpoints at {target_main*100:.0f}% ({model_name}) — Input vs Optimizer",
            xlabel="Optimizer", ylabel="Input strategy",
        )
        save_figure(fig, os.path.join(fig_dir, f"{model_name}_e04_B_vp_heatmap"))

        # ── Fig 6: Heatmap — timing ───────────────────────────────────────
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, max(3, len(present_strats) * 0.4 + 1)))
        t_vals = np.array([[_mean_time(model_name, o, s)
                            for o in present_opts]
                           for s in present_strats])
        heatmap_annotated(
            ax,
            present_strats,
            [_short(o) for o in present_opts],
            t_vals, fmt=".3f",
            title=f"Optimization Time (s) ({model_name})",
            xlabel="Optimizer", ylabel="Input strategy",
        )
        save_figure(fig, os.path.join(fig_dir, f"{model_name}_e04_B_timing_heatmap"))

        # ── Fig 7: Grouped bar — viewpoints by input strategy ────────────
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4))
        vp_data = {
            _short(o): [_mean_vp(model_name, o, s) for s in present_strats]
            for o in present_opts
        }
        grouped_bar(ax, vp_data, present_strats,
                    ylabel="Selected viewpoints",
                    title=f"Optimizer Ranking by Input Strategy ({model_name})")
        ax.tick_params(axis="x", rotation=40)
        ax.set_xlabel("Input sampling strategy")
        save_figure(fig, os.path.join(fig_dir, f"{model_name}_e04_B_ranking"))

    logger.info("Section B figures saved to %s", fig_dir)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="E4: Set Cover Optimizer Comparison")
    p.add_argument("--section", choices=["A", "B", "both"], default="both")
    p.add_argument("--models", nargs="+",
                   default=["duke_of_lancaster"] + TOSCA_REPRESENTATIVE)
    p.add_argument("--optimizers_A", nargs="+", default=E04_OPTIMIZERS_A)
    p.add_argument("--optimizers_B", nargs="+", default=E04_OPTIMIZERS_B)
    p.add_argument("--strategies_B", nargs="+", default=E04_INPUT_STRATEGIES)
    p.add_argument("--targets", type=float, nargs="+", default=E04_COVERAGE_TARGETS)
    p.add_argument("--seeds_A", type=int, nargs="+", default=SEEDS_5)
    p.add_argument("--seeds_B", type=int, nargs="+", default=SEEDS_3)
    p.add_argument("--output_dir",
                   default=os.path.join(RESULTS_DIR, "e04_set_cover_optimizers"))
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--plots_only", action="store_true")
    p.add_argument("--skip_lower_bounds", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
    )

    raw_dir = os.path.join(args.output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    all_results: list[dict] = []
    lower_bounds: dict = {}

    run_A = args.section in ("A", "both")
    run_B = args.section in ("B", "both")

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

            # ── Section A: fixed input=targeted_50, all optimizers ─────
            if run_A:
                logger.info("--- Section A (targeted_50) ---")
                for seed in args.seeds_A:
                    logger.info("  seed=%d: generating candidates (targeted_50)", seed)
                    set_seed(seed)
                    target_points, normals = ctx.sample_surface()
                    vis_query = ctx.build_visibility_query("raycast")

                    with timed() as t_cand:
                        pos_gpu, rot_gpu, n_base, n_iter, _ = sample_strategy(
                            ctx, "targeted_50", cfg.num_candidates,
                            target_points, normals, vis_query, cfg,
                        )
                    V_gpu, _ = vis_query.compute_visibility_batch(pos_gpu, rot_gpu)
                    V_np = cp.asnumpy(V_gpu)
                    num_points = int(len(target_points))
                    logger.info("  %d candidates (%.1fs)", len(pos_gpu), t_cand.elapsed)

                    # Compute lower bounds once (first seed only, Duke only for speed)
                    if (not args.skip_lower_bounds and seed == args.seeds_A[0]
                            and model_name == "duke_of_lancaster" and not lower_bounds):
                        logger.info("  Computing LP lower bounds...")
                        lower_bounds = compute_lower_bounds(V_np, num_points, args.targets)

                    for opt_name in args.optimizers_A:
                        for target in args.targets:
                            rpath = os.path.join(
                                raw_dir,
                                f"A_model={model_name}_opt={opt_name}"
                                f"_target={target}_seed={seed}")
                            if args.skip_existing and os.path.exists(rpath + ".json"):
                                r = load_run_result(rpath)
                                r.setdefault("section", "A")
                                all_results.append(r)
                                continue

                            logger.info("  [A] %s target=%.2f", opt_name, target)
                            try:
                                viz_path = None
                                if (model_name == "duke_of_lancaster"
                                        and seed == SEEDS_3[0]
                                        and abs(target - 0.95) < 1e-6):
                                    viz_dir = os.path.join(args.output_dir, "viz")
                                    os.makedirs(viz_dir, exist_ok=True)
                                    viz_path = os.path.join(
                                        viz_dir,
                                        f"opt={opt_name}_target={target}_seed={seed}")
                                result = run_single(
                                    ctx, opt_name, "targeted_50", target, seed,
                                    pos_gpu, rot_gpu, V_gpu, V_np, num_points,
                                    viz_path=viz_path,
                                    target_points_viz=target_points,
                                    normals_viz=normals)
                                result["section"] = "A"
                                all_results.append(result)
                                save_run_result(result, rpath)
                                logger.info("    VPs=%d cov=%.2f%% t=%.2fs",
                                            result["num_viewpoints"],
                                            result["coverage"] * 100,
                                            result["optimization_time"])
                            except Exception as e:
                                logger.error("    FAILED: %s", e, exc_info=True)
                            finally:
                                free_gpu_memory()

            # ── Section B: all input strategies, key optimizers ────────
            if run_B:
                logger.info("--- Section B (all input strategies) ---")
                for strategy in args.strategies_B:
                    for seed in args.seeds_B:
                        logger.info("  seed=%d input=%s: generating candidates",
                                    seed, strategy)
                        set_seed(seed)
                        target_points, normals = ctx.sample_surface()
                        vis_query = ctx.build_visibility_query("raycast")

                        try:
                            pos_gpu, rot_gpu, _, _, _ = sample_strategy(
                                ctx, strategy, cfg.num_candidates,
                                target_points, normals, vis_query, cfg,
                            )
                        except Exception as e:
                            logger.error("  Candidate gen FAILED: %s", e, exc_info=True)
                            continue

                        V_gpu, _ = vis_query.compute_visibility_batch(pos_gpu, rot_gpu)
                        V_np = cp.asnumpy(V_gpu)
                        num_points = int(len(target_points))

                        for opt_name in args.optimizers_B:
                            for target in args.targets:
                                rpath = os.path.join(
                                    raw_dir,
                                    f"B_model={model_name}_opt={opt_name}"
                                    f"_strat={strategy}_target={target}_seed={seed}")
                                if args.skip_existing and os.path.exists(rpath + ".json"):
                                    r = load_run_result(rpath)
                                    r.setdefault("section", "B")
                                    all_results.append(r)
                                    continue

                                try:
                                    result = run_single(
                                        ctx, opt_name, strategy, target, seed,
                                        pos_gpu, rot_gpu, V_gpu, V_np, num_points)
                                    result["section"] = "B"
                                    all_results.append(result)
                                    save_run_result(result, rpath)
                                except Exception as e:
                                    logger.error("  [B] %s %s FAILED: %s",
                                                 opt_name, strategy, e, exc_info=True)
                                finally:
                                    free_gpu_memory()

    else:
        for fname in sorted(os.listdir(raw_dir)):
            if fname.endswith(".json"):
                r = load_run_result(os.path.join(raw_dir, fname.replace(".json", "")))
                # Infer section from filename if not stored
                if "section" not in r:
                    r["section"] = "A" if fname.startswith("A_") else "B"
                all_results.append(r)

    if all_results:
        setup_thesis_style()
        fig_dir = os.path.join(args.output_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)

        if run_A:
            generate_plots_A(all_results, lower_bounds, args.optimizers_A, fig_dir)
        if run_B:
            generate_plots_B(all_results, args.optimizers_B, args.strategies_B, fig_dir)


if __name__ == "__main__":
    main()
