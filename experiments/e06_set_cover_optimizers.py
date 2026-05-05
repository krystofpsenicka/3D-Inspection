#!/usr/bin/env python3
"""E4: Set Cover Optimizer Comparison.

Section A — fixed input strategy (weighted_curvature):
  GreedySetCover, GreedySetCoverCuda, LazyGreedySetCover, ExpansionIterative_{weighted, weighted_curvature, cmaes}.

Section B — input-strategy robustness: 10 sampling strategies × 4 key optimizers.

Lower bound: greedy matching (anti-chain) + trivial info-theoretic + LP relaxation, computed
once per (model, seed) on the weighted_curvature pool. See common/lower_bounds.py.

    conda run -n isaaclab python -m experiments.e06_set_cover_optimizers
    conda run -n isaaclab python -m experiments.e06_set_cover_optimizers --section A
    conda run -n isaaclab python -m experiments.e06_set_cover_optimizers --section B
    conda run -n isaaclab python -m experiments.e06_set_cover_optimizers --plots_only
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
    E04_COVERAGE_TARGETS,
    E04_INPUT_STRATEGIES,
    E04_OPTIMIZERS_A,
    E04_OPTIMIZERS_B,
    RESULTS_DIR,
    SEEDS_3,
    SEEDS_5,
    TOSCA_REPRESENTATIVE,
    ModelConfig,
)
from experiments.common.lower_bounds import (
    information_theoretic_lb,
    matching_lb_set_cover,
    set_cover_lp_lb,
)
from experiments.common.persistence import load_run_result, save_run_result
from experiments.common.plotting import (
    DOUBLE_COL,
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
    from experiments.common.runner import (
        free_gpu_memory,
        handle_row_exception,
        is_oom,
        set_seed,
        timed,
    )
    from experiments.common.sampling_dispatch import sample_strategy
    from shared.types import Side

    _RUNTIME_AVAILABLE = True
except ImportError as _e:
    cp = None
    set_seed = timed = free_gpu_memory = handle_row_exception = is_oom = None
    PipelineContext = DegenerateNormalsError = None
    sample_strategy = None
    Side = None
    _RUNTIME_AVAILABLE = False
    _RUNTIME_IMPORT_ERROR = _e

logger = logging.getLogger(__name__)


class _WeightedExpansionAdapter:
    """Wrap a TargetedViewpointSampler to expose the ProbabilisticSampler.sample(n) interface
    used by ProbabilisticExpansionSampler.refine()."""

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
            self._s,
            num_candidates,
            side=Side.OUTSIDE,
            curvature_weighting=self._cw,
        )


def _make_optimizer(
    name: str, ctx: PipelineContext, vis_query, num_points: int, pos_gpu, rot_gpu, V_gpu, V_np
):
    from visibility.set_cover import (
        ExpansionIterativeSetCover,
        GreedySetCover,
        GreedySetCoverCuda,
        LazyGreedySetCover,
    )

    pos_np = cp.asnumpy(pos_gpu)
    rot_np = cp.asnumpy(rot_gpu)

    if name == "GreedySetCoverCuda":
        return GreedySetCoverCuda(num_points, pos_gpu, rot_gpu, V_gpu)

    elif name == "GreedySetCover":
        return GreedySetCover(num_points, pos_np, rot_np, V_np)

    elif name == "LazyGreedySetCover":
        return LazyGreedySetCover(num_points, pos_np, rot_np, V_np)

    elif name.startswith("ExpansionIterative_"):
        inner_sampler_type = name.split("_", 1)[1]  # "weighted" / "weighted_curvature" / "cmaes"
        inner = LazyGreedySetCover(num_points, pos_np, rot_np, V_np)

        if inner_sampler_type in ("weighted", "weighted_curvature"):
            from visibility.sampling.samplers.expansion import ProbabilisticExpansionSampler

            curvature = inner_sampler_type == "weighted_curvature"
            raw_sampler = ctx.build_sampler("targeted")
            adapter = _WeightedExpansionAdapter(raw_sampler, curvature_weighting=curvature)
            exp_sampler = ProbabilisticExpansionSampler(
                adapter, vis_query, n_samples=200, radius=0.5
            )

        elif inner_sampler_type == "cmaes":
            from visibility.sampling.samplers.expansion import OptimizingExpansionSampler

            opt_sampler = ctx.build_sampler("optimizing")
            exp_sampler = OptimizingExpansionSampler(opt_sampler, vis_query, radius=0.5)

        else:
            raise ValueError(f"Unknown expansion inner sampler: {inner_sampler_type}")

        return ExpansionIterativeSetCover(inner, exp_sampler)

    else:
        raise ValueError(f"Unknown optimizer: {name!r}")


def _save_viz(opt_result, target_points, normals, viz_path, meta):
    """``target_points`` / ``normals`` may be CuPy when called from the main pipeline — route
    through ``cp.asnumpy`` rather than ``np.asarray`` (which would raise on implicit GPU→CPU copy)."""
    data = {
        "positions": (
            opt_result.positions.get()
            if hasattr(opt_result.positions, "get")
            else opt_result.positions
        ),
        "rotations": (
            opt_result.rotations.get()
            if hasattr(opt_result.rotations, "get")
            else opt_result.rotations
        ),
        "visibility_map": (
            opt_result.visibility_map.get()
            if hasattr(opt_result.visibility_map, "get")
            else opt_result.visibility_map
        ),
        "target_points": (
            cp.asnumpy(target_points).astype(np.float32)
            if hasattr(target_points, "get")
            else np.asarray(target_points, dtype=np.float32)
        ),
        "normals": (
            cp.asnumpy(normals).astype(np.float32)
            if hasattr(normals, "get")
            else np.asarray(normals, dtype=np.float32)
        ),
        **meta,
    }
    save_run_result(data, viz_path)


def run_single(
    ctx: PipelineContext,
    optimizer_name: str,
    input_strategy: str,
    target_coverage: float,
    seed: int,
    pos_gpu,
    rot_gpu,
    V_gpu,
    V_np,
    num_points: int,
    viz_path: str | None = None,
    target_points_viz=None,
    normals_viz=None,
) -> dict:
    """Candidates passed in to avoid recomputing across optimizers."""
    vis_query = ctx.build_visibility_query("raycast")
    optimizer = _make_optimizer(
        optimizer_name,
        ctx,
        vis_query,
        num_points,
        pos_gpu,
        rot_gpu,
        V_gpu,
        V_np,
    )
    with timed() as t_opt:
        opt_result = optimizer.optimize(target_coverage=target_coverage, max_viewpoints=1000)

    if viz_path is not None and target_points_viz is not None:
        _save_viz(
            opt_result,
            target_points_viz,
            normals_viz,
            viz_path,
            {
                "optimizer": optimizer_name,
                "target_coverage": target_coverage,
                "model": ctx.model.name,
                "seed": seed,
                "num_candidates": int(len(pos_gpu)),
                "num_viewpoints": opt_result.num_viewpoints,
                "coverage": float(opt_result.total_coverage),
            },
        )

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


def _lb_cache_path(raw_dir: str, model_name: str, seed: int) -> str:
    return os.path.join(raw_dir, f"lb_model={model_name}_seed={seed}")


def _save_lb(raw_dir: str, model_name: str, seed: int, num_points: int, lb_this: dict) -> None:
    targets = sorted(lb_this.keys())
    record = {
        "model": model_name,
        "seed": int(seed),
        "num_points": int(num_points),
        "K": int(lb_this[targets[0]]["K"]),
        "targets": [float(t) for t in targets],
        "matching": [int(lb_this[t]["matching"]) for t in targets],
        "info_theoretic": [int(lb_this[t]["info_theoretic"]) for t in targets],
        "lp": [int(lb_this[t]["lp"]) for t in targets],
        "best": [int(lb_this[t]["best"]) for t in targets],
    }
    save_run_result(record, _lb_cache_path(raw_dir, model_name, seed))


def _load_lb(raw_dir: str, model_name: str, seed: int, expected_targets: list) -> dict | None:
    """Returns None on miss or target mismatch."""
    path = _lb_cache_path(raw_dir, model_name, seed)
    if not os.path.exists(path + ".json"):
        return None
    r = load_run_result(path)
    cached = [float(t) for t in r["targets"]]
    if sorted(cached) != sorted(float(t) for t in expected_targets):
        return None
    # Invalidate old caches that lack the LP bound.
    if "lp" not in r:
        return None
    K = int(r["K"])
    return {
        float(t): {
            "matching": int(r["matching"][i]),
            "info_theoretic": int(r["info_theoretic"][i]),
            "lp": int(r["lp"][i]),
            "K": K,
            "best": int(r["best"][i]),
        }
        for i, t in enumerate(cached)
    }


def _accumulate_lb(lower_bounds: dict, model_name: str, lb_this: dict) -> None:
    for target, d in lb_this.items():
        key = (model_name, target)
        acc = lower_bounds.setdefault(
            key,
            {
                "best": [],
                "matching": [],
                "info_theoretic": [],
                "lp": [],
                "K": [],
            },
        )
        for k in ("best", "matching", "info_theoretic", "lp", "K"):
            acc[k].append(d[k])


def compute_lower_bounds(V_np: np.ndarray, num_points: int, coverage_targets: list) -> dict:
    """Three bounds per target:
    - Matching (anti-chain): ``max(0, K + ceil(M*t) - M)``
    - Info-theoretic: ``ceil(M*t / max_single_cov)``
    - LP relaxation: ``ceil(LP_opt)`` of the partial set cover LP (tightest; subsumes the others).
    """
    from scipy.sparse import csc_matrix

    V_bool = V_np.astype(np.bool_)
    M = int(V_bool.shape[1])
    max_single_cov = int(V_bool.sum(axis=1).max())
    K = matching_lb_set_cover(V_bool)

    V_sparse = csc_matrix(V_bool)

    bounds = {}
    for target in coverage_targets:
        n_required = int(np.ceil(M * target))
        match_lb = max(0, K + n_required - M)
        info_lb = information_theoretic_lb(num_points, target, max_single_cov)
        lp_lb = set_cover_lp_lb(V_sparse, n_required)
        best = max(match_lb, info_lb, lp_lb)
        bounds[target] = {
            "matching": match_lb,
            "info_theoretic": info_lb,
            "lp": lp_lb,
            "K": K,
            "best": best,
        }
        logger.info(
            "  LB at %.0f%%: Matching=%d (K=%d), Info=%d, LP=%d, Best=%d",
            target * 100,
            match_lb,
            K,
            info_lb,
            lp_lb,
            best,
        )
    return bounds


def _short(optimizer_name: str) -> str:
    return (
        optimizer_name.replace("SetCoverCuda", "Cuda")
        .replace("SetCover", "")
        .replace("ExpansionIterative_", "Exp_")
    )


def _display_model(name: str) -> str:
    return "duke" if name == "duke_of_lancaster" else name


def generate_plots_A(results: list[dict], lower_bounds: dict, optimizers: list[str], fig_dir: str):
    mr = [r for r in results if r.get("section") == "A"]
    if not mr:
        return

    models = sorted(set(r["model"] for r in mr))
    targets = sorted(set(r["target_coverage"] for r in mr))

    def _mean(opt, target, metric, model=None):
        rows = [r for r in mr if r["optimizer"] == opt and r["target_coverage"] == target]
        if model:
            rows = [r for r in rows if r["model"] == model]
        vals = [r[metric] for r in rows]
        return float(np.mean(vals)) if vals else float("nan")

    duke = "duke_of_lancaster"
    if duke in models and lower_bounds:
        duke_mr = [r for r in mr if r["model"] == duke]
        present_opts = [o for o in optimizers if any(r["optimizer"] == o for r in duke_mr)]

        def _lb_mean(t):
            seeds = lower_bounds.get((duke, t), {}).get("best", [])
            return float(np.mean(seeds)) if seeds else 0.0

        valid_targets = [t for t in targets if _lb_mean(t) > 0]
        if valid_targets and present_opts:
            from matplotlib.lines import Line2D

            fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.5))
            vp_data = {
                _short(o): [_mean(o, t, "num_viewpoints", duke) for t in valid_targets]
                for o in present_opts
            }
            grouped_bar(
                ax,
                vp_data,
                [f"{t * 100:.0f}%" for t in valid_targets],
                ylabel="Selected viewpoints",
                title=f"Selected Viewpoints vs. Lower Bound ({_display_model(duke)})",
                value_labels=False,
            )
            for i, t in enumerate(valid_targets):
                ax.hlines(
                    _lb_mean(t),
                    xmin=i - 0.4,
                    xmax=i + 0.4,
                    colors="red",
                    linestyles="--",
                    linewidth=1.2,
                )
            handles, labels = ax.get_legend_handles_labels()
            handles.append(
                Line2D([0], [0], color="red", linestyle="--", linewidth=1.2, label="LP LB")
            )
            labels.append("LP LB")
            ax.legend(handles, labels)
            ax.set_xlabel("Target coverage")
            save_figure(fig, os.path.join(fig_dir, f"{duke}_e06_A_viewpoints_vs_lb"))

    if models:
        target_95 = min(targets, key=lambda t: abs(t - 0.95))
        present_all = [o for o in optimizers if any(r["optimizer"] == o for r in mr)]
        model_labels = [_display_model(m) for m in models]

        from matplotlib.lines import Line2D

        fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4))
        vp_data = {
            _short(o): [_mean(o, target_95, "num_viewpoints", md) for md in models]
            for o in present_all
        }
        grouped_bar(
            ax,
            vp_data,
            model_labels,
            ylabel="Selected viewpoints",
            title=f"Viewpoints at {target_95 * 100:.0f}% — per model",
        )
        for i, md in enumerate(models):
            lb_seeds = lower_bounds.get((md, target_95), {}).get("best", [])
            if lb_seeds:
                ax.hlines(
                    float(np.mean(lb_seeds)),
                    xmin=i - 0.4,
                    xmax=i + 0.4,
                    colors="red",
                    linestyles="--",
                    linewidth=1.2,
                )
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([0], [0], color="red", linestyle="--", linewidth=1.2, label="LP LB"))
        labels.append("LP LB")
        ax.legend(handles, labels)
        ax.set_xlabel("Model")
        ax.tick_params(axis="x", rotation=20)
        save_figure(fig, os.path.join(fig_dir, "cross_model_e06_A_viewpoints_lb"))

        # Log y so fast solvers don't get squashed by Exp_cmaes (~100× slower).
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4))
        time_data = {
            _short(o): [_mean(o, target_95, "optimization_time", md) for md in models]
            for o in present_all
        }
        grouped_bar(
            ax,
            time_data,
            model_labels,
            ylabel="Optimization time (s, log scale)",
            title=f"Optimizer Timing at {target_95 * 100:.0f}% — per model",
        )
        ax.set_yscale("log")
        ax.set_xlabel("Model")
        ax.tick_params(axis="x", rotation=20)
        save_figure(fig, os.path.join(fig_dir, "cross_model_e06_A_timing"))
        logger.info("Cross-model A figures saved (target=%.2f)", target_95)

    logger.info("Section A figures saved to %s", fig_dir)


def main():
    p = argparse.ArgumentParser(description="E4: Set Cover Optimizer Comparison")
    p.add_argument("--section", choices=["A", "B", "both"], default="both")
    p.add_argument("--models", nargs="+", default=["duke_of_lancaster"] + TOSCA_REPRESENTATIVE)
    p.add_argument("--optimizers_A", nargs="+", default=E04_OPTIMIZERS_A)
    p.add_argument("--optimizers_B", nargs="+", default=E04_OPTIMIZERS_B)
    p.add_argument("--strategies_B", nargs="+", default=E04_INPUT_STRATEGIES)
    p.add_argument("--targets", type=float, nargs="+", default=E04_COVERAGE_TARGETS)
    p.add_argument("--seeds_A", type=int, nargs="+", default=SEEDS_5)
    p.add_argument("--seeds_B", type=int, nargs="+", default=SEEDS_3)
    p.add_argument("--output_dir", default=os.path.join(RESULTS_DIR, "e06_set_cover_optimizers"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--plots_only", action="store_true")
    p.add_argument(
        "--lb",
        dest="lb",
        action="store_true",
        default=True,
        help="Compute lower bounds (default: on).",
    )
    p.add_argument("--no_lb", dest="lb", action="store_false")
    p.add_argument(
        "--lb_only",
        action="store_true",
        help="Only compute lower bounds; skip set cover solver runs.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    if args.lb_only and not args.lb:
        p.error("--lb_only is incompatible with --no_lb")

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
    lower_bounds: dict = {}

    run_A = args.section in ("A", "both")

    if args.lb_only:
        # LB is computed inside Section A's candidate-gen loop — force it on.
        run_A = True

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

            if run_A:
                logger.info("--- Section A (weighted_curvature) ---")
                for seed in args.seeds_A:
                    # Try LB cache first so --lb_only --resume can skip candidate generation.
                    lb_this = None
                    if args.lb and args.resume:
                        lb_this = _load_lb(raw_dir, model_name, seed, args.targets)
                    if args.lb_only and lb_this is not None:
                        logger.info("  seed=%d: LB loaded from cache", seed)
                        _accumulate_lb(lower_bounds, model_name, lb_this)
                        continue

                    logger.info("  seed=%d: generating candidates (weighted_curvature)", seed)
                    set_seed(seed)
                    target_points, normals = ctx.sample_surface()
                    vis_query = ctx.build_visibility_query("raycast")

                    with timed() as t_cand:
                        pos_gpu, rot_gpu, n_base, n_iter, _, _ = sample_strategy(
                            ctx,
                            "weighted_curvature",
                            cfg.num_candidates,
                            target_points,
                            normals,
                            vis_query,
                            cfg,
                        )
                    V_gpu, _ = vis_query.compute_visibility_batch(pos_gpu, rot_gpu)
                    V_np = cp.asnumpy(V_gpu)
                    num_points = int(len(target_points))
                    logger.info("  %d candidates (%.1fs)", len(pos_gpu), t_cand.elapsed)

                    if args.lb:
                        if lb_this is None:
                            logger.info("  Computing lower bounds...")
                            with timed() as t_lb:
                                lb_this = compute_lower_bounds(V_np, num_points, args.targets)
                            logger.info("  Lower bounds computed in %.2fs", t_lb.elapsed)
                            _save_lb(raw_dir, model_name, seed, num_points, lb_this)
                        else:
                            logger.info("  Lower bounds loaded from cache")
                        _accumulate_lb(lower_bounds, model_name, lb_this)

                    if args.lb_only:
                        continue

                    for opt_name in args.optimizers_A:
                        for target in args.targets:
                            rpath = os.path.join(
                                raw_dir,
                                f"A_model={model_name}_opt={opt_name}_target={target}_seed={seed}",
                            )
                            if args.resume and os.path.exists(rpath + ".json"):
                                r = load_run_result(rpath)
                                r.setdefault("section", "A")
                                all_results.append(r)
                                continue

                            logger.info("  [A] %s target=%.2f", opt_name, target)
                            try:
                                viz_path = None
                                if (
                                    model_name == "duke_of_lancaster"
                                    and seed == SEEDS_3[0]
                                    and abs(target - 0.95) < 1e-6
                                ):
                                    viz_dir = os.path.join(args.output_dir, "viz")
                                    os.makedirs(viz_dir, exist_ok=True)
                                    viz_path = os.path.join(
                                        viz_dir, f"opt={opt_name}_target={target}_seed={seed}"
                                    )
                                result = run_single(
                                    ctx,
                                    opt_name,
                                    "weighted_curvature",
                                    target,
                                    seed,
                                    pos_gpu,
                                    rot_gpu,
                                    V_gpu,
                                    V_np,
                                    num_points,
                                    viz_path=viz_path,
                                    target_points_viz=target_points,
                                    normals_viz=normals,
                                )
                                result["section"] = "A"
                                all_results.append(result)
                                save_run_result(result, rpath)
                                logger.info(
                                    "    VPs=%d cov=%.2f%% t=%.2fs",
                                    result["num_viewpoints"],
                                    result["coverage"] * 100,
                                    result["optimization_time"],
                                )
                            except Exception as e:
                                handle_row_exception(
                                    e,
                                    f"A model={model_name} opt={opt_name} "
                                    f"target={target} seed={seed}",
                                    resume=args.resume,
                                )
                            finally:
                                free_gpu_memory()

    else:
        for fname in sorted(os.listdir(raw_dir)):
            if not fname.endswith(".json"):
                continue
            base = fname.replace(".json", "")
            r = load_run_result(os.path.join(raw_dir, base))
            if fname.startswith("lb_"):
                model = r["model"]
                K = int(r["K"])
                has_lp = "lp" in r
                for i, t in enumerate(r["targets"]):
                    key = (model, float(t))
                    acc = lower_bounds.setdefault(
                        key,
                        {
                            "best": [],
                            "matching": [],
                            "info_theoretic": [],
                            "lp": [],
                            "K": [],
                        },
                    )
                    acc["best"].append(int(r["best"][i]))
                    acc["matching"].append(int(r["matching"][i]))
                    acc["info_theoretic"].append(int(r["info_theoretic"][i]))
                    acc["lp"].append(int(r["lp"][i]) if has_lp else 0)
                    acc["K"].append(K)
            else:
                if "section" not in r:
                    r["section"] = "A" if fname.startswith("A_") else "B"
                all_results.append(r)

    if all_results:
        setup_thesis_style()
        fig_dir = os.path.join(args.output_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)

        if run_A:
            generate_plots_A(all_results, lower_bounds, args.optimizers_A, fig_dir)


if __name__ == "__main__":
    main()
