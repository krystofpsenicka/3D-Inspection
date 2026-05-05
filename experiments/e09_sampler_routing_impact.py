#!/usr/bin/env python3
"""E09: Sampler Routing Impact - does the sampler choice affect downstream VRP makespan/total_cost?

Strategies: weighted, weighted_curvature, targeted (spi=100, k=3), cmaes (popsize=40,
maxiter=40, k=3). CMA-ES travel_weight sweep over E03_C_TRAVEL_WEIGHTS_DUKE.
Model: Duke. Fixed: 5 robots, 1500 candidates, 0.95 target coverage, 3 seeds.

    python -m experiments.e09_sampler_routing_impact
    python -m experiments.e09_sampler_routing_impact --plots_only
    python -m experiments.e09_sampler_routing_impact --resume
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
    E03_C_TRAVEL_WEIGHTS_DUKE,
    E09_N_CANDIDATES,
    E09_N_ROBOTS,
    RESULTS_DIR,
    SEEDS_3,
    ModelConfig,
)
from experiments.common.lb_sidecar import (
    compute_all_lbs,
    load_raw_lb_dir,
    recompute_lbs,
    save_lb_json,
)

_RUNTIME_IMPORT_ERROR: ImportError | None = None
try:
    import cupy as cp

    from experiments.common.pipeline_setup import PipelineContext
    from experiments.common.runner import (
        free_gpu_memory,
        handle_row_exception,
        set_seed,
        timed,
    )
    from experiments.common.sampling_dispatch import sample_strategy
    from visibility.set_cover import LazyGreedySetCover
    from VRP.core.distance_matrix import compute_distance_matrix
    from VRP.core.geometry import compute_start_grid
    from VRP.core.types import VRPBackend, VRPResult
    from VRP.vrp._helpers import per_vehicle_costs
    from VRP.vrp.vrp_solver import solve_vrp

    _RUNTIME_AVAILABLE = True
except ImportError as _e:
    cp = None
    set_seed = timed = free_gpu_memory = handle_row_exception = None
    PipelineContext = None
    sample_strategy = None
    LazyGreedySetCover = None
    compute_distance_matrix = None
    solve_vrp = None
    VRPBackend = VRPResult = None
    compute_start_grid = None
    per_vehicle_costs = None
    _RUNTIME_AVAILABLE = False
    _RUNTIME_IMPORT_ERROR = _e

_E09_STRATEGIES = ["weighted", "weighted_curvature", "targeted", "cmaes"]
_E09_MODELS = ["duke_of_lancaster"]


def _strategy_kwargs_e09(strategy: str, travel_weight: float | None) -> dict:
    if strategy == "targeted":
        return {"k_coverage": 3, "samples_per_iteration": 100}
    if strategy == "cmaes":
        return {
            "k_coverage": 3,
            "popsize": 40,
            "maxiter": 40,
            "travel_weight": travel_weight if travel_weight is not None else 0.0,
        }
    return {}


from experiments.common.persistence import load_run_result, save_run_result
from experiments.common.plotting import (
    CATEGORICAL_COLORS,
    DOUBLE_COL,
    save_figure,
    setup_thesis_style,
)

# e09 uses default alpha=1.0 (pure makespan); centralised so the LB sidecar uses the same value.
_E09_ALPHA = 1.0

logger = logging.getLogger(__name__)


def _display_label(strategy: str, travel_weight) -> str:
    if travel_weight is not None:
        return f"{strategy}\ntw={travel_weight}"
    return strategy.replace("weighted_curvature", "w_curv")


def _build_run_configs(strategies, cmaes_travel_weights):
    """Expand strategies into (strategy, travel_weight) tuples; one TW entry per cmaes."""
    configs = []
    for s in strategies:
        if s == "cmaes":
            for tw in cmaes_travel_weights:
                configs.append((s, tw))
        else:
            configs.append((s, None))
    return configs


def _result_path(raw_dir, model_name, strategy, travel_weight, seed):
    tw_str = f"tw={travel_weight}" if travel_weight is not None else "tw=none"
    return os.path.join(
        raw_dir,
        f"model={model_name}_strategy={strategy}_{tw_str}_seed={seed}",
    )


def _rebuild_instance_for_lb(
    ctx: PipelineContext,
    strategy: str,
    travel_weight,
    seed: int,
    target_coverage: float,
) -> tuple:
    """Mirror stages 1-6 of ``run_single`` without VRP. Returns ``(K, home_indices, dist_matrix,
    num_viewpoints)``. Logic deliberately duplicated with ``run_single`` - changing run_single
    risks behaviour drift for the main experiment."""
    set_seed(seed)
    model_cfg = ctx.model
    target_points, normals = ctx.sample_surface()
    vis_query = ctx.build_visibility_query("raycast")

    pos_gpu, rot_gpu, *_ = sample_strategy(
        ctx,
        strategy,
        E09_N_CANDIDATES,
        target_points,
        normals,
        vis_query,
        model_cfg,
        **_strategy_kwargs_e09(strategy, travel_weight),
    )
    V, _ = vis_query.compute_visibility_batch(pos_gpu, rot_gpu)
    V_np = cp.asnumpy(V)
    pos_np = cp.asnumpy(pos_gpu)
    rot_np = cp.asnumpy(rot_gpu)
    optimizer = LazyGreedySetCover(len(target_points), pos_np, rot_np, V_np)
    opt_result = optimizer.optimize(
        target_coverage=target_coverage,
        max_viewpoints=1000,
    )

    K = E09_N_ROBOTS
    _, o3d_mesh = ctx.load_mesh()
    bmin, bmax = ctx.mesh_bounds
    robot_xyzs = compute_start_grid(K, bmin, bmax)
    home_pos = np.array(
        [[float(x[0]), float(x[1]), float(x[2])] for x in robot_xyzs],
        dtype=np.float32,
    )
    vp_pos_np = opt_result.positions.get()
    all_pos = np.vstack([home_pos, vp_pos_np])
    home_indices = list(range(K))

    import trimesh

    from shared.grid_builder_utils import compute_grid_bounds, voxelize_mesh
    from shared.grid_utils import inflate_grid
    from shared.occupancy_grid import OccupancyGrid

    _verts = np.asarray(o3d_mesh.vertices)
    _tm = trimesh.Trimesh(vertices=_verts, faces=np.asarray(o3d_mesh.triangles))
    _padding = model_cfg.frustum.far + 2 * model_cfg.collision_radius
    _vrp_origin, _vrp_shape = compute_grid_bounds(
        _verts.min(axis=0),
        _verts.max(axis=0),
        _padding,
        0.20,
    )
    _raw_vrp, _ = voxelize_mesh(_tm, _vrp_shape, _vrp_origin, 0.20, fill_interior=False)
    og_vrp = OccupancyGrid(
        grid=inflate_grid(_raw_vrp, 1),
        origin=_vrp_origin,
        resolution=0.20,
    )
    dist_matrix = compute_distance_matrix(og_vrp, cp.asarray(all_pos))
    return K, home_indices, dist_matrix, opt_result.num_viewpoints


def run_single(
    ctx: PipelineContext, strategy: str, travel_weight, seed: int, target_coverage: float = 0.95
) -> dict:
    set_seed(seed)
    model_cfg = ctx.model
    target_points, normals = ctx.sample_surface()
    vis_query = ctx.build_visibility_query("raycast")

    with timed() as t_sample:
        pos_gpu, rot_gpu, n_base, n_iter, base_name, n_ws_fb = sample_strategy(
            ctx,
            strategy,
            E09_N_CANDIDATES,
            target_points,
            normals,
            vis_query,
            model_cfg,
            **_strategy_kwargs_e09(strategy, travel_weight),
        )

    with timed() as t_vis:
        V, _ = vis_query.compute_visibility_batch(pos_gpu, rot_gpu)

    with timed() as t_opt:
        V_np = cp.asnumpy(V)
        pos_np = cp.asnumpy(pos_gpu)
        rot_np = cp.asnumpy(rot_gpu)
        optimizer = LazyGreedySetCover(len(target_points), pos_np, rot_np, V_np)
        opt_result = optimizer.optimize(
            target_coverage=target_coverage,
            max_viewpoints=1000,
        )

    num_viewpoints = opt_result.num_viewpoints
    coverage = float(opt_result.total_coverage)
    redundancy = float(opt_result.redundancy)

    with timed() as t_vrp:
        K = E09_N_ROBOTS
        _, o3d_mesh = ctx.load_mesh()
        bmin, bmax = ctx.mesh_bounds

        robot_xyzs = compute_start_grid(K, bmin, bmax)
        home_pos = np.array(
            [[float(x[0]), float(x[1]), float(x[2])] for x in robot_xyzs],
            dtype=np.float32,
        )
        np.tile(np.eye(3, dtype=np.float32), (K, 1, 1))

        vp_pos_np = opt_result.positions.get()
        opt_result.rotations.get()
        all_pos = np.vstack([home_pos, vp_pos_np])
        home_indices = list(range(K))

        # og_vrp must accept every viewpoint the sampling OG accepts. At 0.20m resolution
        # (kept coarse because cuGraph OOMs on the 0.10m sampling grid), worst-case voxel
        # alignment mismatch is ~0.67m, so any inflation > 1 voxel can reject a viewpoint
        # the sampling OG considers feasible. Inflate by 1 voxel only (0.20m clearance) -
        # OK for VRP distance estimation (no MAPF in e09; paths only pair waypoints for
        # route ordering). Match sampling OG's padding so grid origins align.
        import trimesh

        from shared.grid_builder_utils import compute_grid_bounds, voxelize_mesh
        from shared.grid_utils import inflate_grid
        from shared.occupancy_grid import OccupancyGrid

        _verts = np.asarray(o3d_mesh.vertices)
        _tm = trimesh.Trimesh(vertices=_verts, faces=np.asarray(o3d_mesh.triangles))
        _padding = model_cfg.frustum.far + 2 * model_cfg.collision_radius
        _vrp_origin, _vrp_shape = compute_grid_bounds(
            _verts.min(axis=0),
            _verts.max(axis=0),
            _padding,
            0.20,
        )
        _raw_vrp, _ = voxelize_mesh(_tm, _vrp_shape, _vrp_origin, 0.20, fill_interior=False)
        og_vrp = OccupancyGrid(
            grid=inflate_grid(_raw_vrp, 1),
            origin=_vrp_origin,
            resolution=0.20,
        )
        dist_matrix = compute_distance_matrix(og_vrp, cp.asarray(all_pos))
        vrp_result: VRPResult = solve_vrp(
            dist_matrix=dist_matrix,
            num_vehicles=K,
            depots=home_indices,
            alpha=_E09_ALPHA,
            backend=VRPBackend.CUOPT,
            time_limit=120,
        )

    pv = per_vehicle_costs(vrp_result.routes, dist_matrix, home_indices)
    makespan = max(pv) if pv else 0.0
    total_cost = vrp_result.total_cost

    # VRP-only LBs (no MAPF in this experiment).
    lb: dict | None = None
    try:
        lb = compute_all_lbs(
            dist_matrix,
            home_indices,
            K,
            num_viewpoints,
            _E09_ALPHA,
            include_mapf=False,
            vrp_best_bound_m=vrp_result.best_bound,
            vrp_objective_value_m=vrp_result.objective_value,
        )
    except Exception as e:
        logger.warning("LB computation failed: %s", e)

    result = {
        "model": model_cfg.name,
        "strategy": strategy,
        "travel_weight": travel_weight,
        "seed": seed,
        "n_base_candidates": n_base,
        "n_iterative_candidates": n_iter,
        "n_warmstart_fallbacks": int(n_ws_fb),
        "num_candidates": int(len(pos_gpu)),
        "num_viewpoints": num_viewpoints,
        "coverage": coverage,
        "redundancy": redundancy,
        "makespan": makespan,
        "total_cost": total_cost,
        "vrp_status": vrp_result.status,
        "t_sample": t_sample.elapsed,
        "t_vis": t_vis.elapsed,
        "t_opt": t_opt.elapsed,
        "t_vrp": t_vrp.elapsed,
        "t_total": t_sample.elapsed + t_vis.elapsed + t_opt.elapsed + t_vrp.elapsed,
    }
    return result, lb


def generate_plots(
    results: list[dict],
    output_dir: str,
    lb_by_stem: dict | None = None,
    target_coverage: float = 0.95,
):
    setup_thesis_style()
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    lb_by_stem = lb_by_stem or {}

    all_ok = [r for r in results if r.get("vrp_status") == "success"]
    if not all_ok:
        logger.warning("No successful runs for plotting")
        return

    models = sorted(set(r.get("model", "unknown") for r in all_ok))
    for model_name in models:
        ok = [r for r in all_ok if r.get("model") == model_name]
        if not ok:
            continue
        _generate_plots_for_model(ok, fig_dir, model_name, lb_by_stem, target_coverage)


def _generate_plots_for_model(
    ok: list[dict], fig_dir: str, model_name: str, lb_by_stem: dict, target_coverage: float
):
    seen = set()
    run_cfgs = []
    for r in ok:
        key = (r["strategy"], r["travel_weight"])
        if key not in seen:
            run_cfgs.append(key)
            seen.add(key)
    non_cmaes = sorted([c for c in run_cfgs if c[1] is None], key=lambda x: x[0])
    cmaes = sorted([c for c in run_cfgs if c[1] is not None], key=lambda x: x[1])
    run_cfgs = non_cmaes + cmaes

    # Each (strategy, travel_weight) is a distinct "method" with its own discrete color.
    # Runs whose coverage fell below the target are drawn with a star marker.
    method_color = {
        cfg: CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)] for i, cfg in enumerate(run_cfgs)
    }

    def _scatter(metric_y: str, ylabel: str, title: str, fname: str):
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4))
        for cfg in run_cfgs:
            s, tw = cfg
            rows = [r for r in ok if r["strategy"] == s and r["travel_weight"] == tw]
            if not rows:
                continue
            color = method_color[cfg]
            label = _display_label(s, tw)
            ok_rows = [r for r in rows if r["coverage"] >= target_coverage]
            bad_rows = [r for r in rows if r["coverage"] < target_coverage]
            if ok_rows:
                ax.scatter(
                    [r["num_viewpoints"] for r in ok_rows],
                    [r[metric_y] for r in ok_rows],
                    color=color,
                    marker="o",
                    s=40,
                    alpha=0.85,
                    label=label,
                )
            if bad_rows:
                ax.scatter(
                    [r["num_viewpoints"] for r in bad_rows],
                    [r[metric_y] for r in bad_rows],
                    color=color,
                    marker="*",
                    s=110,
                    alpha=0.85,
                    edgecolor="0.25",
                    linewidth=0.6,
                    label=None if ok_rows else label,
                )
        if any(r["coverage"] < target_coverage for r in ok):
            from matplotlib.lines import Line2D

            handles, labels = ax.get_legend_handles_labels()
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="*",
                    color="white",
                    markeredgecolor="0.25",
                    markerfacecolor="0.6",
                    markersize=10,
                    label=f"Coverage < {target_coverage * 100:.0f}%",
                )
            )
            labels.append(f"Coverage < {target_coverage * 100:.0f}%")
            ax.legend(handles=handles, labels=labels, fontsize=6, ncol=2, loc="best")
        else:
            ax.legend(fontsize=6, ncol=2, loc="best")
        ax.set_xlabel("Selected viewpoints")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.tight_layout()
        save_figure(fig, os.path.join(fig_dir, fname))

    _scatter(
        "makespan",
        "Makespan (m)",
        "Viewpoints vs Makespan",
        f"{model_name}_e09_scatter_vp_vs_makespan",
    )
    _scatter(
        "total_cost",
        "Total route cost (m)",
        "Viewpoints vs Total Cost",
        f"{model_name}_e09_scatter_vp_vs_total_cost",
    )

    logger.info("E09 figures saved to %s", fig_dir)


def main():
    p = argparse.ArgumentParser(description="E09: Sampler Routing Impact")
    p.add_argument(
        "--models",
        nargs="+",
        default=_E09_MODELS,
    )
    p.add_argument("--strategies", nargs="+", default=_E09_STRATEGIES)
    p.add_argument(
        "--cmaes_travel_weights",
        type=float,
        nargs="+",
        default=None,
        help="Override the CMA-ES travel-weight sweep; defaults to E03_C_TRAVEL_WEIGHTS_DUKE.",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_3)
    p.add_argument("--target_coverage", type=float, default=0.95)
    p.add_argument("--output_dir", default=os.path.join(RESULTS_DIR, "e09_sampler_routing_impact"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--plots_only", action="store_true")
    p.add_argument(
        "--compute_lbs_only",
        action="store_true",
        help="Skip VRP solves; rebuild each existing instance, compute analytical LBs, write "
        "sidecar JSONs into raw_lb/.",
    )
    p.add_argument(
        "--include_cuopt_bound",
        action="store_true",
        help="In --compute_lbs_only mode, also run a short cuOpt solve per instance to extract "
        "the MIP dual bound. Expensive; off by default.",
    )
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
    raw_lb_dir = os.path.join(args.output_dir, "raw_lb")
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

    def _lb_path_for(result_path: str) -> str:
        stem = os.path.basename(result_path)
        return os.path.join(raw_lb_dir, stem)

    if args.compute_lbs_only:
        if not os.path.isdir(raw_dir):
            logger.error("No raw/ directory at %s; nothing to augment.", raw_dir)
            return
        os.makedirs(raw_lb_dir, exist_ok=True)
        # Group existing main JSONs by model so we only build one PipelineContext per model.
        stems = sorted(f[:-5] for f in os.listdir(raw_dir) if f.endswith(".json"))
        by_model: dict[str, list[dict]] = {}
        for stem in stems:
            main = load_run_result(os.path.join(raw_dir, stem))
            all_results.append(main)
            by_model.setdefault(main["model"], []).append({**main, "_stem": stem})

        logger.info(
            "Recomputing LBs for %d rows across %d models%s ...",
            len(stems),
            len(by_model),
            " including cuOpt bound" if args.include_cuopt_bound else "",
        )

        for model_name, rows in by_model.items():
            cfg = _make_cfg(model_name)
            if cfg is None:
                logger.warning("Skipping %d rows for missing model %s", len(rows), model_name)
                continue
            ctx = PipelineContext(cfg)
            ctx.load_mesh()
            ctx.sample_surface(seed=42)
            ctx.build_sampling_og()

            logger.info("[%s] %d rows to recompute", model_name, len(rows))
            for idx, row in enumerate(rows, 1):
                stem = row["_stem"]
                try:
                    K, home_indices, dist_matrix, n_vp = _rebuild_instance_for_lb(
                        ctx,
                        row["strategy"],
                        row["travel_weight"],
                        int(row["seed"]),
                        args.target_coverage,
                    )
                    lb = recompute_lbs(
                        dist_matrix,
                        home_indices,
                        K,
                        n_vp,
                        _E09_ALPHA,
                        include_mapf=False,
                        include_cuopt=args.include_cuopt_bound,
                    )
                    save_lb_json(os.path.join(raw_lb_dir, stem), lb)
                except Exception as e:
                    logger.error("LB recompute failed for %s: %s", stem, e)
                finally:
                    free_gpu_memory()
                if idx % 5 == 0:
                    logger.info("  [%s] ... %d/%d done", model_name, idx, len(rows))

        lb_by_stem = load_raw_lb_dir(raw_lb_dir, raw_dir)
        if all_results:
            generate_plots(
                all_results,
                args.output_dir,
                lb_by_stem=lb_by_stem,
                target_coverage=args.target_coverage,
            )
        return

    if not args.plots_only:
        os.makedirs(raw_lb_dir, exist_ok=True)
        for model_name in args.models:
            cfg = _make_cfg(model_name)
            if cfg is None:
                continue
            tws = (
                list(args.cmaes_travel_weights)
                if args.cmaes_travel_weights is not None
                else list(E03_C_TRAVEL_WEIGHTS_DUKE)
            )
            run_cfgs = _build_run_configs(args.strategies, tws)
            combos = [(s, tw, seed) for s, tw in run_cfgs for seed in args.seeds]
            total = len(combos)

            logger.info("=" * 60)
            logger.info(
                "E09 - Model: %s - %d runs (%d configs x %d seeds)",
                model_name,
                total,
                len(run_cfgs),
                len(args.seeds),
            )

            ctx = PipelineContext(cfg)
            ctx.load_mesh()
            ctx.sample_surface(seed=42)
            ctx.build_sampling_og()

            for idx, (strategy, tw, seed) in enumerate(combos, 1):
                rpath = _result_path(raw_dir, model_name, strategy, tw, seed)

                if args.resume and os.path.exists(rpath + ".json"):
                    logger.info(
                        "[%s %d/%d] SKIP %s tw=%s seed=%d",
                        model_name,
                        idx,
                        total,
                        strategy,
                        tw,
                        seed,
                    )
                    all_results.append(load_run_result(rpath))
                    continue

                logger.info(
                    "[%s %d/%d] strategy=%s travel_weight=%s seed=%d",
                    model_name,
                    idx,
                    total,
                    strategy,
                    tw,
                    seed,
                )
                try:
                    result, lb = run_single(ctx, strategy, tw, seed, args.target_coverage)
                    all_results.append(result)
                    save_run_result(result, rpath)
                    if lb is not None:
                        save_lb_json(_lb_path_for(rpath), lb)
                    logger.info(
                        "  vps=%d cov=%.2f%% makespan=%.1f total_cost=%.1f t=%.1fs",
                        result["num_viewpoints"],
                        result["coverage"] * 100,
                        result["makespan"],
                        result["total_cost"],
                        result["t_total"],
                    )
                except Exception as e:
                    handle_row_exception(
                        e,
                        f"model={model_name} strategy={strategy} tw={tw} seed={seed}",
                        resume=args.resume,
                    )
                finally:
                    free_gpu_memory()
    else:
        for fname in sorted(os.listdir(raw_dir)):
            if fname.endswith(".json"):
                all_results.append(
                    load_run_result(os.path.join(raw_dir, fname.replace(".json", "")))
                )

    lb_by_stem = load_raw_lb_dir(raw_lb_dir, raw_dir)
    if all_results:
        generate_plots(
            all_results,
            args.output_dir,
            lb_by_stem=lb_by_stem,
            target_coverage=args.target_coverage,
        )

        models = sorted(set(r.get("model", "unknown") for r in all_results))
        for model_name in models:
            logger.info("\n%s\nE09 SUMMARY - %s\n%s", "=" * 80, model_name, "=" * 80)
            logger.info(
                "%-22s %6s %8s %10s %10s %10s %8s",
                "Strategy",
                "TW",
                "VPs",
                "Coverage%",
                "Makespan",
                "TotalCost",
                "Time(s)",
            )
            logger.info("-" * 80)
            tws = (
                list(args.cmaes_travel_weights)
                if args.cmaes_travel_weights is not None
                else list(E03_C_TRAVEL_WEIGHTS_DUKE)
            )
            run_cfgs = _build_run_configs(args.strategies, tws)
            for s, tw in run_cfgs:
                sr = [
                    r
                    for r in all_results
                    if r.get("model") == model_name
                    and r["strategy"] == s
                    and r["travel_weight"] == tw
                    and r.get("vrp_status") == "success"
                ]
                if sr:
                    logger.info(
                        "%-22s %6s %8.0f %10.2f %10.1f %10.1f %8.1f",
                        s,
                        f"{tw}" if tw is not None else "-",
                        np.mean([r["num_viewpoints"] for r in sr]),
                        np.mean([r["coverage"] * 100 for r in sr]),
                        np.mean([r["makespan"] for r in sr]),
                        np.mean([r["total_cost"] for r in sr]),
                        np.mean([r["t_total"] for r in sr]),
                    )


if __name__ == "__main__":
    main()
