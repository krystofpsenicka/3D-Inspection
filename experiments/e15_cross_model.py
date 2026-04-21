#!/usr/bin/env python3
"""E15: Cross-Model Generalization

Runs the full 8-stage inspection pipeline on Duke of Lancaster + TOSCA_ALL to
evaluate how well the pipeline generalizes across shape complexity.

Pipeline stages timed:
  1. Mesh loading
  2. Surface sampling
  3. Occupancy grid
  4. Viewpoint sampling  (targeted_50 — chosen based on e01 results)
  5. Visibility          (GPU raycast — ground truth, from e03 results)
  6. Set cover           (LazyGreedy CPU — fastest solver, from e04 results)
  7. VRP routing         (HiGHS, K=2 robots — from e08 results)
  8. MAPF trajectory     (resolution 0.5 m — from e10 results)

Implementation choice rationale (printed in summary and figure annotations):
  - Sampler: targeted_50 achieves the best coverage/viewpoints ratio (e01)
  - Visibility: GPU raycast is exact and fastest for N≤5K candidates (e03)
  - Set cover: LazyGreedy (CPU) beats LazyGreedy (GPU) for N~1500 due to heap (e04)
  - VRP: HiGHS LP solver achieves near-optimal within time limit (e08)
  - MAPF: 0.5 m voxel resolution balances path quality vs planning cost (e10)

Usage:
    conda run -n isaaclab python -m experiments.e15_cross_model
    conda run -n isaaclab python -m experiments.e15_cross_model --plots_only
    conda run -n isaaclab python -m experiments.e15_cross_model --models duke_of_lancaster wolf0
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
from experiments.common.runner import set_seed, timed, free_gpu_memory
from experiments.common.pipeline_setup import PipelineContext, DegenerateNormalsError
from experiments.common.persistence import save_run_result, load_run_result
from experiments.common.sampling_dispatch import sample_strategy
from experiments.common.plotting import (
    setup_thesis_style, save_figure, grouped_bar, stacked_bar,
    THESIS_COL, DOUBLE_COL, CATEGORICAL_COLORS,
)
from visibility.set_cover import LazyGreedySetCover

logger = logging.getLogger(__name__)

ALL_MODELS = ["duke_of_lancaster"] + TOSCA_ALL
TARGET_COVERAGE = 0.95
FLEET_SIZE = 2   # robots for VRP/MAPF

_IMPLEMENTATION_CHOICES = (
    "Sampler: targeted_50 (best coverage/viewpoints, e01)  |  "
    "Visibility: GPU raycast (exact, e03)  |  "
    "Set cover: LazyGreedy CPU (O(log N) heap, fastest, e04)  |  "
    "VRP: HiGHS (near-optimal, e08)  |  "
    "MAPF: 0.5 m resolution (e10)"
)

# Graceful import of VRP/MAPF stack
try:
    from VRP.core.distance_matrix import compute_distance_matrix
    from VRP.vrp.vrp_solver import solve_vrp
    from VRP.core.types import VRPBackend, ExecutionResult
    from VRP.mapf.mapf_planner import MultiAgentPathPlanner
    from VRP.core.geometry import compute_start_grid
    _VRP_AVAILABLE = True
except ImportError as _vrp_err:
    logger.warning("VRP/MAPF stack not available (%s) — stages 7/8 will be skipped.", _vrp_err)
    _VRP_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# Single run logic
# ═══════════════════════════════════════════════════════════════════════════

def run_single(ctx: PipelineContext, model_cfg: ModelConfig, seed: int) -> dict:
    """Run full 8-stage pipeline on one model with one seed."""
    result: dict = {
        "model": model_cfg.name,
        "seed": seed,
        # Stage timings
        "t_mesh": 0.0, "t_surface": 0.0, "t_og": 0.0,
        "t_sample": 0.0, "t_vis": 0.0, "t_opt": 0.0,
        "t_vrp": 0.0, "t_mapf": 0.0,
        # Metrics
        "mesh_vertices": 0, "mesh_faces": 0,
        "num_viewpoints": 0, "coverage": 0.0, "redundancy": 0.0,
        "vrp_makespan": float("nan"), "vrp_status": "skipped",
        "mapf_steps": 0, "mapf_collisions": 0,
    }

    # 1. Mesh loading
    with timed() as t_mesh:
        tm, _ = ctx.load_mesh()
    result["t_mesh"] = t_mesh.elapsed
    result["mesh_vertices"] = len(tm.vertices)
    result["mesh_faces"] = len(tm.faces)

    # 2. Surface sampling
    with timed() as t_surface:
        target_points, normals = ctx.sample_surface()
    result["t_surface"] = t_surface.elapsed

    # 3. Occupancy grid
    with timed() as t_og:
        og = ctx.build_sampling_og()
    result["t_og"] = t_og.elapsed

    # 4. Viewpoint sampling (targeted_50)
    set_seed(seed)
    vis_query = ctx.build_visibility_query("raycast")
    with timed() as t_sample:
        pos_gpu, rot_gpu, _, _, _, _ = sample_strategy(
            ctx, "targeted_50", model_cfg.num_candidates,
            target_points, normals, vis_query, model_cfg,
        )
    result["t_sample"] = t_sample.elapsed

    # 5. Visibility (GPU raycast)
    with timed() as t_vis:
        V, _ = vis_query.compute_visibility_batch(pos_gpu, rot_gpu)
    result["t_vis"] = t_vis.elapsed

    # 6. Set cover (LazyGreedy CPU)
    V_np = cp.asnumpy(V)
    pos_np = cp.asnumpy(pos_gpu)
    rot_np = cp.asnumpy(rot_gpu)
    num_points = int(len(target_points))

    with timed() as t_opt:
        optimizer = LazyGreedySetCover(num_points, pos_np, rot_np, V_np)
        opt_result = optimizer.optimize(
            target_coverage=TARGET_COVERAGE, max_viewpoints=1000)
    result["t_opt"] = t_opt.elapsed

    result["num_viewpoints"] = opt_result.num_viewpoints
    result["coverage"] = float(opt_result.total_coverage)
    result["redundancy"] = float(opt_result.redundancy)

    # Selected viewpoint positions / rotations
    sel_idx = opt_result.selected_indices
    if hasattr(sel_idx, "get"):
        sel_idx = sel_idx.get()
    sel_idx = np.asarray(sel_idx)
    insp_positions = pos_np[sel_idx]   # (N_vp, 3)
    insp_rotmats = rot_np[sel_idx]     # (N_vp, 3, 3)

    if not _VRP_AVAILABLE or opt_result.num_viewpoints == 0:
        result["total_time"] = sum(
            result[k] for k in ("t_mesh", "t_surface", "t_og",
                                 "t_sample", "t_vis", "t_opt"))
        return result

    # 7. VRP routing (HiGHS, K=FLEET_SIZE robots)
    try:
        bounds_min, bounds_max = ctx.mesh_bounds
        robot_starts = compute_start_grid(FLEET_SIZE, bounds_min, bounds_max)
        home_positions = np.array(
            [[float(xyz[0]), float(xyz[1]), float(xyz[2])] for xyz in robot_starts],
            dtype=np.float32)
        home_rotmats = np.tile(np.eye(3, dtype=np.float32), (FLEET_SIZE, 1, 1))
        all_positions = np.vstack([home_positions, insp_positions])
        all_rotmats = np.concatenate([home_rotmats, insp_rotmats])
        home_indices = list(range(FLEET_SIZE))

        with timed() as t_vrp:
            dist_matrix = compute_distance_matrix(og, cp.asarray(all_positions))
            vrp_result = solve_vrp(
                dist_matrix=dist_matrix,
                num_vehicles=FLEET_SIZE,
                depots=home_indices,
                backend=VRPBackend.CUOPT,
                time_limit=60,
            )
        result["t_vrp"] = t_vrp.elapsed
        result["vrp_status"] = vrp_result.status

        if any(vrp_result.routes):
            from VRP.vrp._helpers import per_vehicle_costs
            rc = np.array(per_vehicle_costs(
                vrp_result.routes, dist_matrix, home_indices))
            result["vrp_makespan"] = float(rc.max())

            # 8. MAPF trajectory planning
            routes = [
                [home_indices[i]] + list(r) + [home_indices[i]]
                for i, r in enumerate(vrp_result.routes)
            ]
            start_positions = [
                np.array(xyz, dtype=np.float32) for xyz in robot_starts
            ]
            wp_pos_gpu = cp.asarray(all_positions, dtype=cp.float32)
            wp_rot_gpu = cp.asarray(all_rotmats, dtype=cp.float32)

            with timed() as t_mapf:
                executor = MultiAgentPathPlanner(
                    start_positions=start_positions, og=og)
                exec_result = executor.execute(
                    routes=routes,
                    waypoint_positions=wp_pos_gpu,
                    waypoint_rotmats=wp_rot_gpu,
                    home_indices=set(home_indices),
                    dist_matrix=dist_matrix,
                )
            result["t_mapf"] = t_mapf.elapsed

            if exec_result.all_traj_positions:
                result["mapf_steps"] = max(
                    len(t) for t in exec_result.all_traj_positions)
            result["mapf_collisions"] = sum(exec_result.fail_counts)
        else:
            result["vrp_status"] = "empty_routes"

    except Exception as e:
        logger.error("VRP/MAPF stage failed: %s", e, exc_info=True)
        result["vrp_status"] = f"error: {e}"

    result["total_time"] = sum(
        result[k] for k in ("t_mesh", "t_surface", "t_og",
                             "t_sample", "t_vis", "t_opt",
                             "t_vrp", "t_mapf"))
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Plot generation
# ═══════════════════════════════════════════════════════════════════════════

_STAGE_NAMES = ["Mesh", "Surface", "OG", "Sampling", "Visibility",
                "Set Cover", "VRP", "MAPF"]
_STAGE_KEYS = ["t_mesh", "t_surface", "t_og", "t_sample", "t_vis",
               "t_opt", "t_vrp", "t_mapf"]


def generate_plots(results: list[dict], output_dir: str):
    """Generate all E15 figures."""
    setup_thesis_style()
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    if not results:
        logger.warning("No results for plotting")
        return

    models = sorted(
        set(r["model"] for r in results),
        key=lambda m: np.mean([r["mesh_faces"] for r in results if r["model"] == m]),
    )

    def _mm(model, metric):
        vals = [r[metric] for r in results if r["model"] == model
                and not (isinstance(r[metric], float) and np.isnan(r[metric]))]
        return float(np.mean(vals)) if vals else float("nan")

    def _ms(model, metric):
        vals = [r[metric] for r in results if r["model"] == model
                and not (isinstance(r[metric], float) and np.isnan(r[metric]))]
        return float(np.std(vals)) if vals else 0.0

    subtitle = f"\n{_IMPLEMENTATION_CHOICES}"

    # ── Fig 1: Viewpoints by model ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4))
    vp_m = [_mm(m, "num_viewpoints") for m in models]
    vp_s = [_ms(m, "num_viewpoints") for m in models]
    x = np.arange(len(models))
    bars = ax.bar(x, vp_m, 0.6, yerr=vp_s, capsize=3,
                  color=CATEGORICAL_COLORS[0], alpha=0.85)
    ax.bar_label(bars, fmt="%.0f", fontsize=7, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha="right")
    ax.set_ylabel("Selected viewpoints")
    ax.set_title(f"Viewpoints by Model (95% coverage){subtitle}", fontsize=9)
    save_figure(fig, os.path.join(fig_dir, "e15_viewpoints"))

    # ── Fig 2: Coverage by model ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4))
    cov_m = [_mm(m, "coverage") * 100 for m in models]
    cov_s = [_ms(m, "coverage") * 100 for m in models]
    bars = ax.bar(x, cov_m, 0.6, yerr=cov_s, capsize=3,
                  color=CATEGORICAL_COLORS[1], alpha=0.85)
    ax.bar_label(bars, fmt="%.1f%%", fontsize=7, padding=2)
    ax.axhline(95, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha="right")
    ax.set_ylabel("Coverage (%)")
    ax.set_title(f"Coverage by Model{subtitle}", fontsize=9)
    save_figure(fig, os.path.join(fig_dir, "e15_coverage"))

    # ── Fig 3: Full 8-stage timing breakdown (stacked bar) ───────────────
    # Include only stages that have non-zero values (VRP/MAPF may be skipped)
    active_stages = []
    active_keys = []
    for name, key in zip(_STAGE_NAMES, _STAGE_KEYS):
        if any(r.get(key, 0) > 0 for r in results):
            active_stages.append(name)
            active_keys.append(key)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4.5))
    stage_data = {
        sname: [_mm(m, skey) for m in models]
        for sname, skey in zip(active_stages, active_keys)
    }
    stacked_bar(ax, models, stage_data,
                ylabel="Time (s)",
                title=f"Pipeline Stage Timing by Model (K={FLEET_SIZE} robots){subtitle}")
    ax.tick_params(axis="x", rotation=35)
    save_figure(fig, os.path.join(fig_dir, "e15_timing_breakdown"))

    # ── Fig 4: VRP makespan by model (if available) ─────────────────────
    vrp_results = [r for r in results if not np.isnan(r.get("vrp_makespan", float("nan")))]
    if vrp_results:
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4))
        vrp_m = [_mm(m, "vrp_makespan") for m in models]
        vrp_s = [_ms(m, "vrp_makespan") for m in models]
        valid_mask = [not np.isnan(v) for v in vrp_m]
        valid_models = [m for m, v in zip(models, valid_mask) if v]
        valid_m = [v for v, ok in zip(vrp_m, valid_mask) if ok]
        valid_s = [v for v, ok in zip(vrp_s, valid_mask) if ok]
        if valid_models:
            xv = np.arange(len(valid_models))
            ax.bar(xv, valid_m, 0.6, yerr=valid_s, capsize=3,
                   color=CATEGORICAL_COLORS[3], alpha=0.85)
            ax.set_xticks(xv)
            ax.set_xticklabels(valid_models, rotation=35, ha="right")
            ax.set_ylabel("VRP makespan (m)")
            ax.set_title(f"VRP Makespan by Model (K={FLEET_SIZE})")
            save_figure(fig, os.path.join(fig_dir, "e15_vrp_makespan"))

    # ── Fig 5: Viewpoints vs mesh complexity (scatter) ───────────────────
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3.5))
    for r in results:
        ax.scatter(r["mesh_faces"], r["num_viewpoints"],
                   alpha=0.4, color=CATEGORICAL_COLORS[0], s=15)
    for m in models:
        mr = [r for r in results if r["model"] == m]
        ax.annotate(m,
                    (np.mean([r["mesh_faces"] for r in mr]),
                     np.mean([r["num_viewpoints"] for r in mr])),
                    fontsize=6, textcoords="offset points", xytext=(3, 3))
    ax.set_xlabel("Mesh faces")
    ax.set_ylabel("Selected viewpoints")
    ax.set_title("Viewpoints vs Mesh Complexity")
    save_figure(fig, os.path.join(fig_dir, "e15_scatter_faces_vs_vps"))

    logger.info("E15 figures saved to %s", fig_dir)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="E15: Cross-Model Generalization")
    p.add_argument("--models", nargs="+", default=ALL_MODELS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_3)
    p.add_argument("--output_dir",
                   default=os.path.join(RESULTS_DIR, "e15_cross_model"))
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--plots_only", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
    )

    logger.info("Implementation choices: %s", _IMPLEMENTATION_CHOICES)
    if not _VRP_AVAILABLE:
        logger.warning("VRP/MAPF stack not available — stages 7/8 will be skipped")

    raw_dir = os.path.join(args.output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    all_results: list[dict] = []

    if not args.plots_only:
        model_configs = []
        for name in args.models:
            if name == "duke_of_lancaster":
                model_configs.append(ModelConfig.duke_of_lancaster())
            else:
                try:
                    model_configs.append(ModelConfig.tosca(name))
                except FileNotFoundError:
                    logger.warning("Model %s not found — skipping", name)

        total = len(model_configs) * len(args.seeds)
        run_idx = 0

        for model_cfg in model_configs:
            logger.info("=" * 60)
            logger.info("Model: %s", model_cfg.name)
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
                rpath = os.path.join(raw_dir, f"model={model_cfg.name}_seed={seed}")

                if args.skip_existing and os.path.exists(rpath + ".json"):
                    logger.info("[%d/%d] SKIP %s seed=%d",
                                run_idx, total, model_cfg.name, seed)
                    all_results.append(load_run_result(rpath))
                    continue

                logger.info("[%d/%d] Running: %s seed=%d",
                            run_idx, total, model_cfg.name, seed)
                try:
                    result = run_single(ctx, model_cfg, seed)
                    all_results.append(result)
                    save_run_result(result, rpath)
                    logger.info("  vps=%d cov=%.2f%% t=%.1fs vrp=%s",
                                result["num_viewpoints"],
                                result["coverage"] * 100,
                                result["total_time"],
                                result["vrp_status"])
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
        generate_plots(all_results, args.output_dir)

        logger.info("\n%s\nE15 SUMMARY — %s\n%s",
                    "=" * 80, _IMPLEMENTATION_CHOICES, "=" * 80)
        logger.info("%-20s %8s %8s %8s %8s %8s %8s %8s",
                    "Model", "Faces", "VPs", "Cov%",
                    "t_vis(s)", "t_sc(s)", "t_vrp(s)", "t_mapf(s)")
        logger.info("-" * 80)
        models = sorted(set(r["model"] for r in all_results))
        for m in models:
            mr = [r for r in all_results if r["model"] == m]
            if mr:
                def _m(k):
                    vals = [r.get(k, 0) for r in mr]
                    return float(np.mean(vals))
                logger.info("%-20s %8.0f %8.1f %8.2f %8.2f %8.2f %8.2f %8.2f",
                            m, _m("mesh_faces"), _m("num_viewpoints"),
                            _m("coverage") * 100,
                            _m("t_vis"), _m("t_opt"), _m("t_vrp"), _m("t_mapf"))


if __name__ == "__main__":
    main()
