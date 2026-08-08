#!/usr/bin/env python3
"""E13: post-smoothing collision & coverage audit.

Reviewer R4.12 / R2.2: smoothing is applied *after* the space-time reservation
enforces collision avoidance, so a B-spline shortcut can move the path off the
reserved cells and reintroduce collisions or safety-margin violations; and it is
not shown that coverage survives smoothing. This experiment runs the full
pipeline and audits the *final smoothed + densified* trajectories that the
pipeline actually outputs:

* Environment safety: count trajectory samples that fall inside the inflated
  fine-resolution occupancy grid (should be 0 - OMPL guarantees this, we verify).
* Inter-robot safety: minimum pairwise centre separation over the window where
  both robots are in motion, and the number of robot pairs that ever come closer
  than d_safe = 2*ROBOT_RADIUS. (We truncate each pair to its overlapping active
  window so a robot parked at its final waypoint doesn't create phantom overlaps.)
* Coverage survival: for every selected inspection waypoint, the closest approach
  of the owning robot's smoothed trajectory. A waypoint is "missed" if no sample
  comes within ``visit_tol`` metres, which would mean smoothing dropped an
  inspection pose and degraded coverage.

    python -m experiments.e13_collision_coverage_audit
    python -m experiments.e13_collision_coverage_audit --models duke_of_lancaster wolf0
    python -m experiments.e13_collision_coverage_audit --plots_only
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.common.config import RESULTS_DIR, SEEDS_5, ModelConfig
from experiments.common.persistence import load_run_result, save_run_result

_RUNTIME_IMPORT_ERROR: ImportError | None = None
try:
    import cupy as cp

    from experiments.common.pipeline_run import run_pipeline_to_routes
    from experiments.common.pipeline_setup import DegenerateNormalsError, PipelineContext
    from experiments.common.runner import free_gpu_memory, handle_row_exception
    from VRP.core.constants import ROBOT_RADIUS, SPACE_TIME_DWELL_S
    from VRP.mapf.mapf_planner import MultiAgentPathPlanner
    from VRP.utils.collision import find_environment_collisions, find_trajectory_collisions

    _RUNTIME_AVAILABLE = True
except ImportError as _e:
    cp = None
    _RUNTIME_AVAILABLE = False
    _RUNTIME_IMPORT_ERROR = _e

logger = logging.getLogger(__name__)

VRP_ALPHA = 0.5
FLEET_SIZE = 5
D_SAFE = 2.0 * 0.35  # 2 * ROBOT_RADIUS
VISIT_TOL = 0.5      # metres; a waypoint is "visited" if a sample lands within this


def _coordinated_mapf(art):
    routes = [
        [art.home_indices[i]] + list(r) + [art.home_indices[i]]
        for i, r in enumerate(art.routes)
    ]
    starts = [np.asarray(x, dtype=np.float32) for x in art.robot_starts]
    wp_pos = cp.asarray(art.all_positions, dtype=cp.float32)
    wp_rot = cp.asarray(art.all_rotmats, dtype=cp.float32)
    planner = MultiAgentPathPlanner(start_positions=starts, og=art.og)
    return planner.execute(
        routes=routes, waypoint_positions=wp_pos, waypoint_rotmats=wp_rot,
        home_indices=set(art.home_indices), dist_matrix=art.dist_matrix,
        alpha=VRP_ALPHA, dwell_seconds=SPACE_TIME_DWELL_S, n_priority_trials=20,
    ), routes


def _min_separation(trajs):
    """Minimum inter-robot centre separation over the overlapping active window,
    and the number of robot pairs that ever breach d_safe. Uses per-pair
    truncation to the shorter trajectory so a finished (parked) robot doesn't
    generate phantom proximity against a still-moving one."""
    R = len(trajs)
    min_sep = float("inf")
    breach_pairs = 0
    for a in range(R):
        for b in range(a + 1, R):
            ta, tb = np.asarray(trajs[a])[:, :3], np.asarray(trajs[b])[:, :3]
            L = min(len(ta), len(tb))
            if L == 0:
                continue
            d = np.linalg.norm(ta[:L] - tb[:L], axis=1)
            m = float(d.min())
            min_sep = min(min_sep, m)
            if m < D_SAFE:
                breach_pairs += 1
    return (min_sep if min_sep != float("inf") else float("nan")), breach_pairs


def _coverage_survival(art, routes, exec_result):
    """Closest approach of each robot's trajectory to each of its inspection
    waypoints; count waypoints never visited within VISIT_TOL."""
    trajs = exec_result.all_traj_positions
    all_pos = np.asarray(art.all_positions)
    worst = 0.0
    missed = 0
    total = 0
    for r, route in enumerate(routes):
        customers = route[1:-1]  # strip depots
        if not customers or r >= len(trajs) or len(trajs[r]) == 0:
            continue
        traj = np.asarray(trajs[r])[:, :3]
        for node in customers:
            total += 1
            wp = all_pos[node]
            closest = float(np.linalg.norm(traj - wp, axis=1).min())
            worst = max(worst, closest)
            if closest > VISIT_TOL:
                missed += 1
    return worst, missed, total


def _run_single(ctx, cfg, seed):
    art = run_pipeline_to_routes(ctx, cfg, seed, fleet_size=FLEET_SIZE, vrp_alpha=VRP_ALPHA)
    row = {"model": cfg.name, "seed": seed, "num_viewpoints": art.num_viewpoints,
           "setcover_coverage": art.coverage, "vrp_status": art.vrp_status}
    if not art.routes or not any(art.routes):
        row["vrp_status"] = "empty_routes"
        return row

    exec_result, routes = _coordinated_mapf(art)
    trajs = exec_result.all_traj_positions

    env_counts = find_environment_collisions(trajs, art.og)
    inter = find_trajectory_collisions(trajs, radius=ROBOT_RADIUS)
    min_sep, breach_pairs = _min_separation(trajs)
    worst_visit, missed_wp, total_wp = _coverage_survival(art, routes, exec_result)

    row.update({
        "env_collision_samples": int(sum(env_counts)),
        "env_collision_per_robot": [int(c) for c in env_counts],
        "inter_robot_events_padded": len(inter),   # padding-inclusive (reference)
        "min_separation_m": min_sep,
        "d_safe_m": D_SAFE,
        "d_safe_breach_pairs": breach_pairs,
        "astar_fail": int(sum(exec_result.fail_counts)),
        "worst_waypoint_approach_m": worst_visit,
        "waypoints_missed": missed_wp,
        "waypoints_total": total_wp,
        "makespan_s": exec_result.actual_makespan,
    })
    logger.info("  env=%d min_sep=%.2fm breach_pairs=%d missed_wp=%d/%d worst_visit=%.2fm",
                row["env_collision_samples"], min_sep, breach_pairs,
                missed_wp, total_wp, worst_visit)
    return row


def _print_summary(results):
    ok = [r for r in results if "min_separation_m" in r]
    if not ok:
        logger.info("No audited runs.")
        return
    logger.info("\n%s\nE13 SUMMARY - post-smoothing audit (mean over runs)\n%s",
                "=" * 72, "=" * 72)
    logger.info("  runs audited                : %d", len(ok))
    logger.info("  env-collision samples (sum) : %.2f", np.mean([r["env_collision_samples"] for r in ok]))
    logger.info("  min inter-robot separation  : %.2f m (d_safe=%.2f m)",
                np.nanmin([r["min_separation_m"] for r in ok]), D_SAFE)
    logger.info("  d_safe breach pairs / run   : %.2f", np.mean([r["d_safe_breach_pairs"] for r in ok]))
    logger.info("  waypoints missed / run      : %.2f", np.mean([r["waypoints_missed"] for r in ok]))
    logger.info("  worst waypoint approach     : %.2f m (tol=%.2f m)",
                np.max([r["worst_waypoint_approach_m"] for r in ok]), VISIT_TOL)


def main():
    p = argparse.ArgumentParser(description="E13: collision & coverage audit")
    p.add_argument("--models", nargs="+", default=["duke_of_lancaster"])
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_5)
    p.add_argument("--output_dir", default=os.path.join(RESULTS_DIR, "e13_collision_coverage_audit"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--plots_only", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)-8s %(name)s: %(message)s")

    raw_dir = os.path.join(args.output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    all_results: list[dict] = []

    if args.plots_only:
        for f in sorted(os.listdir(raw_dir)):
            if f.endswith(".json"):
                all_results.append(load_run_result(os.path.join(raw_dir, f[:-5])))
        _print_summary(all_results)
        return

    if not _RUNTIME_AVAILABLE:
        raise SystemExit(f"Runtime imports unavailable ({_RUNTIME_IMPORT_ERROR}).")

    def _cfg(name):
        return (ModelConfig.duke_of_lancaster() if name == "duke_of_lancaster"
                else ModelConfig.tosca(name))

    for name in args.models:
        cfg = _cfg(name)
        try:
            ctx = PipelineContext(cfg)
            ctx.load_mesh()
            ctx.sample_surface(seed=42)
            ctx.build_sampling_og()
        except DegenerateNormalsError as e:
            logger.warning("Skipping %s: %s", name, e)
            continue

        for seed in args.seeds:
            stem = f"model={name}_seed={seed}"
            rpath = os.path.join(raw_dir, stem)
            if args.resume and os.path.exists(rpath + ".json"):
                all_results.append(load_run_result(rpath))
                logger.info("[resume] skip %s", stem)
                continue
            logger.info("Auditing %s seed=%d", name, seed)
            try:
                row = _run_single(ctx, cfg, seed)
                all_results.append(row)
                save_run_result(row, rpath)
            except Exception as exc:  # noqa: BLE001
                handle_row_exception(exc, stem, resume=args.resume)
            finally:
                free_gpu_memory()

    _print_summary(all_results)


if __name__ == "__main__":
    main()
