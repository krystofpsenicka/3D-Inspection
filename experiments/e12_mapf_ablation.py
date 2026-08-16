#!/usr/bin/env python3
"""E12: MAPF ablation - priority orderings and the value of coordination.

Two reviewer requests are answered here:

* R4.4 lists the "4D GPU space-time A* / number of orderings tried" among the
  contributions but shows no ablation. The ``orderings`` arm sweeps the priority
  ordering budget ``n_priority_trials`` in {1, 5, 10, 20} at the default fleet and
  reports the resulting β-blended objective, so the marginal value of trying more
  orderings is quantified.

* R4.11 / R4.23 note that the paper claims "collision-free" trajectories but only
  reports "below one collision per mission for K<=4", and it is unclear whether
  that number is a no-coordination baseline or the pipeline output. The
  ``collisions`` arm makes this explicit: for each fleet size it plans the fleet
  (a) *coordinated* (full priority-based planner with the shared reservation
  table) and (b) *independent* (each robot planned against an empty reservation,
  ignoring the others), then counts geometric inter-robot collisions on the final
  smoothed+densified trajectories for both. The coordinated count should be ~0
  (validating the claim); the independent count is the "<1 collision" baseline.

    python -m experiments.e12_mapf_ablation                 # both arms, Duke
    python -m experiments.e12_mapf_ablation --mode collisions --fleet_sizes 2 3 4 5 6
    python -m experiments.e12_mapf_ablation --mode orderings --trials 1 5 10 20
    python -m experiments.e12_mapf_ablation --plots_only
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
    from VRP.utils.collision import find_trajectory_collisions

    _RUNTIME_AVAILABLE = True
except ImportError as _e:
    cp = None
    _RUNTIME_AVAILABLE = False
    _RUNTIME_IMPORT_ERROR = _e

logger = logging.getLogger(__name__)

VRP_ALPHA = 0.5
DEFAULT_FLEET = 5
D_SAFE = 2.0 * 0.35  # 2 * ROBOT_RADIUS; pairwise centre separation used as the safety distance


def _full_routes(art):
    """[home] + customer route + [home] per robot, in global node indices."""
    return [
        [art.home_indices[i]] + list(r) + [art.home_indices[i]]
        for i, r in enumerate(art.routes)
    ]


def _objective(exec_result) -> float:
    mks = exec_result.actual_makespan
    tot = sum(exec_result.actual_per_vehicle_times)
    return VRP_ALPHA * mks + (1.0 - VRP_ALPHA) * tot


def _coordinated_mapf(art, n_trials: int):
    """Full priority-based planner over the whole fleet (shared reservation)."""
    routes = _full_routes(art)
    starts = [np.asarray(x, dtype=np.float32) for x in art.robot_starts]
    wp_pos = cp.asarray(art.all_positions, dtype=cp.float32)
    wp_rot = cp.asarray(art.all_rotmats, dtype=cp.float32)
    planner = MultiAgentPathPlanner(start_positions=starts, og=art.og)
    return planner.execute(
        routes=routes, waypoint_positions=wp_pos, waypoint_rotmats=wp_rot,
        home_indices=set(art.home_indices), dist_matrix=art.dist_matrix,
        alpha=VRP_ALPHA, dwell_seconds=SPACE_TIME_DWELL_S, n_priority_trials=n_trials,
    )


def _independent_trajs(art):
    """Plan each robot in isolation (own empty reservation), collect trajectories."""
    routes = _full_routes(art)
    wp_pos = cp.asarray(art.all_positions, dtype=cp.float32)
    wp_rot = cp.asarray(art.all_rotmats, dtype=cp.float32)
    trajs = []
    for r in range(len(routes)):
        start_r = [np.asarray(art.robot_starts[r], dtype=np.float32)]
        planner_r = MultiAgentPathPlanner(start_positions=start_r, og=art.og)
        exec_r = planner_r.execute(
            routes=[routes[r]], waypoint_positions=wp_pos, waypoint_rotmats=wp_rot,
            home_indices={art.home_indices[r]}, dist_matrix=art.dist_matrix,
            alpha=VRP_ALPHA, dwell_seconds=SPACE_TIME_DWELL_S, n_priority_trials=1,
        )
        if exec_r.all_traj_positions:
            trajs.append(exec_r.all_traj_positions[0])
    return trajs


def _run_orderings(ctx, cfg, seed, trials):
    art = run_pipeline_to_routes(ctx, cfg, seed, fleet_size=DEFAULT_FLEET, vrp_alpha=VRP_ALPHA)
    row = {"model": cfg.name, "seed": seed, "arm": "orderings",
           "num_viewpoints": art.num_viewpoints, "vrp_status": art.vrp_status,
           "per_trials": {}}
    if not art.routes or not any(art.routes):
        row["vrp_status"] = "empty_routes"
        return row
    for nt in trials:
        exec_result = _coordinated_mapf(art, nt)
        row["per_trials"][str(nt)] = {
            "objective": _objective(exec_result),
            "makespan_s": exec_result.actual_makespan,
            "total_time_s": sum(exec_result.actual_per_vehicle_times),
            "astar_fail": sum(exec_result.fail_counts),
            # Value of more orderings is collision-freedom (exact metric): too few
            # trials can leave a residual colliding pair.
            "true_collision_pairs": int(exec_result.true_collision_pairs),
            "true_min_separation_m": float(exec_result.true_min_separation),
        }
        logger.info("  [orderings] trials=%2d obj=%.1f mks=%.1f true_pairs=%d",
                    nt, row["per_trials"][str(nt)]["objective"],
                    exec_result.actual_makespan, exec_result.true_collision_pairs)
    return row


def _run_collisions(ctx, cfg, seed, fleet_sizes, coord_trials=5):
    rows = []
    for K in fleet_sizes:
        art = run_pipeline_to_routes(ctx, cfg, seed, fleet_size=K, vrp_alpha=VRP_ALPHA)
        row = {"model": cfg.name, "seed": seed, "arm": "collisions", "fleet_size": K,
               "num_viewpoints": art.num_viewpoints, "vrp_status": art.vrp_status}
        if not art.routes or not any(art.routes):
            row["vrp_status"] = "empty_routes"
            rows.append(row)
            continue
        coord = _coordinated_mapf(art, coord_trials)
        indep_trajs = _independent_trajs(art)
        indep_col = find_trajectory_collisions(indep_trajs, radius=ROBOT_RADIUS)
        # Coordinated: exact continuous-time collision count (authoritative).
        # Independent baseline: the per-robot plans share no committed motion, so
        # we use the dense metric (it over-counts, which only strengthens the
        # coordinated-vs-independent contrast).
        row["coord_collision_pairs"] = int(coord.true_collision_pairs)
        row["coord_min_separation_m"] = float(coord.true_min_separation)
        row["indep_collision_pairs"] = len({(a, b) for _, a, b, _ in indep_col})
        row["indep_collision_events"] = len(indep_col)
        row["coord_astar_fail"] = sum(coord.fail_counts)
        logger.info("  [collisions] K=%d coord_pairs=%d (min_sep=%.2fm) indep_pairs=%d",
                    K, row["coord_collision_pairs"], row["coord_min_separation_m"],
                    row["indep_collision_pairs"])
        rows.append(row)
    return rows


def _flatten(records):
    """Expand persisted collision records (which nest the K-sweep under
    ``by_fleet``) into one row per fleet size, leaving other rows untouched."""
    out = []
    for r in records:
        if r.get("arm") == "collisions" and "by_fleet" in r:
            out.extend(r["by_fleet"])
        else:
            out.append(r)
    return out


def _print_summary(results):
    results = _flatten(results)
    ords = [r for r in results if r.get("arm") == "orderings" and r.get("per_trials")]
    if ords:
        trials = sorted({int(k) for r in ords for k in r["per_trials"]})
        logger.info("\n%s\nE12 orderings (mean β-objective vs ordering budget)\n%s",
                    "=" * 60, "=" * 60)
        for nt in trials:
            pt = [r["per_trials"][str(nt)] for r in ords if str(nt) in r["per_trials"]]
            objs = [x["objective"] for x in pt]
            colls = [x.get("true_collision_pairs", 0) for x in pt]
            logger.info("  trials=%2d  obj=%.1f ± %.1f  true_collision_pairs(mean)=%.2f",
                        nt, np.mean(objs), np.std(objs), np.mean(colls))
    cols = [r for r in results if r.get("arm") == "collisions" and "coord_collision_pairs" in r]
    if cols:
        Ks = sorted({r["fleet_size"] for r in cols})
        logger.info("\n%s\nE12 collisions (mean colliding pairs / mission)\n%s",
                    "=" * 60, "=" * 60)
        logger.info("%-4s %14s %16s", "K", "coordinated", "independent")
        for K in Ks:
            cr = [r for r in cols if r["fleet_size"] == K]
            cc = np.mean([r["coord_collision_pairs"] for r in cr])
            ic = np.mean([r["indep_collision_pairs"] for r in cr])
            logger.info("%-4d %14.2f %16.2f", K, cc, ic)


def main():
    p = argparse.ArgumentParser(description="E12: MAPF ablation")
    p.add_argument("--mode", choices=["both", "orderings", "collisions"], default="both")
    p.add_argument("--models", nargs="+", default=["duke_of_lancaster"])
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_5)
    p.add_argument("--trials", type=int, nargs="+", default=[1, 5, 10, 20])
    p.add_argument("--fleet_sizes", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    p.add_argument("--coord_trials", type=int, default=5,
                   help="Priority-ordering budget for the coordinated plan in the "
                        "collisions arm (E13 does the full 20-trial audit).")
    p.add_argument("--output_dir", default=os.path.join(RESULTS_DIR, "e12_mapf_ablation"))
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
            if args.mode in ("both", "orderings"):
                stem = f"model={name}_seed={seed}_arm=orderings"
                rpath = os.path.join(raw_dir, stem)
                if args.resume and os.path.exists(rpath + ".json"):
                    all_results.append(load_run_result(rpath))
                    logger.info("[resume] skip %s", stem)
                else:
                    logger.info("Running orderings: %s seed=%d", name, seed)
                    try:
                        row = _run_orderings(ctx, cfg, seed, args.trials)
                        all_results.append(row)
                        save_run_result(row, rpath)
                    except Exception as exc:  # noqa: BLE001
                        handle_row_exception(exc, stem, resume=args.resume)
                    finally:
                        free_gpu_memory()

            if args.mode in ("both", "collisions"):
                stem = f"model={name}_seed={seed}_arm=collisions"
                rpath = os.path.join(raw_dir, stem)
                if args.resume and os.path.exists(rpath + ".json"):
                    all_results.append(load_run_result(rpath))
                    logger.info("[resume] skip %s", stem)
                else:
                    logger.info("Running collisions: %s seed=%d", name, seed)
                    try:
                        rows = _run_collisions(ctx, cfg, seed, args.fleet_sizes,
                                               coord_trials=args.coord_trials)
                        # Persist as one record per (seed) holding the K-sweep list.
                        rec = {"model": name, "seed": seed, "arm": "collisions",
                               "by_fleet": rows}
                        save_run_result(rec, rpath)
                        all_results.extend(rows)
                    except Exception as exc:  # noqa: BLE001
                        handle_row_exception(exc, stem, resume=args.resume)
                    finally:
                        free_gpu_memory()

    _print_summary(all_results)


if __name__ == "__main__":
    main()
