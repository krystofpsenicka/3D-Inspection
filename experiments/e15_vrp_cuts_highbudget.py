#!/usr/bin/env python3
"""E15: high-budget VRP cut ablation - do the MIP additions actually improve the
solution once the solver has time to beat the warm start?

E11 runs the same four-way cut ablation (none / reach / pair / both) at a fixed
120 s budget, but at that budget cuOpt times out on every config and essentially
returns the nearest-neighbour warm start, so the cuts cannot be distinguished.
E15 gives cuOpt a much larger budget (default 1800 s) so branch-and-bound can
actually improve on the warm start, and for each config reports:

    * final blended objective, makespan, total cost,
    * solver dual bound + achieved optimality gap,
    * the warm-start objective the solve started from, and
    * the improvement over that warm start (absolute and %) -- the key quantity:
      if a config never beats its warm start, the solver added nothing at this
      budget; the story is whether the cuts let it improve more / more often.

All four configs solve the identical seeded instance. Because a single solve can
take the full budget, keep the instance/seed count small and run this LAST.

    python -m experiments.e15_vrp_cuts_highbudget --time_limit 1800 --waypoints 50 --seeds 42 123 7
    python -m experiments.e15_vrp_cuts_highbudget --time_limit 3600 --waypoints 50 --seeds 42
    python -m experiments.e15_vrp_cuts_highbudget --plots_only
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.common.config import RESULTS_DIR, SEEDS_3
from experiments.common.persistence import load_run_result, save_run_result

_RUNTIME_IMPORT_ERROR: ImportError | None = None
try:
    import cupy as cp

    from experiments.common.runner import free_gpu_memory, handle_row_exception
    # Reuse E11's instance construction + mesh/sampler setup so the two ablations
    # solve identically-built instances and only the budget differs.
    from experiments.e11_vrp_cuts_ablation import (
        CUT_CONFIGS,
        _build_instance,
        _setup_mesh_and_sampler,
    )
    from VRP.core.types import VRPBackend
    from VRP.vrp._helpers import nearest_neighbor_warmstart, per_vehicle_costs
    from VRP.vrp.vrp_solver import solve_vrp

    _RUNTIME_AVAILABLE = True
except ImportError as _e:
    cp = None
    _RUNTIME_AVAILABLE = False
    _RUNTIME_IMPORT_ERROR = _e

logger = logging.getLogger(__name__)

N_ROBOTS = 5
VRP_ALPHA = 0.5
MIP_GAP = 0.05


def _warm_objective(dist_matrix, K, home_indices, alpha):
    """Blended objective of the nearest-neighbour warm start that every config
    starts from (solve_vrp seeds the MIP with exactly this)."""
    ws_routes = nearest_neighbor_warmstart(dist_matrix, K, home_indices)
    pv = per_vehicle_costs(ws_routes, dist_matrix, home_indices)
    mk = max(pv) if pv else float("nan")
    tot = float(sum(pv))
    return alpha * mk + (1.0 - alpha) * tot, mk, tot


def _solve_one(dist_matrix, K, home_indices, label, beta_flag, pair_flag,
               backend, time_limit, alpha):
    warm_obj, warm_mk, warm_tot = _warm_objective(dist_matrix, K, home_indices, alpha)
    t0 = time.perf_counter()
    try:
        vr = solve_vrp(
            dist_matrix=dist_matrix,
            num_vehicles=K,
            depots=home_indices,
            alpha=alpha,
            backend=backend,
            time_limit=time_limit,
            mip_gap=MIP_GAP,
            beta_aware_filter=beta_flag,
            forbidden_pair_cuts=pair_flag,
        )
        solve_time = time.perf_counter() - t0
        per_v = per_vehicle_costs(vr.routes, dist_matrix, home_indices)
        makespan = max(per_v) if per_v else float("nan")
        obj = float(vr.objective_value)
        bound = float(vr.best_bound)
        gap = (obj - bound) / abs(obj) if bound > 0 and obj != 0 else float("nan")
        improvement = warm_obj - obj  # > 0  => solver beat the warm start
        return {
            "config": label,
            "solve_time": solve_time,
            "time_limit": time_limit,
            "objective": obj,
            "makespan": makespan,
            "total_cost": float(vr.total_cost),
            "best_bound": bound,
            "gap": gap,
            "gap_reached": bool(not np.isnan(gap) and gap <= MIP_GAP),
            "warm_objective": warm_obj,
            "warm_makespan": warm_mk,
            "warm_total": warm_tot,
            "improvement_abs": improvement,
            "improvement_pct": (100.0 * improvement / warm_obj) if warm_obj else float("nan"),
            "improved": bool(improvement > 1e-6),
            "status": vr.status,
        }
    except Exception as exc:  # noqa: BLE001 - record failure, keep other configs
        return {
            "config": label, "solve_time": time.perf_counter() - t0,
            "time_limit": time_limit, "objective": float("nan"),
            "makespan": float("nan"), "total_cost": float("nan"),
            "best_bound": float("nan"), "gap": float("nan"), "gap_reached": False,
            "warm_objective": warm_obj, "warm_makespan": warm_mk, "warm_total": warm_tot,
            "improvement_abs": float("nan"), "improvement_pct": float("nan"),
            "improved": False, "status": f"failed: {exc}",
        }


def _print_summary(results: list[dict]):
    labels = [c[0] for c in CUT_CONFIGS]
    logger.info("\n%s\nE15 SUMMARY - high-budget VRP cut ablation (mean over seeds/waypoints)\n%s",
                "=" * 92, "=" * 92)
    logger.info("%-8s %10s %10s %10s %10s %9s %9s", "config", "obj", "warm_obj",
                "improv%", "gap%", "solve_s", "improved%")
    for lab in labels:
        rows = [r for r in results if r["config"] == lab and not np.isnan(r["objective"])]
        if not rows:
            logger.info("%-8s   (no successful solves)", lab)
            continue
        obj = np.mean([r["objective"] for r in rows])
        wobj = np.mean([r["warm_objective"] for r in rows])
        imp = np.nanmean([r["improvement_pct"] for r in rows])
        gap = np.nanmean([r["gap"] for r in rows]) * 100
        st = np.mean([r["solve_time"] for r in rows])
        impd = np.mean([1.0 if r["improved"] else 0.0 for r in rows]) * 100
        logger.info("%-8s %10.4f %10.4f %9.2f %9.2f %10.1f %8.0f",
                    lab, obj, wobj, imp, gap, st, impd)


def main():
    p = argparse.ArgumentParser(description="E15: high-budget VRP cut ablation")
    p.add_argument("--time_limit", type=int, default=1800,
                   help="Per-solve cuOpt budget in seconds (default 1800 = 30 min).")
    p.add_argument("--waypoints", type=int, nargs="+", default=[50])
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_3)
    p.add_argument("--alpha", type=float, default=VRP_ALPHA,
                   help="Objective blend (0=total distance, 1=makespan). Default 0.5 "
                        "matches the pipeline; 0.0 gives a tighter MTZ bound if you "
                        "want to probe the gap story.")
    p.add_argument("--backend", choices=["cuopt", "highs"], default="cuopt")
    p.add_argument("--output_dir", default=os.path.join(RESULTS_DIR, "e15_vrp_cuts_highbudget"))
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

    backend = VRPBackend.CUOPT if args.backend == "cuopt" else VRPBackend.HIGHS
    og, sampler, bmin, bmax = _setup_mesh_and_sampler()

    for nwp in args.waypoints:
        for seed in args.seeds:
            # Build the (seeded) instance once; all four configs reuse it.
            instance = None
            for label, beta_flag, pair_flag in CUT_CONFIGS:
                stem = f"wp={nwp}_seed={seed}_tl={args.time_limit}_a={args.alpha}_cfg={label}"
                rpath = os.path.join(raw_dir, stem)
                if args.resume and os.path.exists(rpath + ".json"):
                    all_results.append(load_run_result(rpath))
                    logger.info("[resume] skip %s", stem)
                    continue
                logger.info("Running wp=%d seed=%d tl=%ds cfg=%s", nwp, seed, args.time_limit, label)
                try:
                    if instance is None:
                        instance = _build_instance(nwp, seed, og, sampler, bmin, bmax)
                    K, home_indices, dist_matrix = instance
                    row = _solve_one(dist_matrix, K, home_indices, label,
                                     beta_flag, pair_flag, backend, args.time_limit, args.alpha)
                except Exception as exc:  # noqa: BLE001
                    handle_row_exception(exc, stem, resume=args.resume)
                    row = {"config": label, "status": f"crashed: {exc}",
                           "objective": float("nan"), "gap": float("nan"),
                           "improvement_pct": float("nan"), "improved": False,
                           "warm_objective": float("nan"), "solve_time": 0.0}
                row.update({"n_waypoints": nwp, "seed": seed, "alpha": args.alpha})
                all_results.append(row)
                save_run_result(row, rpath)
                logger.info("  obj=%.4f warm=%.4f improv=%.2f%% gap=%.3f solve=%.1fs status=%s",
                            row["objective"], row.get("warm_objective", float("nan")),
                            row.get("improvement_pct", float("nan")), row["gap"],
                            row["solve_time"], row["status"])
            free_gpu_memory()

    _print_summary(all_results)


if __name__ == "__main__":
    main()
