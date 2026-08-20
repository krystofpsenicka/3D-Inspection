#!/usr/bin/env python3
"""E16: do the two VRP cuts actually help?  Root-LP strength + time-to-5%-gap.

Why this exists
---------------
E11 (120 s) and E15 (1800 s) both answered "no measurable difference", but
neither run could have answered the question:

  * every run in both experiments terminated on the **time limit**, never on the
    5% gap target (E08: 120.85 +- 0.25 s of 120 s; E15: 1801.0 +- 0.2 s of
    1800 s).  cuOpt *was* given CUOPT_MIP_RELATIVE_GAP=0.05, so it would have
    stopped itself had its internal gap closed.  It did not.
  * the `gap` those experiments log is not an optimality gap.  It divides a
    **metres** incumbent (VRPResult.objective_value = b*T + (1-b)*C) by a
    **normalised** dual bound (cuOpt bounds b*T/T_norm + (1-b)*C/C_norm with
    T_norm = T_lb, C_norm = K*T_lb).  That is why it is pinned at ~0.997
    everywhere regardless of solve quality.
  * consequently E11/E15 ranked the four cut configs by b*T + (1-b)*C, which is
    **not** the objective cuOpt minimises -- at K=5 it over-weights total cost
    fivefold relative to the model.

This experiment fixes the measurement and attacks the question from the cheap
end first.

    arm=root   Root LP relaxation bound per cut config (seconds per solve).
               "Does the cut tighten the relaxation" is *defined* by this
               number; it does not require branch-and-bound to close.  Also
               records model size, so a cut that only removes arcs (reach) is
               visibly distinguishable from one that adds rows (pair).

    arm=close  Branch-and-bound on instances small enough that the 5% gap is
               actually reachable.  Reports time-to-termination, whether the
               solver stopped early (== gap target met) and the **true** gap.
               A config that proves optimality faster is the operational
               benefit the cuts are supposed to deliver.

    arm=budget Full-size (50 waypoint) run at a large budget, with correct gap
               reporting.  Run this LAST and only if the small instances say
               something interesting.

Correct quantities (all computed here, no library change needed)
---------------------------------------------------------------
    T_lb      = max_j min_v ( c[d_v, j] + c[j, d_v] )        (mip_model.py:154)
    obj_mip   = b*makespan + (1-b)*total_cost / K            (metres)
    obj_norm  = obj_mip / T_lb                               (what cuOpt sees)
    true_gap  = (obj_norm - best_bound) / obj_norm
    lp_bound_m = lp_bound_norm * T_lb        (valid LB on obj_mip, in metres)

`obj_reported_m` (= b*T + (1-b)*C) is also logged so rows stay comparable with
the E11/E15 tables already in the paper.

Usage
-----
    # 1. cheap, decisive, run this first (~minutes)
    python -m experiments.e16_vrp_cut_strength --arm root --resume

    # 2. small instances where 5% is reachable (hours, unattended)
    python -m experiments.e16_vrp_cut_strength --arm close --resume

    # 3. only if needed
    python -m experiments.e16_vrp_cut_strength --arm budget --resume

    python -m experiments.e16_vrp_cut_strength --plots_only
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

from experiments.common.config import RESULTS_DIR, SEEDS_5
from experiments.common.persistence import load_run_result, save_run_result

_RUNTIME_IMPORT_ERROR: ImportError | None = None
try:
    import cupy as cp
    import pulp

    from experiments.common.runner import free_gpu_memory, handle_row_exception
    # Reuse E11's instance construction so E11/E15/E16 solve identically-built
    # instances and only the measurement differs.
    from experiments.e11_vrp_cuts_ablation import (
        CUT_CONFIGS,
        _build_instance,
        _setup_mesh_and_sampler,
    )
    from VRP.core.types import VRPBackend
    from VRP.vrp._helpers import nearest_neighbor_warmstart, per_vehicle_costs
    from VRP.vrp.mip_model import build_vrp_mip
    from VRP.vrp.vrp_solver import solve_vrp

    _RUNTIME_AVAILABLE = True
except ImportError as _e:
    cp = None
    _RUNTIME_AVAILABLE = False
    _RUNTIME_IMPORT_ERROR = _e

logger = logging.getLogger(__name__)

N_ROBOTS = 5
VRP_ALPHA = 0.5           # beta in the paper
MIP_GAP = 0.05
LP_TIME_LIMIT = 600       # root LP only; should finish in seconds

# Small enough that branch-and-bound has a chance of proving 5%.
CLOSE_WAYPOINTS = [10, 15, 20, 25]
CLOSE_TIME_LIMIT = 900

BUDGET_WAYPOINTS = [50]
BUDGET_TIME_LIMIT = 7200


# ── correct objective / gap arithmetic ─────────────────────────────────────


def compute_t_lb(dist_matrix_np: np.ndarray, depots: list[int]) -> float:
    """T_lb = max_j min_v (c[d_v,j] + c[j,d_v]) over customers j.

    Mirrors VRP/vrp/mip_model.py exactly; it is both the makespan lower bound
    and the objective normalisation constant (T_norm = T_lb, C_norm = K*T_lb).
    """
    n = dist_matrix_np.shape[0]
    depot_set = set(int(d) for d in depots)
    customers = [j for j in range(n) if j not in depot_set]
    t_lb = 0.0
    for j in customers:
        rt = min(
            float(dist_matrix_np[d, j] + dist_matrix_np[j, d]) for d in depot_set
        )
        t_lb = max(t_lb, rt)
    return t_lb


def objectives(makespan: float, total_cost: float, k: int, beta: float, t_lb: float):
    """Return (obj_mip_m, obj_norm, obj_reported_m).

    obj_mip_m is the metre-valued objective the MIP actually minimises (up to
    the 1/T_lb scale); obj_reported_m is the legacy E11/E15 column.
    """
    obj_mip_m = beta * makespan + (1.0 - beta) * total_cost / max(k, 1)
    obj_norm = obj_mip_m / t_lb if t_lb > 0 else float("nan")
    obj_reported_m = beta * makespan + (1.0 - beta) * total_cost
    return obj_mip_m, obj_norm, obj_reported_m


def true_gap(obj_norm: float, best_bound_norm: float) -> float:
    """Relative optimality gap in the solver's own units. NaN if unavailable."""
    if not np.isfinite(obj_norm) or obj_norm <= 0:
        return float("nan")
    if not np.isfinite(best_bound_norm) or best_bound_norm <= 0:
        return float("nan")
    return (obj_norm - best_bound_norm) / obj_norm


# ── arm: root LP ───────────────────────────────────────────────────────────


def _root_lp(dist_matrix, K, home_indices, label, beta_flag, pair_flag, t_lb):
    """Root LP relaxation bound for one cut config.

    Builds the same MIP the solvers build, relaxes the binaries, and solves the
    LP with HiGHS. The LP objective is a valid lower bound on the integer
    optimum, in the model's normalised units; multiplying by T_lb puts it in
    metres on the same scale as obj_mip_m.
    """
    dm_np = cp.asnumpy(dist_matrix)
    ws_routes = nearest_neighbor_warmstart(dist_matrix, K, home_indices)

    t0 = time.perf_counter()
    prob, _warm = build_vrp_mip(
        dm_np,
        K,
        home_indices,
        alpha=VRP_ALPHA,
        warm_start_routes=ws_routes,
        mip_gap=MIP_GAP,
        beta_aware_filter=beta_flag,
        forbidden_pair_cuts=pair_flag,
    )
    build_s = time.perf_counter() - t0

    variables = prob.variables()
    n_vars = len(variables)
    n_binaries = sum(1 for v in variables if v.cat == pulp.LpBinary)
    n_constraints = len(prob.constraints)

    # Relax: LpBinary already carries bounds [0, 1]; set them explicitly so the
    # relaxation is well defined regardless of PuLP version.
    for v in variables:
        if v.cat == pulp.LpBinary:
            v.cat = pulp.LpContinuous
            v.lowBound = 0.0 if v.lowBound is None else v.lowBound
            v.upBound = 1.0 if v.upBound is None else v.upBound

    t0 = time.perf_counter()
    prob.solve(pulp.HiGHS(timeLimit=LP_TIME_LIMIT, msg=0))
    lp_solve_s = time.perf_counter() - t0

    status_str = pulp.LpStatus[prob.status]
    lp_bound_norm = float("nan")
    if prob.status == pulp.constants.LpStatusOptimal:
        val = pulp.value(prob.objective)
        if val is not None:
            lp_bound_norm = float(val)

    return {
        "arm": "root",
        "config": label,
        "t_lb": t_lb,
        "lp_bound_norm": lp_bound_norm,
        "lp_bound_m": lp_bound_norm * t_lb if np.isfinite(lp_bound_norm) else float("nan"),
        "n_vars": n_vars,
        "n_binaries": n_binaries,
        "n_constraints": n_constraints,
        "build_s": build_s,
        "lp_solve_s": lp_solve_s,
        "lp_status": status_str,
        "status": "success" if np.isfinite(lp_bound_norm) else f"lp_failed ({status_str})",
    }


# ── arm: branch-and-bound to the gap ───────────────────────────────────────


def _solve_to_gap(dist_matrix, K, home_indices, label, beta_flag, pair_flag,
                  backend, time_limit, t_lb, arm):
    ws_routes = nearest_neighbor_warmstart(dist_matrix, K, home_indices)
    ws_pv = per_vehicle_costs(ws_routes, dist_matrix, home_indices)
    ws_mk = max(ws_pv) if ws_pv else float("nan")
    ws_tot = float(sum(ws_pv))
    ws_obj_mip, ws_obj_norm, ws_obj_rep = objectives(ws_mk, ws_tot, K, VRP_ALPHA, t_lb)

    t0 = time.perf_counter()
    vr = solve_vrp(
        dist_matrix=dist_matrix,
        num_vehicles=K,
        depots=home_indices,
        alpha=VRP_ALPHA,
        backend=backend,
        time_limit=time_limit,
        mip_gap=MIP_GAP,
        beta_aware_filter=beta_flag,
        forbidden_pair_cuts=pair_flag,
    )
    solve_time = time.perf_counter() - t0

    per_v = per_vehicle_costs(vr.routes, dist_matrix, home_indices)
    makespan = max(per_v) if per_v else float("nan")
    total_cost = float(vr.total_cost)
    bound_norm = float(vr.best_bound)
    obj_mip, obj_norm, obj_rep = objectives(makespan, total_cost, K, VRP_ALPHA, t_lb)
    gap = true_gap(obj_norm, bound_norm)

    # The solver was given mip_gap=MIP_GAP, so terminating before the wall clock
    # *is* the gap target being met. A 2% margin absorbs teardown overhead.
    terminated_early = bool(solve_time < 0.98 * time_limit)

    improvement = ws_obj_mip - obj_mip
    return {
        "arm": arm,
        "config": label,
        "time_limit": time_limit,
        "solve_time": solve_time,
        "terminated_early": terminated_early,
        "makespan": makespan,
        "total_cost": total_cost,
        "t_lb": t_lb,
        "obj_mip_m": obj_mip,
        "obj_norm": obj_norm,
        "obj_reported_m": obj_rep,          # legacy E11/E15 column
        "best_bound_norm": bound_norm,
        "best_bound_m": bound_norm * t_lb if np.isfinite(bound_norm) else float("nan"),
        "true_gap": gap,
        "gap_reached": bool(np.isfinite(gap) and gap <= MIP_GAP),
        "warm_obj_mip_m": ws_obj_mip,
        "warm_makespan": ws_mk,
        "warm_total": ws_tot,
        "improvement_abs": improvement,
        "improvement_pct": (100.0 * improvement / ws_obj_mip) if ws_obj_mip else float("nan"),
        "status": "success",
    }


def _crashed_row(arm, label, exc):
    row = {"arm": arm, "config": label, "status": f"crashed: {exc}"}
    for key in ("t_lb", "lp_bound_norm", "lp_bound_m", "obj_mip_m", "obj_norm",
                "obj_reported_m", "best_bound_norm", "true_gap", "solve_time",
                "makespan", "total_cost"):
        row.setdefault(key, float("nan"))
    row.setdefault("gap_reached", False)
    row.setdefault("terminated_early", False)
    return row


# ── summary ────────────────────────────────────────────────────────────────


def _agg(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return float("nan"), float("nan"), 0
    if len(vals) == 1:
        return vals[0], 0.0, 1
    return float(np.mean(vals)), float(np.std(vals, ddof=1)), len(vals)


def _print_summary(results):
    ok = [r for r in results if r.get("status") == "success"]

    root = [r for r in ok if r.get("arm") == "root"]
    if root:
        logger.info("")
        logger.info("=== arm=root : root LP bound per cut config "
                    "(higher = tighter relaxation) ===")
        logger.info("%-6s %-8s %14s %12s %10s %8s %8s",
                    "wp", "config", "LP bound (m)", "LP norm", "binaries", "rows", "lp_s")
        for nwp in sorted({r["n_waypoints"] for r in root}):
            base = None
            for label, _b, _p in CUT_CONFIGS:
                rows = [r for r in root if r["n_waypoints"] == nwp and r["config"] == label]
                if not rows:
                    continue
                m, sd, n = _agg([r["lp_bound_m"] for r in rows])
                if label == "none":
                    base = m
                delta = ""
                if base and np.isfinite(m) and base > 0 and label != "none":
                    delta = f"  ({100.0 * (m - base) / base:+.2f}% vs none)"
                logger.info("%-6d %-8s %8.3f+-%-5.3f %12.5f %10.0f %8.0f %8.1f%s",
                            nwp, label, m, sd,
                            _agg([r["lp_bound_norm"] for r in rows])[0],
                            _agg([r["n_binaries"] for r in rows])[0],
                            _agg([r["n_constraints"] for r in rows])[0],
                            _agg([r["lp_solve_s"] for r in rows])[0], delta)
        logger.info("A cut helps the relaxation iff its LP bound is strictly "
                    "above `none` by more than seed noise.")

    for arm in ("close", "budget"):
        rows_arm = [r for r in ok if r.get("arm") == arm]
        if not rows_arm:
            continue
        logger.info("")
        logger.info("=== arm=%s : branch-and-bound (mip_gap=%.0f%%) ===", arm, MIP_GAP * 100)
        logger.info("%-6s %-8s %12s %10s %10s %9s %9s",
                    "wp", "config", "obj_mip (m)", "true_gap", "solve_s", "early/gap", "n")
        for nwp in sorted({r["n_waypoints"] for r in rows_arm}):
            for label, _b, _p in CUT_CONFIGS:
                rows = [r for r in rows_arm
                        if r["n_waypoints"] == nwp and r["config"] == label]
                if not rows:
                    continue
                om, osd, n = _agg([r["obj_mip_m"] for r in rows])
                gm, _gsd, _ = _agg([r["true_gap"] for r in rows])
                tm, _tsd, _ = _agg([r["solve_time"] for r in rows])
                closed = sum(1 for r in rows if r.get("gap_reached"))
                early = sum(1 for r in rows if r.get("terminated_early"))
                logger.info("%-6d %-8s %6.3f+-%-5.3f %9.4f %10.1f %3d/%-3d/%-3d %6d",
                            nwp, label, om, osd, gm, tm, early, closed, len(rows), n)
        logger.info("early/gap = (runs that stopped before the wall clock) / "
                    "(runs whose recomputed gap is <= %.0f%%) / (runs). The two "
                    "should agree; a mismatch means the dual bound was "
                    "unavailable.", MIP_GAP * 100)

    bad = [r for r in results if r.get("status") != "success"]
    if bad:
        logger.info("")
        logger.info("%d non-success rows: %s", len(bad),
                    sorted({r.get("status", "?")[:60] for r in bad}))


# ── main ───────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="E16: VRP cut strength (root LP + time-to-gap)")
    p.add_argument("--arm", choices=["root", "close", "budget"], default="root")
    p.add_argument("--waypoints", type=int, nargs="+", default=None,
                   help="override the arm's default waypoint counts")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_5)
    p.add_argument("--time_limit", type=int, default=None,
                   help="override the arm's default B&B budget (seconds)")
    p.add_argument("--backend", choices=["cuopt", "highs"], default="cuopt")
    p.add_argument("--output_dir", default=os.path.join(RESULTS_DIR, "e16_vrp_cut_strength"))
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

    if args.arm == "root":
        waypoints = args.waypoints or (CLOSE_WAYPOINTS + BUDGET_WAYPOINTS)
        time_limit = None
    elif args.arm == "close":
        waypoints = args.waypoints or CLOSE_WAYPOINTS
        time_limit = args.time_limit or CLOSE_TIME_LIMIT
    else:
        waypoints = args.waypoints or BUDGET_WAYPOINTS
        time_limit = args.time_limit or BUDGET_TIME_LIMIT

    backend = VRPBackend.CUOPT if args.backend == "cuopt" else VRPBackend.HIGHS
    og, sampler, bmin, bmax = _setup_mesh_and_sampler()

    for nwp in waypoints:
        for seed in args.seeds:
            # All cut configs solve the identical (seeded) instance.
            instance = None
            t_lb = None
            for label, beta_flag, pair_flag in CUT_CONFIGS:
                stem = f"arm={args.arm}_wp={nwp}_seed={seed}_cfg={label}"
                rpath = os.path.join(raw_dir, stem)
                if args.resume and os.path.exists(rpath + ".json"):
                    all_results.append(load_run_result(rpath))
                    logger.info("[resume] skip %s", stem)
                    continue
                logger.info("Running arm=%s wp=%d seed=%d cfg=%s",
                            args.arm, nwp, seed, label)
                try:
                    if instance is None:
                        instance = _build_instance(nwp, seed, og, sampler, bmin, bmax)
                        K, home_indices, dist_matrix = instance
                        t_lb = compute_t_lb(cp.asnumpy(dist_matrix), home_indices)
                        logger.info("  T_lb = %.3f m", t_lb)
                    K, home_indices, dist_matrix = instance
                    if args.arm == "root":
                        row = _root_lp(dist_matrix, K, home_indices, label,
                                       beta_flag, pair_flag, t_lb)
                    else:
                        row = _solve_to_gap(dist_matrix, K, home_indices, label,
                                            beta_flag, pair_flag, backend,
                                            time_limit, t_lb, args.arm)
                except Exception as exc:  # noqa: BLE001
                    handle_row_exception(exc, stem, resume=args.resume)
                    row = _crashed_row(args.arm, label, exc)
                row.update({"n_waypoints": nwp, "seed": seed, "K": N_ROBOTS,
                            "beta": VRP_ALPHA})
                all_results.append(row)
                save_run_result(row, rpath)
                if args.arm == "root":
                    logger.info("  LP bound = %.4f (norm) = %.3f m  [%s]",
                                row.get("lp_bound_norm", float("nan")),
                                row.get("lp_bound_m", float("nan")), row["status"])
                else:
                    logger.info("  obj_mip=%.3f m  true_gap=%.4f  closed=%s  %.1fs",
                                row.get("obj_mip_m", float("nan")),
                                row.get("true_gap", float("nan")),
                                row.get("terminated_early"), row.get("solve_time", 0.0))
            free_gpu_memory()

    _print_summary(all_results)


if __name__ == "__main__":
    main()
