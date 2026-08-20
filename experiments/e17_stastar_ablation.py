#!/usr/bin/env python3
"""E17: is the 4D GPU space-time A* actually worth anything?

Why this exists
---------------
Reviewer 4 item 4 lists two contributions that the submitted paper claims but
never ablates: the beta-aware tour bound + forbidden-pair cuts, and the *4D GPU
space-time A*.  E16 closed the first one (answer: the cuts are inert at
production sizes).  Nothing closes the second.  Reviewers 1 and 2 make the same
complaint in general terms -- "the individual impact is not sufficiently
isolated in the experimental evaluation".

The paper currently supports the A* contribution with exactly two things, and
neither is an ablation:

  * E10's timing breakdown says MAPF is 84% of end-to-end wall-clock.  That says
    the stage is expensive, not that our version of it is good.
  * E12 (coordinated vs independent) isolates the *continuous-time conflict
    filter*, which is a different contribution living in `committed_motion.py`.

So the two claims actually attached to `space_time_search.py` are untested:

    (C1) Parallel frontier expansion (Zhou & Zeng 2015, ported to a 4D
         space-time state) is faster than expanding one state at a time.
    (C2) It pays for that with bounded suboptimality.  `space_time_astar_gpu`
         expands every frontier cell with f <= f_min + f_threshold_delta, so
         returned paths are f_threshold_delta-suboptimal.  The paper admits this
         in one sentence and never quantifies it.

Design
------
arm=legs    Leg-level microbenchmark, the decisive one.  Runs the real pipeline
            on Duke, then intercepts the first `--max_legs` single-agent A*
            queries the route planner issues and solves each one THREE ways on
            byte-identical inputs (same grid, same start/goal/t_offset, same live
            CommittedMotion):

              gpu_pf    f_delta = 2.0   -- production; whole f-layer per iteration
              gpu_seq   f_delta = 0.0   -- same code, same arrays, one f-layer
                                           at a time.  This is the honest
                                           control: it isolates the *parallel
                                           frontier* idea from every other
                                           implementation choice, because
                                           nothing else changes.
              cpu_seq                   -- sequential heapq A* over the same 4D
                                           state, same cost model, same
                                           heuristic, same obstacle test, same
                                           continuous-time filter, same
                                           holdable-goal termination.  NumPy is
                                           used to expand a state's 27
                                           successors at once, so this is a
                                           competent CPU A*, not a strawman.

            gpu_pf vs cpu_seq is the "GPU-resident" number a reviewer will look
            for; gpu_pf vs gpu_seq is the one that actually attributes the
            speedup to the algorithm rather than to CUDA.  Path costs from all
            three make C2 measurable: cpu_seq and gpu_seq are optimal for the
            given filter, so (cost(gpu_pf) / cost(gpu_seq) - 1) is the realised
            price of the relaxation, as opposed to its f_delta bound.

arm=fdelta  End-to-end consequence.  Full Duke pipeline at K=5, sweeping
            f_threshold_delta in {0, 0.5, 2, 8}, reporting MAPF wall-clock, the
            beta-blended objective, makespan, true collision pairs and A*
            failures.  Answers "does the relaxation cost anything that survives
            to the mission level, and does turning it up buy wall-clock?".

Neither arm modifies library code.  `f_threshold_delta` is already a parameter
of `space_time_astar_gpu`; `route_planner` just never passes it, so both arms
work by wrapping `VRP.mapf.route_planner.space_time_astar_gpu` (the name the
route planner resolves at call time) and restoring it afterwards.

Expected shape of the answer
----------------------------
Unknown, and E16 is the reason to say so out loud: it was written expecting the
cuts to help and found they do not.  Three outcomes are publishable here:

  * gpu_pf >> cpu_seq and gpu_pf > gpu_seq, with small cost inflation
    -> the contribution stands, report the speedup and the realised price.
  * gpu_pf ~= gpu_seq -> the parallel frontier is not what makes it fast; the
    GPU port is.  Say that, and drop "parallel-frontier" from the claim.
  * gpu_pf cheaper but materially worse paths -> f_delta = 2.0 is mistuned;
    the fdelta arm then tells us what to ship.

Usage
-----
    # 1. decisive, ~30-60 min depending on --cpu_timeout (run first)
    python -m experiments.e17_stastar_ablation --arm legs --resume

    # 2. end-to-end consequence, hours
    python -m experiments.e17_stastar_ablation --arm fdelta --resume

    python -m experiments.e17_stastar_ablation --plots_only

Cost control: the CPU reference is given `--cpu_timeout` seconds per leg (120 by
default) and is allowed to give up.  A timeout is itself a result -- it is
recorded as `cpu_timeout` and reported as a lower bound on the speedup, not
silently dropped.
"""

from __future__ import annotations

import argparse
import heapq
import logging
import math
import os
import sys
import time

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.common.config import RESULTS_DIR, SEEDS_3, ModelConfig
from experiments.common.persistence import load_run_result, save_run_result

_RUNTIME_IMPORT_ERROR: ImportError | None = None
try:
    import cupy as cp

    from experiments.common.pipeline_run import run_pipeline_to_routes
    from experiments.common.pipeline_setup import DegenerateNormalsError, PipelineContext
    from experiments.common.runner import free_gpu_memory, handle_row_exception
    from VRP.core.constants import SPACE_TIME_DWELL_S
    from VRP.mapf import route_planner as _route_planner
    from VRP.mapf.mapf_planner import MultiAgentPathPlanner
    from VRP.mapf.space_time_search import space_time_astar_gpu

    _RUNTIME_AVAILABLE = True
except ImportError as _e:  # pragma: no cover - import guard mirrors E12/E16
    cp = None
    _RUNTIME_AVAILABLE = False
    _RUNTIME_IMPORT_ERROR = _e

logger = logging.getLogger(__name__)

VRP_ALPHA = 0.5
DEFAULT_FLEET = 5

# Production default, hard-coded in space_time_astar_gpu's signature.
PRODUCTION_F_DELTA = 2.0
FDELTA_GRID = [0.0, 0.5, 2.0, 8.0]

# Sequential CPU A* budget per leg. A leg that blows this is reported as a
# timeout, which bounds the speedup from below rather than biasing it.
DEFAULT_CPU_TIMEOUT_S = 120.0
DEFAULT_MAX_LEGS = 20


# ── CPU reference: sequential space-time A* ────────────────────────────────
#
# Faithful to space_time_astar_gpu: same 4D (t,x,y,z) state, same 27 successors,
# same cost w*resolution + time_step_cost, same admissible heuristic
# (resolution * L2 + time_step_cost * Linf), same static obstacle test, same
# continuous-time committed-motion filter, same "reachable AND holdable" goal
# condition. The only difference is that it expands one state per iteration
# instead of a whole f-layer -- which is exactly the variable under test.


def _segment_min_gap_sq_np(A0, A1, B0, B1):
    """NumPy port of committed_motion.segment_min_gap_sq (same formula, same
    inf/NaN behaviour: an absent robot's inf endpoint yields NaN, and NaN < thr
    is False, i.e. no conflict)."""
    eps = 1e-9
    r0 = A0 - B0
    dr = (A1 - A0) - (B1 - B0)
    dr2 = (dr * dr).sum(-1)
    r0dr = (r0 * dr).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        tau = np.where(dr2 > eps, np.clip(-r0dr / np.maximum(dr2, eps), 0.0, 1.0), 0.0)
        rel = r0 + tau[..., None] * dr
        return (rel * rel).sum(-1)


class _CpuCommitted:
    """Host-side mirror of a live CommittedMotion, built once per leg.

    Copies the (C, T, 3) committed-position table to the host so the CPU search
    never touches the GPU -- otherwise we would be timing PCIe traffic and
    calling it a CPU baseline.
    """

    def __init__(self, committed):
        self.T = int(committed.T)
        self.two_r_sq = float(committed.two_r_sq)
        self.W = None if committed.W is None else cp.asnumpy(committed.W).astype(np.float64)

    @property
    def num_committed(self) -> int:
        return 0 if self.W is None else int(self.W.shape[0])

    def moves_valid(self, from_world, to_world, parent_t, pad: float = 0.0):
        K = len(from_world)
        if self.W is None or self.W.shape[0] == 0:
            return np.ones(K, dtype=bool)
        thr = (math.sqrt(self.two_r_sq) + pad) ** 2 if pad else self.two_r_sq
        pt = np.clip(np.asarray(parent_t, dtype=np.intp), 0, self.T - 1)
        nt = np.clip(pt + 1, 0, self.T - 1)
        B0 = self.W[:, pt, :].transpose(1, 0, 2)          # (K, C, 3)
        B1 = self.W[:, nt, :].transpose(1, 0, 2)
        A0 = np.asarray(from_world, dtype=np.float64)[:, None, :]
        A1 = np.asarray(to_world, dtype=np.float64)[:, None, :]
        gap2 = _segment_min_gap_sq_np(A0, A1, B0, B1)     # (K, C)
        with np.errstate(invalid="ignore"):
            conflict = (gap2 < thr).any(axis=1)
        return ~conflict


_OFFSETS_27_NP = np.array(
    [
        (di, dj, dk)
        for di in (-1, 0, 1)
        for dj in (-1, 0, 1)
        for dk in (-1, 0, 1)
        if not (di == 0 and dj == 0 and dk == 0)
    ]
    + [(0, 0, 0)],
    dtype=np.int64,
)
_WEIGHTS_27_NP = np.sqrt((_OFFSETS_27_NP.astype(np.float64) ** 2).sum(axis=1))


def cpu_space_time_astar(
    coarse_og,
    start_ijk,
    goal_ijk,
    t_offset: int,
    committed,
    time_step_cost: float,
    max_time_steps: int,
    hold_steps: int,
    max_expansions: int = 5_000_000,
    timeout_s: float = DEFAULT_CPU_TIMEOUT_S,
):
    """Sequential CPU space-time A*.

    Returns ``(path_ijk, path_t_abs, stats)``; ``path_ijk`` is None on failure or
    timeout, and ``stats`` always reports ``expansions``, ``elapsed_s``,
    ``path_cost`` and ``outcome`` in {"success", "no_path", "timeout",
    "expansion_cap"}.

    Optimality: with f_delta effectively 0 and an admissible heuristic, the first
    goal state popped that is also holdable is a minimum-cost holdable goal, so
    this doubles as the optimal-cost reference for the parallel-frontier runs.
    """
    t_start = time.perf_counter()
    grid = cp.asnumpy(coarse_og.grid).astype(bool)
    resolution = float(coarse_og.resolution)
    origin = cp.asnumpy(cp.asarray(coarse_og.origin)).astype(np.float64).reshape(3)
    Nx, Ny, Nz = grid.shape
    T_local = int(max_time_steps) if max_time_steps > 0 else max(1, int(committed.T) - t_offset)

    cc = _CpuCommitted(committed)

    sx, sy, sz = (int(start_ijk[0]), int(start_ijk[1]), int(start_ijk[2]))
    gx, gy, gz = (int(goal_ijk[0]), int(goal_ijk[1]), int(goal_ijk[2]))

    stats = {"expansions": 0, "elapsed_s": 0.0, "path_cost": float("nan"),
             "path_len": 0, "outcome": "no_path"}

    if grid[sx, sy, sz] or grid[gx, gy, gz]:
        stats["outcome"] = "blocked_endpoint"
        stats["elapsed_s"] = time.perf_counter() - t_start
        return None, None, stats
    if (sx, sy, sz) == (gx, gy, gz):
        stats.update(outcome="success", path_cost=0.0, path_len=1,
                     elapsed_s=time.perf_counter() - t_start)
        return (np.array([[sx, sy, sz]], dtype=np.intp),
                np.array([t_offset], dtype=np.intp), stats)

    def voxel_to_world(ijk):
        return np.asarray(ijk, dtype=np.float64) * resolution + origin + resolution * 0.5

    goal_world = voxel_to_world(np.array([gx, gy, gz]))
    # Same padding rationale as the GPU version: the holdability check uses the
    # cell centre but the true dwell point can be a half-diagonal off-centre.
    hold_pad = resolution * math.sqrt(3.0) / 2.0 + 0.02

    def h(x, y, z):
        dx, dy, dz = x - gx, y - gy, z - gz
        return (resolution * math.sqrt(dx * dx + dy * dy + dz * dz)
                + time_step_cost * max(abs(dx), abs(dy), abs(dz)))

    def goal_holdable(t_local: int) -> bool:
        if hold_steps <= 0 or cc.num_committed == 0:
            return True
        ts = (t_local + t_offset) + np.arange(hold_steps, dtype=np.intp)
        g = np.broadcast_to(goal_world, (hold_steps, 3))
        return bool(cc.moves_valid(g, g, ts, pad=hold_pad).all())

    stride_xyz = Nx * Ny * Nz
    stride_x = Ny * Nz

    def key(t, x, y, z):
        return int(t) * stride_xyz + int(x) * stride_x + int(y) * Nz + int(z)

    g_cost = {key(0, sx, sy, sz): 0.0}
    pred: dict[int, int] = {}
    closed: set[int] = set()
    open_heap = [(h(sx, sy, sz), 0.0, 0, sx, sy, sz)]

    check_every = 512
    expansions = 0
    found_t = None

    while open_heap:
        f, g, t, x, y, z = heapq.heappop(open_heap)
        k = key(t, x, y, z)
        if k in closed:
            continue
        # Stale heap entry (a better g was recorded after this was pushed).
        if g > g_cost.get(k, float("inf")) + 1e-12:
            continue
        closed.add(k)
        expansions += 1

        if (x, y, z) == (gx, gy, gz) and goal_holdable(t):
            found_t = t
            break

        if expansions % check_every == 0 and time.perf_counter() - t_start > timeout_s:
            stats.update(outcome="timeout", expansions=expansions,
                         elapsed_s=time.perf_counter() - t_start)
            return None, None, stats
        if expansions >= max_expansions:
            stats.update(outcome="expansion_cap", expansions=expansions,
                         elapsed_s=time.perf_counter() - t_start)
            return None, None, stats

        nt = t + 1
        if nt >= T_local:
            continue

        # All 27 successors of this one state, vectorised.
        nbr = _OFFSETS_27_NP + np.array([x, y, z], dtype=np.int64)
        nx_, ny_, nz_ = nbr[:, 0], nbr[:, 1], nbr[:, 2]
        ok = (nx_ >= 0) & (nx_ < Nx) & (ny_ >= 0) & (ny_ < Ny) & (nz_ >= 0) & (nz_ < Nz)
        if not ok.any():
            continue
        idx = np.where(ok)[0]
        nx_, ny_, nz_ = nx_[idx], ny_[idx], nz_[idx]
        w = _WEIGHTS_27_NP[idx]

        free = ~grid[nx_, ny_, nz_]
        if not free.any():
            continue
        idx2 = np.where(free)[0]
        nx_, ny_, nz_, w = nx_[idx2], ny_[idx2], nz_[idx2], w[idx2]

        from_world = np.broadcast_to(voxel_to_world(np.array([x, y, z])), (len(nx_), 3))
        to_world = voxel_to_world(np.stack([nx_, ny_, nz_], axis=1))
        parent_t = np.full(len(nx_), t + t_offset, dtype=np.intp)
        mv = cc.moves_valid(from_world, to_world, parent_t)
        if not mv.any():
            continue
        idx3 = np.where(mv)[0]
        nx_, ny_, nz_, w = nx_[idx3], ny_[idx3], nz_[idx3], w[idx3]

        new_g = g + w * resolution + time_step_cost
        for i in range(len(nx_)):
            cx, cy, cz = int(nx_[i]), int(ny_[i]), int(nz_[i])
            ck = key(nt, cx, cy, cz)
            if ck in closed:
                continue
            ng = float(new_g[i])
            if ng < g_cost.get(ck, float("inf")):
                g_cost[ck] = ng
                pred[ck] = k
                heapq.heappush(open_heap, (ng + h(cx, cy, cz), ng, nt, cx, cy, cz))

    if found_t is None:
        stats.update(outcome="no_path", expansions=expansions,
                     elapsed_s=time.perf_counter() - t_start)
        return None, None, stats

    # Backtrack.
    path = []
    k = key(found_t, gx, gy, gz)
    cost = g_cost[k]
    while True:
        t = k // stride_xyz
        rem = k % stride_xyz
        x = rem // stride_x
        rem2 = rem % stride_x
        path.append((x, rem2 // Nz, rem2 % Nz, t))
        if k not in pred:
            break
        k = pred[k]
    path.reverse()
    arr = np.array(path, dtype=np.intp)
    stats.update(outcome="success", expansions=expansions, path_cost=float(cost),
                 path_len=len(arr), elapsed_s=time.perf_counter() - t_start)
    return arr[:, :3], arr[:, 3] + t_offset, stats


# ── path cost, recomputed identically for every variant ────────────────────


def _path_cost(path_ijk, resolution: float, time_step_cost: float) -> float:
    """Cost of a returned path under the search's own cost model.

    Recomputed from the path rather than read out of any solver, so all three
    variants are scored by the same function and a difference cannot be an
    accounting artefact.
    """
    if path_ijk is None or len(path_ijk) < 2:
        return 0.0 if path_ijk is not None else float("nan")
    p = np.asarray(path_ijk, dtype=np.float64)
    d = np.abs(np.diff(p, axis=0))
    step_w = np.sqrt((d ** 2).sum(axis=1))
    return float((step_w * resolution + time_step_cost).sum())


# ── arm: legs ──────────────────────────────────────────────────────────────


def _make_leg_interceptor(records: list, max_legs: int, cpu_timeout: float,
                          run_cpu: bool):
    """Wrap space_time_astar_gpu so the first `max_legs` calls are solved three
    ways on identical inputs. Returns the production result either way, so the
    surrounding pipeline is unaffected and the legs stay realistic (each leg's
    CommittedMotion is whatever the priority loop had actually committed by
    then)."""

    def wrapper(coarse_og, start_ijk, goal_ijk, t_offset, committed, **kwargs):
        kwargs.pop("f_threshold_delta", None)
        resolution = float(coarse_og.resolution)
        time_step_cost = float(kwargs.get("time_step_cost", 0.01))
        hold_steps = int(kwargs.get("hold_steps", 0))

        # Production run always happens; it is what we return.
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        prod = space_time_astar_gpu(coarse_og, start_ijk, goal_ijk, t_offset,
                                    committed, f_threshold_delta=PRODUCTION_F_DELTA,
                                    **kwargs)
        cp.cuda.Stream.null.synchronize()
        t_pf = time.perf_counter() - t0

        if len(records) >= max_legs:
            return prod

        rec = {
            "leg_index": len(records),
            "t_offset": int(t_offset),
            "hold_steps": hold_steps,
            "num_committed": int(committed.num_committed),
            "max_time_steps": int(kwargs.get("max_time_steps", 0)),
            "grid_shape": [int(s) for s in coarse_og.shape],
            "resolution": resolution,
            "time_step_cost": time_step_cost,
            "straight_line_cells": float(
                np.linalg.norm(np.asarray([int(goal_ijk[i]) - int(start_ijk[i])
                                           for i in range(3)], dtype=np.float64))
            ),
        }

        prod_path = None if prod is None else cp.asnumpy(prod[0])
        rec["gpu_pf"] = {
            "time_s": t_pf,
            "success": prod is not None,
            "path_cost": _path_cost(prod_path, resolution, time_step_cost),
            "path_len": 0 if prod_path is None else int(len(prod_path)),
            "outcome": "success" if prod is not None else "no_path",
        }

        # Same code, same arrays, one f-layer per iteration.
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        seq = space_time_astar_gpu(coarse_og, start_ijk, goal_ijk, t_offset,
                                   committed, f_threshold_delta=0.0, **kwargs)
        cp.cuda.Stream.null.synchronize()
        t_seq = time.perf_counter() - t0
        seq_path = None if seq is None else cp.asnumpy(seq[0])
        rec["gpu_seq"] = {
            "time_s": t_seq,
            "success": seq is not None,
            "path_cost": _path_cost(seq_path, resolution, time_step_cost),
            "path_len": 0 if seq_path is None else int(len(seq_path)),
            "outcome": "success" if seq is not None else "no_path",
        }

        if run_cpu:
            cpu_path, _cpu_t, cpu_stats = cpu_space_time_astar(
                coarse_og, start_ijk, goal_ijk, t_offset, committed,
                time_step_cost=time_step_cost,
                max_time_steps=int(kwargs.get("max_time_steps", 0)),
                hold_steps=hold_steps,
                timeout_s=cpu_timeout,
            )
            rec["cpu_seq"] = {
                "time_s": cpu_stats["elapsed_s"],
                "success": cpu_stats["outcome"] == "success",
                "path_cost": _path_cost(cpu_path, resolution, time_step_cost)
                if cpu_path is not None else float("nan"),
                "path_len": int(cpu_stats["path_len"]),
                "expansions": int(cpu_stats["expansions"]),
                "outcome": cpu_stats["outcome"],
            }
        else:
            rec["cpu_seq"] = {"time_s": float("nan"), "success": False,
                              "path_cost": float("nan"), "path_len": 0,
                              "expansions": 0, "outcome": "skipped"}

        records.append(rec)
        logger.info(
            "  [leg %2d] committed=%d  gpu_pf=%.3fs  gpu_seq=%.3fs  cpu_seq=%.3fs (%s)"
            "  cost pf/seq=%s",
            rec["leg_index"], rec["num_committed"], t_pf, t_seq,
            rec["cpu_seq"]["time_s"], rec["cpu_seq"]["outcome"],
            _ratio_str(rec["gpu_pf"]["path_cost"], rec["gpu_seq"]["path_cost"]),
        )
        return prod

    return wrapper


def _ratio_str(a, b):
    if not (np.isfinite(a) and np.isfinite(b)) or b <= 0:
        return "n/a"
    return f"{a / b:.4f}"


def _full_routes(art):
    return [
        [art.home_indices[i]] + list(r) + [art.home_indices[i]]
        for i, r in enumerate(art.routes)
    ]


def _plan(art, n_trials: int):
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


def _run_legs(ctx, cfg, seed, max_legs, cpu_timeout, run_cpu):
    art = run_pipeline_to_routes(ctx, cfg, seed, fleet_size=DEFAULT_FLEET,
                                 vrp_alpha=VRP_ALPHA)
    row = {"model": cfg.name, "seed": seed, "arm": "legs",
           "num_viewpoints": art.num_viewpoints, "vrp_status": art.vrp_status,
           "legs": []}
    if not art.routes or not any(art.routes):
        row["vrp_status"] = "empty_routes"
        return row

    records: list = []
    original = _route_planner.space_time_astar_gpu
    _route_planner.space_time_astar_gpu = _make_leg_interceptor(
        records, max_legs, cpu_timeout, run_cpu)
    try:
        # One ordering: the corpus should be real legs from a real plan, and a
        # second ordering would only re-solve the same geometry.
        _plan(art, 1)
    finally:
        _route_planner.space_time_astar_gpu = original

    row["legs"] = records
    row["n_legs"] = len(records)
    return row


# ── arm: fdelta ────────────────────────────────────────────────────────────


def _make_fdelta_wrapper(f_delta: float, counter: dict):
    def wrapper(coarse_og, start_ijk, goal_ijk, t_offset, committed, **kwargs):
        counter["calls"] += 1
        kwargs.pop("f_threshold_delta", None)
        return space_time_astar_gpu(coarse_og, start_ijk, goal_ijk, t_offset,
                                    committed, f_threshold_delta=f_delta, **kwargs)
    return wrapper


def _objective(exec_result) -> float:
    return (VRP_ALPHA * exec_result.actual_makespan
            + (1.0 - VRP_ALPHA) * sum(exec_result.actual_per_vehicle_times))


def _run_fdelta(ctx, cfg, seed, deltas, n_trials):
    art = run_pipeline_to_routes(ctx, cfg, seed, fleet_size=DEFAULT_FLEET,
                                 vrp_alpha=VRP_ALPHA)
    row = {"model": cfg.name, "seed": seed, "arm": "fdelta",
           "num_viewpoints": art.num_viewpoints, "vrp_status": art.vrp_status,
           "n_priority_trials": n_trials, "per_delta": {}}
    if not art.routes or not any(art.routes):
        row["vrp_status"] = "empty_routes"
        return row

    original = _route_planner.space_time_astar_gpu
    try:
        for fd in deltas:
            counter = {"calls": 0}
            _route_planner.space_time_astar_gpu = _make_fdelta_wrapper(fd, counter)
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            ex = _plan(art, n_trials)
            cp.cuda.Stream.null.synchronize()
            mapf_s = time.perf_counter() - t0
            row["per_delta"][str(fd)] = {
                "mapf_wall_s": mapf_s,
                "astar_calls": counter["calls"],
                "objective_s": _objective(ex),
                "makespan_s": float(ex.actual_makespan),
                "total_time_s": float(sum(ex.actual_per_vehicle_times)),
                "astar_fail": int(sum(ex.fail_counts)),
                "true_collision_pairs": int(ex.true_collision_pairs),
                "true_min_separation_m": float(ex.true_min_separation),
            }
            logger.info("  [fdelta=%.2f] mapf=%.1fs obj=%.1fs mks=%.1fs "
                        "pairs=%d calls=%d",
                        fd, mapf_s, row["per_delta"][str(fd)]["objective_s"],
                        ex.actual_makespan, ex.true_collision_pairs, counter["calls"])
    finally:
        _route_planner.space_time_astar_gpu = original
    return row


# ── summary ────────────────────────────────────────────────────────────────


def _agg(vals):
    vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return float("nan"), float("nan"), 0
    if len(vals) == 1:
        return vals[0], 0.0, 1
    return float(np.mean(vals)), float(np.std(vals, ddof=1)), len(vals)


def _print_summary(results):
    legs_rows = [r for r in results if r.get("arm") == "legs" and r.get("legs")]
    if legs_rows:
        legs = [lg for r in legs_rows for lg in r["legs"]]
        logger.info("\n%s\nE17 arm=legs : %d intercepted legs over %d seeds\n%s",
                    "=" * 68, len(legs), len(legs_rows), "=" * 68)

        both = [lg for lg in legs if lg["gpu_pf"]["success"] and lg["gpu_seq"]["success"]]
        cpu_ok = [lg for lg in legs if lg["gpu_pf"]["success"]
                  and lg["cpu_seq"]["outcome"] == "success"]
        cpu_to = [lg for lg in legs if lg["cpu_seq"]["outcome"] == "timeout"]

        logger.info("%-26s %12s %12s %8s", "variant", "mean time (s)", "std", "n")
        for name in ("gpu_pf", "gpu_seq", "cpu_seq"):
            src = legs if name != "cpu_seq" else cpu_ok
            m, sd, n = _agg([lg[name]["time_s"] for lg in src
                             if lg[name].get("success")])
            logger.info("%-26s %12.4f %12.4f %8d", name, m, sd, n)

        if both:
            sp = [lg["gpu_seq"]["time_s"] / lg["gpu_pf"]["time_s"]
                  for lg in both if lg["gpu_pf"]["time_s"] > 0]
            m, sd, n = _agg(sp)
            logger.info("\nparallel-frontier speedup (gpu_seq / gpu_pf): "
                        "%.2fx +- %.2f  (n=%d)", m, sd, n)
            ci = [lg["gpu_pf"]["path_cost"] / lg["gpu_seq"]["path_cost"]
                  for lg in both if lg["gpu_seq"]["path_cost"] > 0]
            m, sd, n = _agg(ci)
            logger.info("realised cost inflation of f_delta=%.1f "
                        "(cost_pf / cost_seq): %.4f +- %.4f  (n=%d)",
                        PRODUCTION_F_DELTA, m, sd, n)
            worst = max(ci) if ci else float("nan")
            logger.info("  worst single leg: %.4f", worst)

        if cpu_ok:
            sp = [lg["cpu_seq"]["time_s"] / lg["gpu_pf"]["time_s"]
                  for lg in cpu_ok if lg["gpu_pf"]["time_s"] > 0]
            m, sd, n = _agg(sp)
            logger.info("\nGPU speedup (cpu_seq / gpu_pf): %.1fx +- %.1f  (n=%d "
                        "legs the CPU solved within its budget)", m, sd, n)
            agree = [lg for lg in cpu_ok
                     if np.isfinite(lg["gpu_seq"]["path_cost"])
                     and abs(lg["cpu_seq"]["path_cost"]
                             - lg["gpu_seq"]["path_cost"]) <= 1e-4]
            logger.info("  cpu_seq and gpu_seq agree on cost for %d/%d legs "
                        "(cross-validates both as the optimal reference)",
                        len(agree), len(cpu_ok))
        if cpu_to:
            logger.info("  %d/%d legs timed out on CPU -> the speedup above is a "
                        "LOWER bound", len(cpu_to), len(legs))

    fd_rows = [r for r in results if r.get("arm") == "fdelta" and r.get("per_delta")]
    if fd_rows:
        deltas = sorted({float(k) for r in fd_rows for k in r["per_delta"]})
        logger.info("\n%s\nE17 arm=fdelta : end-to-end effect of the relaxation "
                    "(%d seeds)\n%s", "=" * 68, len(fd_rows), "=" * 68)
        logger.info("%-10s %14s %14s %14s %10s %8s",
                    "f_delta", "mapf wall (s)", "objective (s)", "makespan (s)",
                    "collisions", "fails")
        for fd in deltas:
            pd = [r["per_delta"][str(fd)] for r in fd_rows if str(fd) in r["per_delta"]]
            wm, wsd, _ = _agg([x["mapf_wall_s"] for x in pd])
            om, osd, _ = _agg([x["objective_s"] for x in pd])
            mm, msd, _ = _agg([x["makespan_s"] for x in pd])
            cm, _, _ = _agg([x["true_collision_pairs"] for x in pd])
            fm, _, _ = _agg([x["astar_fail"] for x in pd])
            logger.info("%-10.2f %6.1f+-%-6.1f %6.1f+-%-6.1f %6.1f+-%-6.1f "
                        "%10.2f %8.1f", fd, wm, wsd, om, osd, mm, msd, cm, fm)
        logger.info("f_delta=0 is the optimal-expansion reference; the production "
                    "default is %.1f.", PRODUCTION_F_DELTA)


# ── main ───────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(
        description="E17: ablation of the 4D GPU space-time A* (R4.4, R1, R2)")
    p.add_argument("--arm", choices=["legs", "fdelta"], default="legs")
    p.add_argument("--models", nargs="+", default=["duke_of_lancaster"])
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_3)
    p.add_argument("--max_legs", type=int, default=DEFAULT_MAX_LEGS,
                   help="legs intercepted per seed in arm=legs")
    p.add_argument("--cpu_timeout", type=float, default=DEFAULT_CPU_TIMEOUT_S,
                   help="per-leg budget for the sequential CPU reference (s)")
    p.add_argument("--no_cpu", action="store_true",
                   help="skip the CPU reference (GPU parallel-vs-sequential only)")
    p.add_argument("--deltas", type=float, nargs="+", default=FDELTA_GRID)
    p.add_argument("--trials", type=int, default=5,
                   help="priority-ordering budget in arm=fdelta (E12 says 5 suffices)")
    p.add_argument("--output_dir",
                   default=os.path.join(RESULTS_DIR, "e17_stastar_ablation"))
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
            stem = f"model={name}_seed={seed}_arm={args.arm}"
            rpath = os.path.join(raw_dir, stem)
            if args.resume and os.path.exists(rpath + ".json"):
                all_results.append(load_run_result(rpath))
                logger.info("[resume] skip %s", stem)
                continue
            logger.info("Running %s: %s seed=%d", args.arm, name, seed)
            try:
                if args.arm == "legs":
                    row = _run_legs(ctx, cfg, seed, args.max_legs,
                                    args.cpu_timeout, not args.no_cpu)
                else:
                    row = _run_fdelta(ctx, cfg, seed, args.deltas, args.trials)
                all_results.append(row)
                save_run_result(row, rpath)
            except Exception as exc:  # noqa: BLE001
                handle_row_exception(exc, stem, resume=args.resume)
            finally:
                free_gpu_memory()

    _print_summary(all_results)


if __name__ == "__main__":
    main()
