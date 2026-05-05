"""Lower bounds for set cover and VRP/MAPF.

- matching_lb_set_cover     -- greedy anti-chain LB for partial set cover
- information_theoretic_lb  -- ceil(points / max_single_cover)
- set_cover_lp_lb           -- LP relaxation LB for partial set cover
- vrp_lb_meters             -- 2-factor LP LB for multi-depot VRP (meters)
- vrp_mapf_lb_seconds       -- VRP LB mapped to seconds for post-MAPF objective
- joint_problem_lb          -- LB on the joint problem of thesis

"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable

import numpy as np
from scipy.optimize import linprog

logger = logging.getLogger(__name__)


# ── Set cover ──────────────────────────────────────────────────────────────


def matching_lb_set_cover(V_binary: np.ndarray) -> int:
    """Greedy anti-chain (matching) LB for set cover.

    Builds S of pairwise-conflicting target points (no single viewpoint covers
    two of them). Any IP covering all of S uses >= |S| viewpoints.

    Partial-cover extension: covering n_required of M means skipping at most
    M − n_required, so the IP still uses >= max(0, K + n_required − M)
    viewpoints. Compute K once; callers derive partial LBs.

    Greedy: iterate ascending degree (hardest first), keep a point iff no
    already-kept point shares a viewpoint.

    Returns K = anti-chain size.
    """
    V = V_binary.astype(np.bool_, copy=False)
    N, M = V.shape
    deg = V.sum(axis=0)
    order = np.argsort(deg, kind="stable")
    used = np.zeros(N, dtype=np.bool_)
    K = 0
    for j in order:
        col = V[:, j]
        if not col.any():
            continue
        if not (col & used).any():
            used |= col
            K += 1
    logger.info("Matching LB set cover: N=%d M=%d K=%d", N, M, K)
    return K


def information_theoretic_lb(
    num_points: int,
    target_coverage: float,
    max_single_vp_coverage: int,
) -> int:
    """Trivial LB: ceil(points_to_cover / max_coverage_per_vp)."""
    points_needed = int(num_points * target_coverage)
    if max_single_vp_coverage <= 0:
        return points_needed
    return math.ceil(points_needed / max_single_vp_coverage)


def set_cover_lp_lb(V_bool_sparse, n_required: int) -> int:
    """LP relaxation LB for partial set cover.

        min  Σ x_i
        s.t. Σ_i V[i,j] x_i >= z_j    ∀j
             Σ_j z_j >= n_required
             x_i >= 0,  0 <= z_j <= 1

    ⌈LP_opt⌉ is a valid LB on the IP (subsumes antichain + info-theoretic).

    V_bool_sparse: (N, M) scipy.sparse (any format; converted to CSC).
    Returns ⌈LP_opt⌉ or 0 on solver failure.
    """
    from scipy.sparse import csc_matrix, eye, hstack
    from scipy.sparse import vstack as svstack

    V_csc = csc_matrix(V_bool_sparse, dtype=np.float64)
    N, M = V_csc.shape
    if n_required <= 0:
        return 0

    # Variables: [x_1 .. x_N, z_1 .. z_M]
    c = np.zeros(N + M)
    c[:N] = 1.0

    # Linking: -V^T x + z <= 0  (M rows)
    A_link = hstack([-V_csc.T, eye(M, format="csc")], format="csc")

    # Target: -Σ z_j <= -n_required  (1 row)
    A_target = hstack(
        [csc_matrix((1, N)), -csc_matrix(np.ones((1, M)))],
        format="csc",
    )

    A_ub = svstack([A_link, A_target], format="csc")
    b_ub = np.zeros(M + 1)
    b_ub[M] = -float(n_required)

    bounds = [(0.0, None)] * N + [(0.0, 1.0)] * M

    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if result.success:
            return math.ceil(result.fun - 1e-9)
        logger.warning("Set cover LP failed: %s", result.message)
        return 0
    except Exception as e:
        logger.warning("Set cover LP error: %s", e)
        return 0


# ── VRP / MAPF (multi-depot, beta-blended) ─────────────────────────────────


def vrp_lb_meters(
    dist_matrix: np.ndarray,
    depot_indices: Iterable[int],
    k: int,
    beta: float,
) -> dict:
    """LBs (meters) for the multi-depot VRP. Objective: β·makespan + (1-β)·total.

    Returns dict with three valid LBs on the IP optimum:

    - total_cost_lb_m  -- LP of the 2-factor relaxation: each waypoint has
      in/out-degree 1, each depot has in/out-degree <= k, no self-loops.
    - makespan_lb_m    -- max(2·max_w(min_depot_dist(w)), total_cost_lb_m / k).
    - objective_lb_m   -- β · makespan_lb_m + (1-β) · total_cost_lb_m.

    dist_matrix: (n, n) symmetric non-negative metric over depots ∪ waypoints.
    All zeros on LP failure.
    """
    D = np.asarray(dist_matrix, dtype=np.float64)
    n = D.shape[0]
    depot_set = set(int(i) for i in depot_indices)
    waypoints = [i for i in range(n) if i not in depot_set]
    depots = sorted(depot_set)

    if n < 2 or not waypoints or not depots:
        return {"total_cost_lb_m": 0.0, "makespan_lb_m": 0.0, "objective_lb_m": 0.0}

    # ── Total-cost LB: 2-factor LP ────────────────────────────────────
    # Variables x_{ij} for i != j; self-loops set to 0 via upper bounds.
    num_vars = n * n
    c = D.flatten()

    # Constraints:
    #   waypoint w: Σ_j x_{wj} = 1, Σ_i x_{iw} = 1
    #   depot d:    Σ_j x_{dj} <= k, Σ_i x_{id} <= k
    A_eq_rows, b_eq = [], []
    A_ub_rows, b_ub = [], []

    for w in waypoints:
        row_out = np.zeros(num_vars)
        row_out[w * n : (w + 1) * n] = 1.0
        A_eq_rows.append(row_out)
        b_eq.append(1.0)

        row_in = np.zeros(num_vars)
        row_in[np.arange(n) * n + w] = 1.0
        A_eq_rows.append(row_in)
        b_eq.append(1.0)

    for d in depots:
        row_out = np.zeros(num_vars)
        row_out[d * n : (d + 1) * n] = 1.0
        A_ub_rows.append(row_out)
        b_ub.append(float(k))

        row_in = np.zeros(num_vars)
        row_in[np.arange(n) * n + d] = 1.0
        A_ub_rows.append(row_in)
        b_ub.append(float(k))

    A_eq = np.vstack(A_eq_rows) if A_eq_rows else None
    b_eq_arr = np.array(b_eq) if b_eq else None
    A_ub = np.vstack(A_ub_rows) if A_ub_rows else None
    b_ub_arr = np.array(b_ub) if b_ub else None

    bounds = [(0.0, 1.0)] * num_vars
    for i in range(n):
        bounds[i * n + i] = (0.0, 0.0)  # forbid self-loops

    try:
        result = linprog(
            c, A_ub=A_ub, b_ub=b_ub_arr, A_eq=A_eq, b_eq=b_eq_arr,
            bounds=bounds, method="highs",
        )
        if result.success:
            total_cost_lb = float(result.fun)
        else:
            logger.warning("VRP 2-factor LP failed: %s", result.message)
            total_cost_lb = 0.0
    except Exception as e:
        logger.warning("VRP 2-factor LP error: %s", e)
        total_cost_lb = 0.0

    # ── Makespan LB: max(round-trip, total/k) ─────────────────────────
    wp_idx = np.array(waypoints, dtype=int)
    dep_idx = np.array(depots, dtype=int)
    min_depot_dist = D[np.ix_(wp_idx, dep_idx)].min(axis=1)  # (|wp|,)
    round_trip_lb = float(2.0 * min_depot_dist.max()) if wp_idx.size > 0 else 0.0
    divided_lb = total_cost_lb / max(1, k)
    makespan_lb = max(round_trip_lb, divided_lb)

    objective_lb = beta * makespan_lb + (1.0 - beta) * total_cost_lb

    return {
        "total_cost_lb_m": total_cost_lb,
        "makespan_lb_m": makespan_lb,
        "objective_lb_m": objective_lb,
    }


def vrp_mapf_lb_seconds(
    vrp_lb_m: dict,
    n_waypoints: int,
    k: int,
    beta: float,
    cruise_speed: float,
    dwell_s: float,
) -> dict:
    """Map meter-based VRP LB to seconds-based LB on post-MAPF result.

    MAPF only adds detours (collision waits) and per-waypoint dwell:
    - total_time_lb_s = vrp_total_cost / cruise_speed + n_waypoints · dwell_s
    - makespan_time_lb_s = max(vrp_makespan / cruise_speed + dwell_s,
                                total_time_lb_s / k)
    - objective_time_lb_s = β · makespan + (1-β) · total
    """
    total_cost_m = float(vrp_lb_m.get("total_cost_lb_m", 0.0))
    makespan_m = float(vrp_lb_m.get("makespan_lb_m", 0.0))

    total_time_lb_s = (total_cost_m / cruise_speed) + n_waypoints * dwell_s
    travel_per_robot_s = makespan_m / cruise_speed + (dwell_s if n_waypoints > 0 else 0.0)
    makespan_time_lb_s = max(travel_per_robot_s, total_time_lb_s / max(1, k))
    objective_time_lb_s = beta * makespan_time_lb_s + (1.0 - beta) * total_time_lb_s

    return {
        "total_time_lb_s": total_time_lb_s,
        "makespan_time_lb_s": makespan_time_lb_s,
        "objective_time_lb_s": objective_time_lb_s,
    }


# ── Joint-problem LB (full pipeline, thesis eq. 1.1) ───────────────────────


def joint_problem_lb(
    V_sampled: np.ndarray,
    target_points: np.ndarray,
    depot_positions: np.ndarray,
    *,
    alpha_coverage: float,
    beta_blend: float,
    frustum_far: float,
    cruise_speed: float,
    dwell_s: float,
    fleet_size: int,
) -> dict:
    """LB on the joint problem of thesis eq. (1.1):
    min β·max_r T_r + (1-β)·Σ_r T_r  s.t. Coverage >= α + free-space + collision.

    Composes three independently valid LBs:

    1. **Dwell count** N_poses_LB =
       max(matching_lb_set_cover(V) − (1-α)·M, information_theoretic_lb).
       Matching is rigorous on the sampled set-cover instance; proxy for the
       continuous problem assuming the candidate set is dense enough.

    2. **Travel** -- depot-to-farthest-mandatory round-trip. For each target p,
       a(p) = max(0, d_euc(nearest_depot, p) − frustum_far) is a valid lower
       bound on single-leg travel (geodesic >= Euclidean, feasible camera
       positions form a ball of radius frustum_far). The adversary skips the
       ⌈(1-α)M⌉ farthest mandatory points; the robot covering the
       ⌈αM⌉-th-easiest does a round-trip of >= 2·a(p). NumPy on (k × M).

    3. **Dwell time** -- N_poses_LB · dwell_s, split at best /k across fleet.

    Combined:
    - joint_total_time_lb_s = N_poses_LB · dwell_s + travel_lb_s
    - joint_makespan_lb_s   = max(N_poses_LB · dwell_s / k,
                                  travel_lb_s + dwell_s,
                                  joint_total_time_lb_s / k)
    - joint_objective_lb_s  = β · makespan_lb + (1-β) · total_time_lb

    Future tighter LBs: geodesic distances via inflated OG; covering Steiner
    tree on depots ∪ mandatory targets.

    Returns dict with eight entries (seconds, except joint_n_poses_lb /
    joint_info_n_poses_lb which are integer counts).
    """
    out = {
        "joint_n_poses_lb": 0,
        "joint_info_n_poses_lb": 0,
        "joint_dwell_total_lb_s": 0.0,
        "joint_travel_makespan_lb_s": 0.0,
        "joint_travel_total_lb_s": 0.0,
        "joint_total_time_lb_s": 0.0,
        "joint_makespan_lb_s": 0.0,
        "joint_objective_lb_s": 0.0,
    }

    M = int(len(target_points))
    k = int(fleet_size)
    if M == 0 or k == 0 or not (0.0 < alpha_coverage <= 1.0):
        return out

    # 1. Number of inspection poses.
    V_bin = V_sampled.astype(np.bool_, copy=False)
    max_single = int(V_bin.sum(axis=1).max()) if V_bin.size else 0
    n_info = information_theoretic_lb(M, alpha_coverage, max_single)
    try:
        # matching returns full-cover K; adjust to partial: skip budget = (1-α)·M.
        n_matching_full = matching_lb_set_cover(V_bin)
        skip_budget = int(round((1.0 - alpha_coverage) * M))
        n_matching = max(0, n_matching_full - skip_budget)
    except Exception as exc:
        logger.warning("matching_lb_set_cover failed in joint LB: %s", exc)
        n_matching = 0
    n_poses_lb = int(max(n_info, n_matching))
    out["joint_info_n_poses_lb"] = int(n_info)
    out["joint_n_poses_lb"] = n_poses_lb

    # 2. Travel LB via Euclidean distances.
    depots = np.asarray(depot_positions, dtype=np.float64)
    tgt = np.asarray(target_points, dtype=np.float64)
    diff = depots[:, None, :] - tgt[None, :, :]
    d_dep_tgt = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))  # (k, M)
    min_to_depot = d_dep_tgt.min(axis=0)  # (M,)
    approach = np.maximum(0.0, min_to_depot - float(frustum_far))
    # Adversary skips (M − n_min) hardest; n_min-th smallest is worst mandatory.
    n_min = max(1, math.ceil(alpha_coverage * M))
    n_min = min(n_min, M)
    worst_required = float(np.partition(approach, n_min - 1)[n_min - 1])
    travel_mks_lb_s = 2.0 * worst_required / max(cruise_speed, 1e-9)

    out["joint_travel_makespan_lb_s"] = travel_mks_lb_s
    out["joint_travel_total_lb_s"] = travel_mks_lb_s

    # 3. Dwell.
    dwell_total = n_poses_lb * dwell_s
    out["joint_dwell_total_lb_s"] = dwell_total

    # Compose.
    total_time_lb = dwell_total + travel_mks_lb_s
    mks_lb = max(
        dwell_total / max(k, 1),
        (travel_mks_lb_s + dwell_s) if n_poses_lb > 0 else travel_mks_lb_s,
        total_time_lb / max(k, 1),
    )
    obj_lb = beta_blend * mks_lb + (1.0 - beta_blend) * total_time_lb

    out["joint_total_time_lb_s"] = float(total_time_lb)
    out["joint_makespan_lb_s"] = float(mks_lb)
    out["joint_objective_lb_s"] = float(obj_lb)

    logger.info(
        "Joint LB: N_poses=%d (info=%d, match=%d) dwell=%.1fs travel_mks=%.1fs "
        "→ makespan_lb=%.1fs total_lb=%.1fs obj_lb=%.1fs (β=%.2f)",
        n_poses_lb, n_info, n_matching, dwell_total, travel_mks_lb_s,
        mks_lb, total_time_lb, obj_lb, beta_blend,
    )
    return out


JOINT_LB_FIELDS = (
    "joint_n_poses_lb",
    "joint_info_n_poses_lb",
    "joint_dwell_total_lb_s",
    "joint_travel_makespan_lb_s",
    "joint_travel_total_lb_s",
    "joint_total_time_lb_s",
    "joint_makespan_lb_s",
    "joint_objective_lb_s",
)
