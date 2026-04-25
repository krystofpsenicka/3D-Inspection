"""Lower bound computations for set cover and VRP/MAPF.

These provide theoretical baselines to assess solution quality:
- ``matching_lb_set_cover``    — greedy anti-chain LB for partial set cover
- ``information_theoretic_lb`` — trivial ceil(points / max_single_cover) bound
- ``vrp_lb_meters``            — 2-factor LP LB for multi-depot VRP (meters)
- ``vrp_mapf_lb_seconds``      — VRP LB mapped to seconds for the post-MAPF
                                 objective (accounts for cruise speed + dwell)

Legacy, superseded helpers:
- ``held_karp_tsp_lb`` and ``fleet_tsp_lb`` — assignment-relaxation TSP bound.
  Kept for backward compatibility; do not use for multi-depot VRPs.
  Prefer ``vrp_lb_meters`` / ``vrp_mapf_lb_seconds`` instead.
"""

from __future__ import annotations

import logging
import math
from typing import Iterable

import numpy as np
from scipy.optimize import linprog

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Set cover
# ═══════════════════════════════════════════════════════════════════════════

def matching_lb_set_cover(V_binary: np.ndarray) -> int:
    """Greedy anti-chain (matching) lower bound for set cover.

    Builds a set S of target points that pairwise conflict — no single
    candidate viewpoint covers two points in S. Any IP solution that covers
    all of S must use at least ``|S|`` viewpoints, since each viewpoint
    covers at most one point in S.

    The combinatorial argument extends to partial cover: if the IP must
    cover ``n_required`` out of ``M`` points, it may skip at most
    ``M − n_required`` points, so it still covers at least
    ``K + n_required − M`` points from S, giving the partial-cover LB
    ``max(0, K + n_required − M)``. Compute ``K`` once here; callers can
    derive the partial LB for any target.

    Greedy construction: iterate points in ascending-degree order
    (hardest-to-cover first — they tend to produce larger conflict sets)
    and keep a point iff no already-kept point shares a viewpoint with it.

    Args:
        V_binary: (N_candidates, M_points) binary visibility matrix.

    Returns:
        ``K`` — size of the greedy anti-chain. A valid LB on the IP
        optimum for full coverage.
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
    """Trivial lower bound: ceil(points_to_cover / max_coverage_per_vp).

    Args:
        num_points: total surface points
        target_coverage: fraction to cover
        max_single_vp_coverage: max points any single viewpoint covers
    """
    points_needed = int(num_points * target_coverage)
    if max_single_vp_coverage <= 0:
        return points_needed
    return math.ceil(points_needed / max_single_vp_coverage)


# ═══════════════════════════════════════════════════════════════════════════
# VRP / MAPF (multi-depot, alpha-blended objective)
# ═══════════════════════════════════════════════════════════════════════════

def vrp_lb_meters(
    dist_matrix: np.ndarray,
    depot_indices: Iterable[int],
    k: int,
    alpha: float,
) -> dict:
    """Lower bounds (in meters) for the multi-depot VRP used in this pipeline.

    The VRP objective solved by cuOpt is
    ``alpha · makespan + (1-alpha) · total_cost`` where both terms are in
    meters (from the shortest-path distance matrix).

    Returns a dict with three keys, each a provably valid LB on the integer
    optimum:

    - ``total_cost_lb_m``  — LP of the 2-factor relaxation over the full
      matrix. Each waypoint has in-degree = 1 and out-degree = 1; each
      depot has in-degree ≤ k and out-degree ≤ k (since at most k vehicles
      leave/return to the depot pool). Self-loops forbidden.
    - ``makespan_lb_m``    — max of (i) the maximum over waypoints of
      ``2 · min_depot_distance`` (a round-trip LB on the robot that visits
      the farthest waypoint), and (ii) ``total_cost_lb_m / k``.
    - ``objective_lb_m``   — ``alpha · makespan_lb_m + (1-alpha) · total_cost_lb_m``.

    Args:
        dist_matrix:  (n, n) symmetric non-negative metric distance matrix,
                      in meters, over ``depots ∪ waypoints``.
        depot_indices: indices of the depot rows/columns in ``dist_matrix``.
        k: number of vehicles (should equal ``len(depot_indices)`` in this
           pipeline but we don't enforce it).
        alpha: blend coefficient in [0, 1]; matches what the VRP solver used.

    Returns:
        Dict with ``total_cost_lb_m``, ``makespan_lb_m``, ``objective_lb_m``.
        All zeros on LP failure.
    """
    D = np.asarray(dist_matrix, dtype=np.float64)
    n = D.shape[0]
    depot_set = set(int(i) for i in depot_indices)
    waypoints = [i for i in range(n) if i not in depot_set]
    depots = sorted(depot_set)

    if n < 2 or not waypoints or not depots:
        return {
            "total_cost_lb_m": 0.0,
            "makespan_lb_m": 0.0,
            "objective_lb_m": 0.0,
        }

    # ── Total-cost LB: 2-factor LP ────────────────────────────────────────
    # Variables x_{ij} for i != j; self-loops set to 0 via upper bounds.
    num_vars = n * n
    c = D.flatten()

    # Constraints:
    #   for each waypoint w:  Σ_j x_{wj} = 1  and  Σ_i x_{iw} = 1
    #   for each depot   d:   Σ_j x_{dj} ≤ k  and  Σ_i x_{id} ≤ k
    A_eq_rows, b_eq = [], []
    A_ub_rows, b_ub = [], []

    for w in waypoints:
        row_out = np.zeros(num_vars)
        row_out[w * n:(w + 1) * n] = 1.0
        A_eq_rows.append(row_out)
        b_eq.append(1.0)

        row_in = np.zeros(num_vars)
        row_in[np.arange(n) * n + w] = 1.0
        A_eq_rows.append(row_in)
        b_eq.append(1.0)

    for d in depots:
        row_out = np.zeros(num_vars)
        row_out[d * n:(d + 1) * n] = 1.0
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
    # Forbid self-loops
    for i in range(n):
        bounds[i * n + i] = (0.0, 0.0)

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

    # ── Makespan LB: max(round-trip, total/k) ────────────────────────────
    # round-trip: for each waypoint, 2 * min_{d∈depots} d(d, w)
    wp_idx = np.array(waypoints, dtype=int)
    dep_idx = np.array(depots, dtype=int)
    min_depot_dist = D[np.ix_(wp_idx, dep_idx)].min(axis=1)  # (|wp|,)
    round_trip_lb = float(2.0 * min_depot_dist.max()) if wp_idx.size > 0 else 0.0
    divided_lb = total_cost_lb / max(1, k)
    makespan_lb = max(round_trip_lb, divided_lb)

    # ── Blended objective LB ─────────────────────────────────────────────
    objective_lb = alpha * makespan_lb + (1.0 - alpha) * total_cost_lb

    return {
        "total_cost_lb_m": total_cost_lb,
        "makespan_lb_m": makespan_lb,
        "objective_lb_m": objective_lb,
    }


def vrp_mapf_lb_seconds(
    vrp_lb_m: dict,
    n_waypoints: int,
    k: int,
    alpha: float,
    cruise_speed: float,
    dwell_s: float,
) -> dict:
    """Map a meter-based VRP LB to a seconds-based LB on the post-MAPF result.

    MAPF only adds detours (collision-avoidance waits and re-routes) and
    per-waypoint dwell time; it cannot shorten the travel paths. So the
    VRP meters-based LB maps to seconds as:

    - ``total_time_lb_s = vrp_total_cost / cruise_speed + n_waypoints · dwell_s``
      Each waypoint is dwelled on exactly once by some robot; every dwell
      contributes to the sum of robot times.
    - ``makespan_time_lb_s = max(vrp_makespan / cruise_speed + dwell_s,
                                total_time_lb_s / k)``
      The slowest robot has at least its VRP travel time (in seconds), plus
      at least one dwell; separately, ``total / k`` lower-bounds the
      slowest robot when dwell load dominates.
    - ``objective_time_lb_s`` is the alpha blend of the two, matching the
      MAPF objective (``alpha · makespan + (1-alpha) · total``) expressed
      in seconds.

    Args:
        vrp_lb_m: dict returned by :func:`vrp_lb_meters`.
        n_waypoints: number of inspection waypoints (excluding depots).
        k: number of vehicles.
        alpha: blend coefficient (should match the VRP solver's alpha).
        cruise_speed: constant travel speed in m/s (AUV_CRUISE_SPEED).
        dwell_s: per-waypoint dwell time in seconds (SPACE_TIME_DWELL_S).

    Returns:
        Dict with ``total_time_lb_s``, ``makespan_time_lb_s``,
        ``objective_time_lb_s``.
    """
    total_cost_m = float(vrp_lb_m.get("total_cost_lb_m", 0.0))
    makespan_m = float(vrp_lb_m.get("makespan_lb_m", 0.0))

    total_time_lb_s = (total_cost_m / cruise_speed) + n_waypoints * dwell_s
    travel_per_robot_s = makespan_m / cruise_speed + (dwell_s if n_waypoints > 0 else 0.0)
    makespan_time_lb_s = max(travel_per_robot_s, total_time_lb_s / max(1, k))
    objective_time_lb_s = alpha * makespan_time_lb_s + (1.0 - alpha) * total_time_lb_s

    return {
        "total_time_lb_s": total_time_lb_s,
        "makespan_time_lb_s": makespan_time_lb_s,
        "objective_time_lb_s": objective_time_lb_s,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Joint-problem LB (full pipeline, eq. (1.1) of thesis Ch. 1)
# ═══════════════════════════════════════════════════════════════════════════

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
    """Lower bound on the joint planning problem of thesis eq. (1.1).

    The joint problem, given mesh M, target points P, fleet k, coverage
    α and blend β, minimises ``β · max_r T_r + (1-β) · Σ_r T_r`` subject
    to ``Coverage(π₁,…,π_k) ≥ α`` plus free-space and pairwise-collision
    constraints. All T_r are trajectory durations in seconds.

    We compose three independently-valid LBs and combine them:

    1. **Dwell count**: ``N_poses_LB`` — lower bound on the number of
       distinct inspection poses any feasible fleet must visit. Computed
       as ``max(matching_lb_set_cover(V) − (1-α)·|P|,
                information_theoretic_lb(…))``  over the sampled
       candidate set. The matching bound subtracts the partial-cover
       skip budget (an IP solution may omit up to ``(1-α)·|P|`` points
       of the anti-chain). Rigorous on the sampled discrete set-cover
       instance; a proxy for the continuous joint problem under the
       (empirical) assumption that the candidate set is dense enough
       that refining it does not shrink the matching antichain.

    2. **Travel**: a rigorous depot-to-farthest-mandatory round-trip LB.
       For each target p, the robot covering it must get within
       ``frustum_far`` of p; the approach distance
       ``a(p) = max(0, d_euclidean(nearest_depot, p) − frustum_far)``
       is a valid per-point lower bound on single-leg travel because the
       geodesic (free-space) distance is ≥ Euclidean, and the feasible
       camera positions form a ball of radius ``frustum_far`` around p.
       To cover ≥ ⌈αM⌉ points the adversary skips the ⌈(1-α)M⌉ farthest
       ones; the robot covering the ⌈αM⌉-th-easiest mandatory point
       does a round-trip of ≥ 2·a(p). That single round-trip
       contributes to both some robot's makespan *and* to the fleet
       total. No geodesic distance matrix is needed — this is pure
       NumPy on (k × M) vectors.

    3. **Dwell time**: each of the ``N_poses_LB`` inspection poses
       consumes ``dwell_s`` seconds; total dwell is
       ``N_poses_LB · dwell_s``, split at best ``/k`` across the fleet.

    The three combine to:

    - ``joint_total_time_lb_s = N_poses_LB · dwell_s + travel_lb_s``
    - ``joint_makespan_lb_s  = max(N_poses_LB · dwell_s / k,
                                   travel_lb_s + dwell_s,
                                   joint_total_time_lb_s / k)``
    - ``joint_objective_lb_s = β · makespan_lb + (1-β) · total_time_lb``

    **Not implemented (future, tighter)**: geodesic distances via the
    inflated OG (would require snapping target points to nearest free
    voxels) and a covering-Steiner-tree on depots ∪ mandatory targets
    for a much tighter total-travel LB.

    Args:
        V_sampled: (N_candidates, M) binary visibility matrix over the
            pipeline's sampled candidate viewpoints.
        target_points: (M, 3) world-frame target points.
        depot_positions: (k, 3) world-frame depot positions.
        alpha_coverage: required coverage fraction in (0, 1].
        beta_blend: objective blend β ∈ [0, 1] (1 = pure makespan).
        frustum_far: camera frustum far range in metres.
        cruise_speed: constant cruise speed in m/s.
        dwell_s: per-pose dwell time in seconds.
        fleet_size: number of vehicles k.

    Returns:
        Dict with eight entries (all seconds except
        ``joint_n_poses_lb`` and ``joint_info_n_poses_lb`` which are
        integer counts)::

            joint_n_poses_lb, joint_info_n_poses_lb,
            joint_dwell_total_lb_s, joint_travel_makespan_lb_s,
            joint_travel_total_lb_s, joint_total_time_lb_s,
            joint_makespan_lb_s, joint_objective_lb_s
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

    # Component 1 — number of inspection poses.
    V_bin = V_sampled.astype(np.bool_, copy=False)
    max_single = int(V_bin.sum(axis=1).max()) if V_bin.size else 0
    n_info = information_theoretic_lb(M, alpha_coverage, max_single)
    try:
        # matching_lb_set_cover returns the full-cover LB K. Adjust to a
        # partial-cover LB: the IP may skip at most (1-α)·M points, so it
        # covers ≥ K + n_required − M points of the anti-chain and
        # therefore uses ≥ max(0, K − (1-α)·M) viewpoints.
        n_matching_full = matching_lb_set_cover(V_bin)
        skip_budget = int(round((1.0 - alpha_coverage) * M))
        n_matching = max(0, n_matching_full - skip_budget)
    except Exception as exc:
        logger.warning("matching_lb_set_cover failed in joint LB: %s", exc)
        n_matching = 0
    n_poses_lb = int(max(n_info, n_matching))
    out["joint_info_n_poses_lb"] = int(n_info)
    out["joint_n_poses_lb"] = n_poses_lb

    # Component 2 — rigorous travel LB via Euclidean distances.
    # a(p) = max(0, min_d d_euc(depot_d, p) − frustum_far) is a valid
    # lower bound on the closest free-space camera position for target p.
    depots = np.asarray(depot_positions, dtype=np.float64)
    tgt = np.asarray(target_points, dtype=np.float64)
    # (k, M) pairwise Euclidean distances.
    diff = depots[:, None, :] - tgt[None, :, :]
    d_dep_tgt = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    min_to_depot = d_dep_tgt.min(axis=0)  # (M,)
    approach = np.maximum(0.0, min_to_depot - float(frustum_far))
    # Mandatory count: any feasible solution must cover ≥ ⌈α·M⌉ targets.
    # The adversary skips the (M − n_min) hardest; the farthest mandatory
    # approach distance is the n_min-th smallest value in ``approach``.
    n_min = max(1, math.ceil(alpha_coverage * M))
    n_min = min(n_min, M)
    worst_required = float(np.partition(approach, n_min - 1)[n_min - 1])
    travel_mks_lb_s = 2.0 * worst_required / max(cruise_speed, 1e-9)

    out["joint_travel_makespan_lb_s"] = travel_mks_lb_s
    out["joint_travel_total_lb_s"] = travel_mks_lb_s

    # Component 3 — dwell.
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


# ═══════════════════════════════════════════════════════════════════════════
# Legacy TSP bounds (retained for backward compatibility — DO NOT USE)
# ═══════════════════════════════════════════════════════════════════════════

def held_karp_tsp_lb(distance_matrix: np.ndarray) -> float:
    """[Legacy] Assignment-LP relaxation lower bound for TSP.

    .. deprecated::
        Not tight enough for multi-depot VRP. Use :func:`vrp_lb_meters`.
    """
    n = distance_matrix.shape[0]
    if n <= 2:
        if n == 2:
            return float(distance_matrix[0, 1] + distance_matrix[1, 0])
        return 0.0

    num_vars = n * n
    c = distance_matrix.flatten().astype(np.float64)
    for i in range(n):
        c[i * n + i] = 1e12

    A_eq_rows = np.zeros((n, num_vars))
    for i in range(n):
        A_eq_rows[i, i * n:(i + 1) * n] = 1.0
    b_eq_rows = np.ones(n)

    A_eq_cols = np.zeros((n, num_vars))
    for j in range(n):
        for i in range(n):
            A_eq_cols[j, i * n + j] = 1.0
    b_eq_cols = np.ones(n)

    A_eq = np.vstack([A_eq_rows, A_eq_cols])
    b_eq = np.concatenate([b_eq_rows, b_eq_cols])
    bounds = [(0, 1)] * num_vars

    try:
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if result.success:
            logger.info("Held-Karp LP bound: %.2f", result.fun)
            return float(result.fun)
        logger.warning("Held-Karp LP failed: %s", result.message)
        return 0.0
    except Exception as e:
        logger.warning("Held-Karp LP error: %s", e)
        return 0.0


def fleet_tsp_lb(distance_matrix: np.ndarray, num_vehicles: int) -> float:
    """[Legacy] Held-Karp TSP bound / k. Use :func:`vrp_lb_meters` instead."""
    tsp_lb = held_karp_tsp_lb(distance_matrix)
    return tsp_lb / max(1, num_vehicles)
