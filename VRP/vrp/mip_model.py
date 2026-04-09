"""Combined-objective VRP MIP formulation.

Minimises ``alpha * T/T_norm + (1-alpha) * total_cost/C_norm`` where T is
the makespan and total_cost is the sum of all arc costs. Uses lifted MTZ
subtour elimination (Desrochers & Laporte, 1991) and variable reduction
(Lalla-Ruiz & Mes, 2021).

Arc validity is computed on GPU via CuPy boolean masks, then PuLP
variables are created only for valid arcs.

References:
    Desrochers, M. & Laporte, G. (1991). Improvements and Extensions to
        the Miller-Tucker-Zemlin Subtour Elimination Constraints.
        Operations Research Letters.
    Lalla-Ruiz, E. & Mes, M.R.K. (2021). Mathematical Formulations and
        Improvements for the Multi-Depot Open Vehicle Routing Problem.
        Optimization Letters.
    Toth, P. & Vigo, D. (2014). Vehicle Routing: Problems, Methods, and
        Applications. MOS-SIAM Series on Optimization.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import cupy as cp
import numpy as np

from ._helpers import per_vehicle_costs as _per_vehicle_costs

logger = logging.getLogger(__name__)


# ─── GPU-accelerated arc validity ───────────────────────────────────────────

def _compute_valid_arcs(
    n: int,
    K: int,
    depots: list[int],
    depot_set: set[int],
    cost: np.ndarray,
    alpha: float,
    T_ub: float,
) -> np.ndarray:
    """Compute valid (i, j, v) arc tuples via GPU boolean masks.

    Returns an (M, 3) int32 NumPy array where each row is (i, j, v).
    """
    is_depot = cp.zeros(n, dtype=cp.bool_)
    depot_arr = cp.array(list(depot_set), dtype=cp.intp)
    is_depot[depot_arr] = True

    depots_gpu = cp.array(depots, dtype=cp.intp)
    i_idx = cp.arange(n, dtype=cp.intp)

    no_self_loop = i_idx[:, None] != i_idx[None, :]
    not_depot_depot = ~(is_depot[:, None] & is_depot[None, :])

    cost_gpu = cp.asarray(cost)
    if alpha >= 1.0 and T_ub < float("inf"):
        within_budget = cost_gpu <= T_ub
    else:
        within_budget = cp.ones((n, n), dtype=cp.bool_)

    base_mask = no_self_loop & not_depot_depot & within_budget

    i_allowed = (~is_depot[None, :]) | (i_idx[None, :] == depots_gpu[:, None])
    j_allowed = (~is_depot[None, :]) | (i_idx[None, :] == depots_gpu[:, None])

    full_mask = i_allowed[:, :, None] & j_allowed[:, None, :] & base_mask[None, :, :]

    v_indices, i_indices, j_indices = cp.where(full_mask)
    valid_arcs = cp.stack([i_indices, j_indices, v_indices], axis=1)
    return cp.asnumpy(valid_arcs).astype(np.int32)


# ─── MIP model builder ─────────────────────────────────────────────────────

def build_vrp_mip(
    dist_matrix: np.ndarray,
    num_vehicles: int,
    depots: list[int],
    capacity: int,
    alpha: float = 1.0,
    warm_start_routes: Optional[List[List[int]]] = None,
    mip_gap: float = 0.05,
) -> Tuple:
    """Build the combined-objective VRP MIP.

    Parameters
    ----------
    dist_matrix : (N, N) numpy float
        Full distance matrix. Numpy because PuLP needs CPU-side
        coefficient access.
    num_vehicles : int
        Number of vehicles (K).
    depots : list[int]
        Per-vehicle depot indices.
    capacity : int
        Max customers per vehicle (Q).
    alpha : float
        Objective blending parameter in [0, 1].
    warm_start_routes : list[list[int]], optional
        Feasible solution for warm-starting.
    mip_gap : float
        Relative optimality gap (logging only).

    Returns
    -------
    (prob, warm_start_dict)
        PuLP LpProblem and optional warm-start variable values dict.
    """
    try:
        import pulp
    except ImportError:
        raise ImportError(
            "PuLP is required for the VRP MIP solver. "
            "Install with: pip install pulp"
        )

    n = dist_matrix.shape[0]
    depot_set = set(depots)
    customers = [i for i in range(n) if i not in depot_set]
    n_c = len(customers)
    K = num_vehicles

    logger.info("[MIP] Building model: %d nodes (%d customers, %d depots), "
                "%d vehicles, capacity=%d",
                n, n_c, len(depot_set), K, capacity)

    cost = dist_matrix.astype(np.float64).copy()

    # ── Bound tightening ─────────────────────────────────────────────
    T_lb = 0.0
    for v in range(K):
        dv = depots[v]
        best_rt = min(
            (float(cost[dv, j] + cost[j, dv]) for j in customers),
            default=0.0,
        )
        T_lb = max(T_lb, best_rt)

    T_ub = float("inf")
    if warm_start_routes:
        per_v = _per_vehicle_costs(warm_start_routes, cp.asarray(cost), depots)
        T_ub = max(per_v) if per_v else float("inf")
        logger.info("[MIP] Warm-start makespan upper bound: %.2f", T_ub)

    logger.info("[MIP] T bounds: [%.2f, %s]",
                T_lb, f"{T_ub:.2f}" if T_ub < float("inf") else "inf")

    # ── GPU-vectorized arc validity ──────────────────────────────────
    valid_arcs = _compute_valid_arcs(n, K, depots, depot_set, cost, alpha, T_ub)
    logger.info("[MIP] Valid arcs: %d (of %d possible)",
                len(valid_arcs), n * (n - 1) * K)

    # ── Model ────────────────────────────────────────────────────────
    prob = pulp.LpProblem("MinMaxVRP", pulp.LpMinimize)

    x: Dict[Tuple[int, int, int], pulp.LpVariable] = {}
    for row in valid_arcs:
        i, j, v = int(row[0]), int(row[1]), int(row[2])
        x[i, j, v] = pulp.LpVariable(f"x_{i}_{j}_{v}", cat=pulp.LpBinary)

    u: Dict[int, pulp.LpVariable] = {}
    for i in customers:
        u[i] = pulp.LpVariable(f"u_{i}", lowBound=1, upBound=n_c,
                                cat=pulp.LpContinuous)

    T = pulp.LpVariable("T", lowBound=T_lb, cat=pulp.LpContinuous)
    if T_ub < float("inf"):
        T.upBound = T_ub * 1.01

    # ── Objective ────────────────────────────────────────────────────
    total_cost_expr = pulp.lpSum(
        cost[i, j] * x[i, j, v] for (i, j, v) in x
    )
    if alpha >= 1.0:
        prob += T, "Makespan"
    elif alpha <= 0.0:
        prob += total_cost_expr, "TotalCost"
    else:
        T_norm = T_lb if T_lb > 0 else 1.0
        C_norm = K * T_lb if T_lb > 0 else 1.0
        prob += (
            alpha * (T / T_norm)
            + (1 - alpha) * (total_cost_expr / C_norm),
            "Combined",
        )

    for i in customers:
        prob += (
            pulp.lpSum(
                x[i, j, v]
                for v in range(K) for j in range(n)
                if (i, j, v) in x
            ) == 1,
            f"visit_{i}",
        )

    for v in range(K):
        for i in customers:
            prob += (
                pulp.lpSum(x[j, i, v] for j in range(n) if (j, i, v) in x)
                ==
                pulp.lpSum(x[i, j, v] for j in range(n) if (i, j, v) in x),
                f"flow_{v}_{i}",
            )

    for v in range(K):
        dv = depots[v]
        prob += (
            pulp.lpSum(x[dv, j, v] for j in customers if (dv, j, v) in x)
            <= 1,
            f"depot_depart_{v}",
        )

    for v in range(K):
        dv = depots[v]
        prob += (
            pulp.lpSum(x[dv, j, v] for j in customers if (dv, j, v) in x)
            ==
            pulp.lpSum(x[j, dv, v] for j in customers if (j, dv, v) in x),
            f"depot_balance_{v}",
        )

    for v in range(K):
        prob += (
            pulp.lpSum(
                cost[i, j] * x[i, j, v]
                for (i, j, vv) in x if vv == v
            ) <= T,
            f"makespan_{v}",
        )

    for i in customers:
        for j in customers:
            if i == j:
                continue
            for v in range(K):
                fwd = x.get((i, j, v))
                bwd = x.get((j, i, v))
                if fwd is None:
                    continue
                terms = u[i] - u[j] + n_c * fwd
                if bwd is not None:
                    terms += (n_c - 2) * bwd
                prob += (terms <= n_c - 1, f"mtz_{i}_{j}_{v}")

    for v in range(K):
        prob += (
            pulp.lpSum(
                x[i, j, v]
                for i in customers for j in range(n)
                if (i, j, v) in x
            ) <= capacity,
            f"capacity_{v}",
        )

    n_vars = len(x) + len(u) + 1
    n_cons = len(prob.constraints)
    logger.info("[MIP] Model: %d variables (%d binary), %d constraints",
                n_vars, len(x), n_cons)

    warm_start = None
    if warm_start_routes:
        warm_start = _build_warm_start(
            warm_start_routes, depots, customers, n, K, x, u, T, cost,
        )

    return prob, warm_start


def _build_warm_start(
    routes: List[List[int]],
    depots: List[int],
    customers: List[int],
    n: int,
    K: int,
    x: Dict,
    u: Dict,
    T_var,
    cost: np.ndarray,
) -> Dict[str, float]:
    """Convert a feasible route solution into variable assignments."""
    vals: Dict[str, float] = {}

    for key, var in x.items():
        vals[var.name] = 0.0

    for v, route in enumerate(routes):
        if not route:
            continue
        dv = depots[v]
        full = [dv] + list(route) + [dv]
        for a, b in zip(full[:-1], full[1:]):
            key = (a, b, v)
            if key in x:
                vals[x[key].name] = 1.0

    customer_set = set(customers)
    order_counter = 1
    for v, route in enumerate(routes):
        for pos, node in enumerate(route):
            if node in customer_set and node in u:
                vals[u[node].name] = float(order_counter)
                order_counter += 1

    per_v = _per_vehicle_costs(routes, cp.asarray(cost), depots)
    vals[T_var.name] = max(per_v) if per_v else 0.0

    return vals


# ─── Route extraction from solved PuLP model ───────────────────────────────

def extract_routes(
    prob,
    num_vehicles: int,
    depots: List[int],
    n: int,
) -> List[List[int]]:
    """Extract per-vehicle routes from solved PuLP variables.

    Parses active x variables (CPU string ops), then builds a (K, N)
    next-node matrix on GPU and traces all vehicles in parallel.
    """
    depot_set = set(depots)

    # Parse active arcs from PuLP variable names
    active_i, active_j, active_v = [], [], []
    for var in prob.variables():
        if var.name.startswith("x_") and var.varValue is not None and var.varValue > 0.5:
            parts = var.name.split("_")
            active_i.append(int(parts[1]))
            active_j.append(int(parts[2]))
            active_v.append(int(parts[3]))

    if not active_i:
        return [[] for _ in range(num_vehicles)]

    K = num_vehicles
    next_node = cp.full((K, n), -1, dtype=cp.int32)
    i_gpu = cp.array(active_i, dtype=cp.int32)
    j_gpu = cp.array(active_j, dtype=cp.int32)
    v_gpu = cp.array(active_v, dtype=cp.int32)
    next_node[v_gpu, i_gpu] = j_gpu

    is_depot = cp.zeros(n, dtype=cp.bool_)
    for d in depot_set:
        is_depot[d] = True

    depots_gpu = cp.array(depots, dtype=cp.int32)
    vehicle_range = cp.arange(K, dtype=cp.int32)
    current = next_node[vehicle_range, depots_gpu]
    active = (current >= 0) & ~is_depot[current]

    max_route_len = n
    route_matrix = cp.full((K, max_route_len), -1, dtype=cp.int32)

    for step in range(max_route_len):
        if not active.any():
            break
        route_matrix[active, step] = current[active]
        next_step = cp.full(K, -1, dtype=cp.int32)
        next_step[active] = next_node[vehicle_range[active], current[active]]
        current = next_step
        active = (current >= 0) & ~is_depot[current]

    route_matrix_cpu = cp.asnumpy(route_matrix)
    routes: List[List[int]] = []
    for v in range(K):
        row = route_matrix_cpu[v]
        route = [int(node) for node in row if node >= 0]
        routes.append(route)

    return routes
