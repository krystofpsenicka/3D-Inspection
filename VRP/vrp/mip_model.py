"""Combined-objective VRP MIP: alpha * makespan + (1 - alpha) * total cost.

Lifted MTZ subtour elimination with variable reduction; arc validity is
filtered on GPU before PuLP variable creation. Formulation details are
in the Bachelor thesis.
"""

from __future__ import annotations

import logging

import cupy as cp
import numpy as np

from ..core.constants import MIP_GAP, VRP_ALPHA
from ._helpers import per_vehicle_costs as _per_vehicle_costs

logger = logging.getLogger(__name__)


# ─── arc validity ───────────────────────────────────────────


def _compute_valid_arcs(
    n: int,
    K: int,
    depots: list[int],
    depot_set: set[int],
    cost: np.ndarray,
    T_tour_ub: float,
) -> tuple[np.ndarray, cp.ndarray]:
    """Compute valid (i, j, v) arc tuples and the GPU cost matrix.

    Applies structural filters (no self-loops, no depot-to-different-depot)
    and the Desrochers-Laporte (1991) Prop. 6 reachability filter: arc
    (i, j, v) is kept only if the minimum tour cost through it,
    ``c[d_v, i] + c[i, j] + c[j, d_v]``, is within the per-tour budget
    ``T_tour_ub``.

    Returns
    -------
    (valid_arcs, cost_gpu)
        ``valid_arcs`` is an (M, 3) int32 NumPy array; ``cost_gpu`` is the
        CuPy cost matrix (reused downstream for K_inf construction).
    """
    is_depot = cp.zeros(n, dtype=cp.bool_)
    depot_arr = cp.array(list(depot_set), dtype=cp.intp)
    is_depot[depot_arr] = True

    depots_gpu = cp.array(depots, dtype=cp.intp)
    i_idx = cp.arange(n, dtype=cp.intp)

    no_self_loop = i_idx[:, None] != i_idx[None, :]
    not_depot_depot = ~(is_depot[:, None] & is_depot[None, :])

    cost_gpu = cp.asarray(cost)
    base_mask = no_self_loop & not_depot_depot

    i_allowed = (~is_depot[None, :]) | (i_idx[None, :] == depots_gpu[:, None])
    j_allowed = (~is_depot[None, :]) | (i_idx[None, :] == depots_gpu[:, None])

    full_mask = i_allowed[:, :, None] & j_allowed[:, None, :] & base_mask[None, :, :]

    if T_tour_ub < float("inf"):
        # Reachability filter (DL1991 Prop. 6 adapted to multi-depot VRP):
        # arc (i, j, v) is infeasible when c[d_v, i] + c[i, j] + c[j, d_v] > T_tour_ub.
        # Treat depot-indexed terms as 0 when i or j is vehicle v's own depot (the
        # round-trip shortcut collapses to just c[i, j] plus one zero leg).
        s_iv = cost_gpu[depots_gpu][:, :].copy()  # (K, n): cost from d_v to i
        t_jv = cost_gpu[:, depots_gpu].T.copy()  # (K, n): cost from j to d_v
        for v in range(K):
            s_iv[v, depots[v]] = 0.0
            t_jv[v, depots[v]] = 0.0
        reach_mask = (s_iv[:, :, None] + cost_gpu[None, :, :] + t_jv[:, None, :]) <= T_tour_ub
        full_mask = full_mask & reach_mask

    v_indices, i_indices, j_indices = cp.where(full_mask)
    valid_arcs = cp.stack([i_indices, j_indices, v_indices], axis=1)
    return cp.asnumpy(valid_arcs).astype(np.int32), cost_gpu


# ─── MIP model builder ─────────────────────────────────────────────────────


def build_vrp_mip(
    dist_matrix: np.ndarray,
    num_vehicles: int,
    depots: list[int],
    alpha: float = VRP_ALPHA,
    warm_start_routes: list[list[int]] | None = None,
    mip_gap: float = MIP_GAP,
) -> tuple:
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
        raise ImportError("PuLP is required for the VRP MIP solver. Install with: pip install pulp")

    n = dist_matrix.shape[0]
    depot_set = set(depots)
    customers = [i for i in range(n) if i not in depot_set]
    n_c = len(customers)
    K = num_vehicles

    logger.info(
        "[MIP] Building model: %d nodes (%d customers, %d depots), %d vehicles",
        n,
        n_c,
        len(depot_set),
        K,
    )

    cost = dist_matrix.astype(np.float64).copy()

    # ── Bound tightening ─────────────────────────────────────────────
    # Valid makespan LB: for each customer j, the vehicle that serves j
    # incurs a tour cost >= its own depot-j-depot round trip (triangle
    # inequality holds since costs are shortest paths on the voxel graph).
    # Hence T >= max_j min_v (c[d_v, j] + c[j, d_v]).
    T_lb = 0.0
    for j in customers:
        cheapest_rt = min(float(cost[depots[v], j] + cost[j, depots[v]]) for v in range(K))
        T_lb = max(T_lb, cheapest_rt)

    T_ws = float("inf")
    C_ws = float("inf")
    T_ub = float("inf")
    if warm_start_routes:
        per_v = _per_vehicle_costs(warm_start_routes, cp.asarray(cost), depots)
        if per_v:
            T_ws = max(per_v)
            C_ws = sum(per_v)
            T_ub = T_ws
        logger.info("[MIP] Warm-start makespan: %.2f, total cost: %.2f", T_ws, C_ws)

    # Objective normalisation.
    T_norm = T_lb if T_lb > 0 else 1.0
    C_norm = K * T_lb if T_lb > 0 else 1.0

    # α-aware per-tour cost upper bound for the Desrochers & Laporte (1991) Prop. 6
    # reachability filter and the triple-based coefficient lift. Derived from
    # f* <= f_ws (warm-start is feasible) plus t_v* <= T* and t_v* <= C*:
    #   α = 1  -> T_tour_ub = T_ws  (makespan-only).
    #   α = 0  -> T_tour_ub = C_ws  (total-cost: any single tour <= total).
    #   mixed  -> α.T*/T_norm <= f_ws gives T* <= T_ws + ((1−α)/α).C_ws.(T_norm/C_norm),
    #            capped by C_ws (the total-cost bound still applies).
    # Keeping T_norm/C_norm explicit so changing the
    # normalisation does not silently invalidate the bound.
    if T_ws == float("inf"):
        T_tour_ub = float("inf")
    elif alpha >= 1.0:
        T_tour_ub = T_ws
    elif alpha <= 0.0:
        T_tour_ub = C_ws
    else:
        T_from_T = T_ws + ((1.0 - alpha) / alpha) * C_ws * (T_norm / C_norm)
        T_tour_ub = min(T_from_T, C_ws)

    logger.info(
        "[MIP] T bounds: [%.2f, %s]; per-tour UB: %s",
        T_lb,
        f"{T_ub:.2f}" if T_ub < float("inf") else "inf",
        f"{T_tour_ub:.2f}" if T_tour_ub < float("inf") else "inf",
    )

    # ── arc validity ──────────────────────────────────
    valid_arcs, cost_gpu = _compute_valid_arcs(n, K, depots, depot_set, cost, T_tour_ub)
    logger.info("[MIP] Valid arcs: %d (of %d possible)", len(valid_arcs), n * (n - 1) * K)

    # ── K_inf: jointly infeasible (predecessor, arc) pairs ──────────────
    # For each surviving arc (i, j, v), K_inf[(i,j,v)] lists customers k with
    # (k, i, v) also surviving and c[d_v, k] + c[k, i] + c[i, j] + c[j, d_v] > T_tour_ub.
    # Under triangle inequality (holds here since costs are shortest paths on the
    # voxel graph), any tour using both (k, i, v) and (i, j, v) has cost at least
    # c[d_v, k] + c[k, i] + c[i, j] + c[j, d_v]; if that exceeds T_tour_ub the pair
    # cannot coexist, giving the forbidden-pair cut x[k,i,v] + x[i,j,v] <= 1 added
    # below. (The DL1991 Eq. 22 coefficient lift encodes the same information but
    # requires CVRP's cumulative-load u_i semantics; adapting it to position-based
    # MTZ without a per-vehicle rank variable cuts off feasible solutions  --  see the
    # comment next to the MTZ loop  --  so we add the cuts as separate constraints.)
    K_inf_map: dict[tuple[int, int, int], list[int]] = {}
    if T_tour_ub < float("inf") and n_c >= 3:
        valid_arc_set = {(int(r[0]), int(r[1]), int(r[2])) for r in valid_arcs}
        # Group surviving arcs by (j, v) to look up predecessors quickly.
        arcs_by_v: dict[int, list[tuple[int, int]]] = {v: [] for v in range(K)}
        preds_by_iv: dict[tuple[int, int], set[int]] = {}
        for i, j, v in valid_arc_set:
            arcs_by_v[v].append((i, j))
            preds_by_iv.setdefault((j, v), set()).add(i)
        for v in range(K):
            if not arcs_by_v[v]:
                continue
            dv = depots[v]
            s_v = cost_gpu[dv, :]  # (n,)
            t_v = cost_gpu[:, dv]  # (n,)
            # tour_cost[k, i, j] = s_v[k] + cost[k,i] + cost[i,j] + t_v[j].
            # Depot-collapse not applied to k because k ranges over customers only.
            tour_cost = (
                s_v[:, None, None]
                + cost_gpu[:, :, None]
                + cost_gpu[None, :, :]
                + t_v[None, None, :]
            )
            infeasible = cp.asnumpy(tour_cost > T_tour_ub)  # (n, n, n) bool
            del tour_cost
            for i, j in arcs_by_v[v]:
                if i in depot_set or j in depot_set:
                    continue
                preds = preds_by_iv.get((i, v), ())
                ks: list[int] = []
                for k in preds:
                    if k in depot_set or k == i or k == j:
                        continue
                    if bool(infeasible[k, i, j]):
                        ks.append(int(k))
                if ks:
                    K_inf_map[(i, j, v)] = ks
    n_pair_cuts = sum(len(v) for v in K_inf_map.values())
    logger.info("[MIP] K_inf triples: %d arcs, %d forbidden-pair cuts", len(K_inf_map), n_pair_cuts)

    # ── Model ────────────────────────────────────────────────────────
    prob = pulp.LpProblem("MinMaxVRP", pulp.LpMinimize)

    x: dict[tuple[int, int, int], pulp.LpVariable] = {}
    for row in valid_arcs:
        i, j, v = int(row[0]), int(row[1]), int(row[2])
        x[i, j, v] = pulp.LpVariable(f"x_{i}_{j}_{v}", cat=pulp.LpBinary)

    u: dict[int, pulp.LpVariable] = {}
    for i in customers:
        u[i] = pulp.LpVariable(f"u_{i}", lowBound=1, upBound=n_c, cat=pulp.LpContinuous)

    T = pulp.LpVariable("T", lowBound=T_lb, cat=pulp.LpContinuous)

    # Lifted u_i upper bound (DL1991 Eq. 11 adapted to multi-depot VRP).
    # If customer i is visited directly from some depot (i.e. i is first on that
    # vehicle's tour), it must sit at position 1 of that tour. The sum over v
    # is at most 1 in any integer feasible solution (visit constraint), so the
    # lift is valid and strictly tightens the LP relaxation.
    #
    # NOTE: The symmetric lifted LOWER bound from DL1991 (u_i >= n_c when i ends
    # some tour) is *not* applied here. DL1991's LB assumes a single TSP tour of
    # length n_c, but in our multi-depot VRP tour lengths L_v are variable  --  a
    # customer ending a short tour has u_i = L_v << n_c, so forcing u_i >= n_c
    # cuts off feasible integer solutions (e.g. a singleton tour has u_i = 1).
    # Generalising the LB would require per-vehicle rank variables u[i, v] plus
    # an explicit L_v expression, which is out of scope.
    if n_c >= 2:
        for i in customers:
            leaves_depot_to_i = pulp.lpSum(
                x[depots[v], i, v] for v in range(K) if (depots[v], i, v) in x
            )
            prob += (
                u[i] <= n_c - (n_c - 1) * leaves_depot_to_i,
                f"u_ub_{i}",
            )
    if T_ub < float("inf"):
        T.upBound = T_ub * 1.01

    # ── Objective ────────────────────────────────────────────────────
    total_cost_expr = pulp.lpSum(cost[i, j] * x[i, j, v] for (i, j, v) in x)
    if alpha >= 1.0:
        prob += T, "Makespan"
    elif alpha <= 0.0:
        prob += total_cost_expr, "TotalCost"
    else:
        prob += (
            alpha * (T / T_norm) + (1 - alpha) * (total_cost_expr / C_norm),
            "Combined",
        )

    # Constraints

    for i in customers:
        prob += (
            pulp.lpSum(x[i, j, v] for v in range(K) for j in range(n) if (i, j, v) in x) == 1,
            f"visit_{i}",
        )

    for v in range(K):
        for i in customers:
            prob += (
                pulp.lpSum(x[j, i, v] for j in range(n) if (j, i, v) in x)
                == pulp.lpSum(x[i, j, v] for j in range(n) if (i, j, v) in x),
                f"flow_{v}_{i}",
            )

    for v in range(K):
        dv = depots[v]
        prob += (
            pulp.lpSum(x[dv, j, v] for j in customers if (dv, j, v) in x) <= 1,
            f"depot_depart_{v}",
        )

    for v in range(K):
        dv = depots[v]
        prob += (
            pulp.lpSum(x[dv, j, v] for j in customers if (dv, j, v) in x)
            == pulp.lpSum(x[j, dv, v] for j in customers if (j, dv, v) in x),
            f"depot_balance_{v}",
        )

    for v in range(K):
        prob += (
            pulp.lpSum(cost[i, j] * x[i, j, v] for (i, j, vv) in x if vv == v) <= T,
            f"makespan_{v}",
        )

    # Lifted MTZ (DL1991 Eq. 8). DL1991's Eq. 22 coefficient lift
    # (+ (n_c − 2) . x[k,i,v] on the MTZ LHS for k ∈ K_inf) is *not* applied
    # here: it is derived for the CVRP cumulative-load u_i formulation and
    # is unsound for position-based MTZ  --  when x[k,i,v] = 1 and x[i,j,v] = 0,
    # the lifted MTZ forces u_i − u_j <= 1 even though j doesn't need to be the
    # successor of i on v, cutting off feasible routes. The same joint
    # infeasibility is captured soundly by the forbidden-pair cuts below.
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

    # Forbidden-pair cuts from K_inf (DL1991 Prop. 6 extended to 3-arc paths).
    # If c[d_v, k] + c[k, i] + c[i, j] + c[j, d_v] > T_tour_ub then no feasible
    # tour on v uses both (k, i) and (i, j), giving x[k,i,v] + x[i,j,v] <= 1.
    for (i, j, v), ks in K_inf_map.items():
        x_ij = x.get((i, j, v))
        if x_ij is None:
            continue
        for k in ks:
            x_ki = x.get((k, i, v))
            if x_ki is None:
                continue
            prob += (x_ki + x_ij <= 1, f"pair_{k}_{i}_{j}_{v}")

    n_vars = len(x) + len(u) + 1
    n_cons = len(prob.constraints)
    logger.info("[MIP] Model: %d variables (%d binary), %d constraints", n_vars, len(x), n_cons)

    warm_start = None
    if warm_start_routes:
        warm_start = _build_warm_start(
            warm_start_routes,
            depots,
            customers,
            n,
            K,
            x,
            u,
            T,
            cost,
        )

    return prob, warm_start


def _build_warm_start(
    routes: list[list[int]],
    depots: list[int],
    customers: list[int],
    n: int,
    K: int,
    x: dict,
    u: dict,
    T_var,
    cost: np.ndarray,
) -> dict[str, float]:
    """Convert a feasible route solution into variable assignments."""
    vals: dict[str, float] = {}

    for key, var in x.items():
        vals[var.name] = 0.0

    for v, route in enumerate(routes):
        if not route:
            continue
        dv = depots[v]
        full = [dv] + list(route) + [dv]
        for a, b in zip(full[:-1], full[1:], strict=False):
            key = (a, b, v)
            if key in x:
                vals[x[key].name] = 1.0

    # Restart the rank counter at the beginning of each vehicle's tour. The new
    # lifted upper bound u_i <= n_c - (n_c - 1) * Σ_v x[d_v, i, v] pins the
    # first customer of every tour to u = 1; a global running counter would set
    # vehicle v>0's first customer to u = 1 + L_0 + ... + L_{v-1} > 1, violating
    # the bound. MTZ is satisfied per vehicle because the counter still
    # increments monotonically inside each tour, and cross-vehicle MTZ
    # constraints are inactive (no arcs between customers of different vehicles).
    customer_set = set(customers)
    for v, route in enumerate(routes):
        order_counter = 1
        for node in route:
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
    depots: list[int],
    n: int,
) -> list[list[int]]:
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
    routes: list[list[int]] = []
    for v in range(K):
        row = route_matrix_cpu[v]
        route = [int(node) for node in row if node >= 0]
        routes.append(route)

    return routes
