"""Combined-objective VRP via Mixed-Integer Programming.

Minimises ``alpha * T/T_norm + (1-alpha) * total_cost/C_norm`` where T is
the makespan and total_cost is the sum of all arc costs. Uses lifted MTZ
subtour elimination (Desrochers & Laporte, 1991) and variable reduction
(Lalla-Ruiz & Mes, 2021). Both GPU (cuOpt) and CPU (PuLP/CBC) backends
share the same MIP formulation.

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
import math
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ._helpers import per_vehicle_costs as _per_vehicle_costs

logger = logging.getLogger(__name__)


# ─── MIP model builder ───────────────────────────────────────────────────────

def build_makespan_mip(
    dist_matrix: np.ndarray,
    num_vehicles: int,
    depot: list[int],
    capacity: int,
    alpha: float = 1.0,
    warm_start_routes: Optional[List[List[int]]] = None,
    mip_gap: float = 0.05,
) -> Tuple[str, Optional[Dict]]:
    """Build the min-max makespan MIP and export as an MPS string.

    Parameters
    ----------
    dist_matrix : (N, N) float
        Full distance matrix (depots + customers).
    num_vehicles : int
        Number of vehicles (K).
    depot : int or list[int]
        Per-vehicle depot indices.
    capacity : int
        Max customers per vehicle (Q).
    alpha : float
        Objective blending parameter in [0, 1].  1.0 = pure makespan,
        0.0 = pure total distance, intermediate = normalised blend.
    warm_start_routes : list[list[int]], optional
        Feasible solution for warm-starting (e.g. from nearest-neighbor).
    mip_gap : float
        Relative optimality gap passed to the solver later (used here
        only for logging).

    Returns
    -------
    (prob, warm_start_dict)
        PuLP LpProblem and optional warm-start variable values dict.
    """
    try:
        import pulp
    except ImportError:
        raise ImportError(
            "PuLP is required for the MIP makespan solver. "
            "Install with: pip install pulp"
        )

    n = dist_matrix.shape[0]
    depots = depot
    depot_set = set(depots)
    customers = [i for i in range(n) if i not in depot_set]
    n_c = len(customers)
    K = num_vehicles

    logger.info("[MIP] Building model: %d nodes (%d customers, %d depots), "
                "%d vehicles, capacity=%d",
                n, n_c, len(depot_set), K, capacity)

    # Closed VRP: use real return-to-depot costs so the makespan bound
    # (constraint 5) accounts for the full round trip.
    cost = dist_matrix.astype(np.float64).copy()

    # ── Bound tightening ─────────────────────────────────────────────
    # Lower bound: longest shortest round-trip from any depot
    T_lb = 0.0
    for v in range(K):
        dv = depots[v]
        best_rt = min(
            (float(cost[dv, j] + cost[j, dv]) for j in customers),
            default=0.0,
        )
        T_lb = max(T_lb, best_rt)

    # Upper bound from warm start
    T_ub = float("inf")
    if warm_start_routes:
        per_v = _per_vehicle_costs(warm_start_routes, cost, depots)
        T_ub = max(per_v) if per_v else float("inf")
        logger.info("[MIP] Warm-start makespan upper bound: %.2f", T_ub)

    logger.info("[MIP] T bounds: [%.2f, %s]",
                T_lb, f"{T_ub:.2f}" if T_ub < float("inf") else "∞")

    # ── Model ────────────────────────────────────────────────────────
    prob = pulp.LpProblem("MinMaxVRP", pulp.LpMinimize)

    # Decision variables
    # x[i,j,v] — only created for valid arcs (variable reduction per
    # Lalla-Ruiz & Mes 2021: no self-loops, no cross-depot, no depot-depot)
    x: Dict[Tuple[int, int, int], pulp.LpVariable] = {}
    for v in range(K):
        dv = depots[v]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Depot constraints: vehicle v only uses its own depot
                if i in depot_set and i != dv:
                    continue
                if j in depot_set and j != dv:
                    continue
                # No depot-to-depot arcs
                if i in depot_set and j in depot_set:
                    continue
                # Arc fixing: if cost exceeds best known T, skip.
                # Only valid for pure makespan (alpha=1); for blended
                # objectives the optimal T may exceed T_ub.
                if alpha >= 1.0 and T_ub < float("inf") and cost[i, j] > T_ub:
                    continue
                x[i, j, v] = pulp.LpVariable(
                    f"x_{i}_{j}_{v}", cat=pulp.LpBinary,
                )

    # u[i] — visit order for customers (single-indexed, shared across
    # vehicles; valid because constraint 1 ensures each customer is
    # assigned to exactly one vehicle — Toth & Vigo 2014, §2.3)
    u: Dict[int, pulp.LpVariable] = {}
    for i in customers:
        u[i] = pulp.LpVariable(f"u_{i}", lowBound=1, upBound=n_c,
                                cat=pulp.LpContinuous)

    # T — makespan
    T = pulp.LpVariable("T", lowBound=T_lb, cat=pulp.LpContinuous)
    if T_ub < float("inf"):
        T.upBound = T_ub * 1.01  # tiny slack for numerical safety

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

    # ── Constraint 1: each customer visited exactly once ─────────────
    for i in customers:
        prob += (
            pulp.lpSum(
                x[i, j, v]
                for v in range(K) for j in range(n)
                if (i, j, v) in x
            ) == 1,
            f"visit_{i}",
        )

    # ── Constraint 2: flow conservation (customers only) ─────────────
    for v in range(K):
        for i in customers:
            prob += (
                pulp.lpSum(x[j, i, v] for j in range(n) if (j, i, v) in x)
                ==
                pulp.lpSum(x[i, j, v] for j in range(n) if (i, j, v) in x),
                f"flow_{v}_{i}",
            )

    # ── Constraint 3: depot departure ≤ 1 ───────────────────────────
    for v in range(K):
        dv = depots[v]
        prob += (
            pulp.lpSum(x[dv, j, v] for j in customers if (dv, j, v) in x)
            <= 1,
            f"depot_depart_{v}",
        )

    # ── Constraint 4: depot return = departure ──────────────────────
    for v in range(K):
        dv = depots[v]
        prob += (
            pulp.lpSum(x[dv, j, v] for j in customers if (dv, j, v) in x)
            ==
            pulp.lpSum(x[j, dv, v] for j in customers if (j, dv, v) in x),
            f"depot_balance_{v}",
        )

    # ── Constraint 5: makespan bound ────────────────────────────────
    for v in range(K):
        prob += (
            pulp.lpSum(
                cost[i, j] * x[i, j, v]
                for (i, j, vv) in x if vv == v
            ) <= T,
            f"makespan_{v}",
        )

    # ── Constraint 6: lifted MTZ (Desrochers & Laporte 1991) ────────
    # u[i] - u[j] + |C|·x[i,j,v] + (|C|-2)·x[j,i,v] ≤ |C| - 1
    # Tighter than standard MTZ: adds the (|C|-2)·x[j,i,v] term which
    # strengthens the LP relaxation at no extra variable cost.
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

    # ── Constraint 7: capacity ──────────────────────────────────────
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

    # ── Warm start ───────────────────────────────────────────────────
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

    # Set all x to 0
    for key, var in x.items():
        vals[var.name] = 0.0

    # Set active arcs
    for v, route in enumerate(routes):
        if not route:
            continue
        dv = depots[v]
        full = [dv] + list(route) + [dv]
        for a, b in zip(full[:-1], full[1:]):
            key = (a, b, v)
            if key in x:
                vals[x[key].name] = 1.0

    # Set u values (visit order within route)
    customer_set = set(customers)
    order_counter = 1
    for v, route in enumerate(routes):
        for pos, node in enumerate(route):
            if node in customer_set and node in u:
                vals[u[node].name] = float(order_counter)
                order_counter += 1

    # Compute makespan
    per_v = _per_vehicle_costs(routes, cost, depots)
    vals[T_var.name] = max(per_v) if per_v else 0.0

    return vals


# ─── CPU backend: HiGHS ──────────────────────────────────────────────────────

class MIPMakespanCPU:
    """Min-max VRP via MIP — HiGHS (CPU).

    Uses the shared MIP formulation from :func:`build_makespan_mip` and
    solves with the HiGHS solver via the PuLP interface.
    """

    def __init__(self, time_limit: int = 120, mip_gap: float = 0.05):
        self.time_limit = time_limit
        self.mip_gap = mip_gap

    def solve(
        self,
        dist_matrix: np.ndarray,
        num_vehicles: int,
        depot: list[int] = None,
        alpha: float = 1.0,
        warm_start_routes: Optional[List[List[int]]] = None,
    ):
        from ..core.types import VRPResult

        depots = depot
        depot_set = set(depots)
        n = dist_matrix.shape[0]
        n_inspection = n - len(depot_set)
        base_cap = math.ceil(n_inspection / num_vehicles)
        capacity = base_cap + max(1, math.ceil(0.15 * base_cap))

        try:
            import pulp
        except ImportError:
            logger.error("[MIPMakespanCPU] PuLP not installed.")
            return VRPResult(routes=[], total_cost=float("inf"),
                             solver="highs_mip", status="import_error: pulp")

        prob, warm_start = build_makespan_mip(
            dist_matrix, num_vehicles, depot, capacity,
            alpha=alpha,
            warm_start_routes=warm_start_routes,
            mip_gap=self.mip_gap,
        )

        # Solve with HiGHS via PuLP
        solver = pulp.HiGHS(
            timeLimit=self.time_limit,
            gapRel=self.mip_gap,
            msg=1,
            warmStart=(warm_start is not None),
        )

        # Apply warm start values
        if warm_start:
            for var in prob.variables():
                if var.name in warm_start:
                    var.varValue = warm_start[var.name]

        logger.info("[MIPMakespanCPU] Solving with HiGHS (limit=%ds, gap=%.1f%%) …",
                    self.time_limit, self.mip_gap * 100)

        prob.solve(solver)

        status_str = pulp.LpStatus[prob.status]
        logger.info("[MIPMakespanCPU] Solver status: %s", status_str)

        if prob.status not in (pulp.constants.LpStatusOptimal,):
            # Also accept "Not Solved" if we have a feasible incumbent
            has_solution = any(
                v.varValue is not None and v.varValue > 0.5
                for v in prob.variables() if v.name.startswith("x_")
            )
            if not has_solution:
                logger.warning("[MIPMakespanCPU] No feasible solution found.")
                return VRPResult(routes=[], total_cost=float("inf"),
                                 solver="highs_mip",
                                 status=f"no_solution ({status_str})")

        # Extract routes from x variables
        routes = _extract_routes_from_pulp(prob, num_vehicles, depots, n)
        total_cost = sum(
            float(dist_matrix[a, b])
            for route, dv in zip(routes, depots)
            for a, b in zip([dv] + route, route + [dv])
            if route
        )
        per_v = _per_vehicle_costs(routes, dist_matrix, depots)
        makespan = max(per_v) if per_v else 0.0

        logger.info("[MIPMakespanCPU] makespan=%.2f  total_cost=%.2f  "
                    "per_vehicle=%s",
                    makespan, total_cost,
                    [f"{c:.1f}" for c in per_v])

        return VRPResult(
            routes=routes,
            total_cost=total_cost,
            makespan=makespan,
            per_vehicle_costs=per_v,
            solver="highs_mip",
            status="success",
        )


# ─── GPU backend: cuOpt MILP ─────────────────────────────────────────────────

class MIPMakespanGPU:
    """Min-max VRP via MIP — cuOpt MILP solver (GPU).

    Builds the MIP with PuLP, exports to MPS, and runs the cuOpt MILP
    solver in a subprocess inside the ``rapids_solver`` conda environment.
    """

    def __init__(
        self,
        rapids_python: str = "",
        time_limit: int = 120,
        mip_gap: float = 0.05,
        timeout: int = 300,
    ):
        import os
        from ..core.constants import RAPIDS_PYTHON, VRP_ROOT
        self.rapids_python = os.path.expanduser(rapids_python or RAPIDS_PYTHON)
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.timeout = timeout
        self._script = os.path.join(VRP_ROOT, "solver", "mip_makespan_subprocess.py")

    def solve(
        self,
        dist_matrix: np.ndarray,
        num_vehicles: int,
        depot: list[int] = None,
        alpha: float = 1.0,
        warm_start_routes: Optional[List[List[int]]] = None,
    ):
        import json
        import os
        import subprocess

        from ..core.types import VRPResult

        depots = depot
        depot_set = set(depots)
        n = dist_matrix.shape[0]
        n_inspection = n - len(depot_set)
        base_cap = math.ceil(n_inspection / num_vehicles)
        capacity = base_cap + max(1, math.ceil(0.15 * base_cap))

        # Build MIP and export MPS
        prob, warm_start = build_makespan_mip(
            dist_matrix, num_vehicles, depot, capacity,
            alpha=alpha,
            warm_start_routes=warm_start_routes,
            mip_gap=self.mip_gap,
        )

        # Write MPS to temp file
        mps_path = tempfile.mktemp(suffix=".mps")
        prob.writeMPS(mps_path)

        cfg = {
            "mps_path": mps_path,
            "time_limit": self.time_limit,
            "mip_gap": self.mip_gap,
            "num_vehicles": num_vehicles,
            "depot": depots,
            "n": n,
        }
        if warm_start:
            cfg["warm_start"] = warm_start

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as cfg_f:
            json.dump(cfg, cfg_f)
            cfg_path = cfg_f.name

        out_path = cfg_path.replace(".json", "_out.json")

        try:
            cmd = [self.rapids_python, self._script, cfg_path, out_path]
            logger.info("[MIPMakespanGPU] Running cuOpt MILP subprocess: %s",
                        " ".join(cmd))

            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout,
            )

            if proc.returncode != 0:
                logger.warning(
                    "[MIPMakespanGPU] Subprocess failed (rc=%d):\n%s",
                    proc.returncode, proc.stderr,
                )
                return VRPResult(
                    routes=[], total_cost=float("inf"),
                    solver="cuopt_mip",
                    status=f"subprocess_error rc={proc.returncode}",
                )

            with open(out_path) as f:
                result = json.load(f)

            if result.get("status") != "success":
                return VRPResult(
                    routes=[], total_cost=float("inf"),
                    solver="cuopt_mip",
                    status=result.get("status", "unknown"),
                )

            routes = result["routes"]
            per_v = _per_vehicle_costs(routes, dist_matrix, depots)
            makespan = max(per_v) if per_v else 0.0
            total_cost = sum(per_v)

            logger.info("[MIPMakespanGPU] makespan=%.2f  total_cost=%.2f  "
                        "per_vehicle=%s",
                        makespan, total_cost,
                        [f"{c:.1f}" for c in per_v])

            return VRPResult(
                routes=routes,
                total_cost=total_cost,
                makespan=makespan,
                per_vehicle_costs=per_v,
                solver="cuopt_mip",
                status="success",
            )

        except subprocess.TimeoutExpired:
            logger.error("[MIPMakespanGPU] Subprocess timed out after %ds",
                         self.timeout)
            return VRPResult(routes=[], total_cost=float("inf"),
                             solver="cuopt_mip", status="timeout")
        except Exception as exc:
            logger.error("[MIPMakespanGPU] Unexpected error: %s", exc)
            return VRPResult(routes=[], total_cost=float("inf"),
                             solver="cuopt_mip", status=f"error: {exc}")
        finally:
            for p in (mps_path, cfg_path, out_path):
                try:
                    os.unlink(p)
                except FileNotFoundError:
                    pass


# ─── Route extraction from solved PuLP model ─────────────────────────────────

def _extract_routes_from_pulp(
    prob,
    num_vehicles: int,
    depots: List[int],
    n: int,
) -> List[List[int]]:
    """Extract per-vehicle routes from solved PuLP variables."""
    depot_set = set(depots)

    # Build adjacency from active x variables
    # x_{i}_{j}_{v} with value > 0.5
    adj: Dict[int, Dict[int, int]] = {v: {} for v in range(num_vehicles)}
    for var in prob.variables():
        if var.name.startswith("x_") and var.varValue is not None and var.varValue > 0.5:
            parts = var.name.split("_")
            # x_{i}_{j}_{v}
            i, j, v = int(parts[1]), int(parts[2]), int(parts[3])
            adj[v][i] = j

    routes: List[List[int]] = []
    for v in range(num_vehicles):
        route: List[int] = []
        dv = depots[v]
        current = adj[v].get(dv)
        visited = set()
        while current is not None and current not in depot_set and current not in visited:
            route.append(current)
            visited.add(current)
            current = adj[v].get(current)
        routes.append(route)

    return routes
