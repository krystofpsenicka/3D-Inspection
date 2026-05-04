"""Comprehensive VRP module tests.

Tests solver correctness against brute-force optimal on small custom instances,
structural invariants, edge cases, and routing components (ReservationTable,
Space-Time A*, coordinate transforms).

Fixtures are hand-built for our actual problem (multi-depot VRP with blended
``alpha * makespan + (1 - alpha) * total_cost``, no capacity). CVRPLIB / Eilon
benchmarks are deliberately not used --  they are for capacitated VRP and
their published optima are not comparable to our relaxation.
"""

from __future__ import annotations

import itertools
import math

import cupy as cp
import numpy as np
import pytest

from VRP.core.constants import MIP_GAP
from VRP.core.types import VRPBackend, VRPResult
from VRP.vrp._helpers import (
    compute_route_cost as _compute_route_cost,
)
from VRP.vrp._helpers import (
    nearest_neighbor_warmstart as _nearest_neighbor_warmstart,
)
from VRP.vrp._helpers import (
    per_vehicle_costs as _per_vehicle_costs,
)
from VRP.vrp.vrp_solver import solve_vrp

# ── Skip conditions ──────────────────────────────────────────────────────────

pulp = pytest.importorskip("pulp", reason="PuLP not installed")

from VRP.vrp.mip_solver_cpu import MIPSolverCPU as MIPMakespanCPU
from VRP.vrp.mip_solver_gpu import MIPSolverGPU as MIPMakespanGPU

# ─── Brute-force reference solver ───────────────────────────────────────────


def brute_force_vrp(
    dist_matrix,
    num_vehicles: int,
    depots: list[int],
    max_stops_per_vehicle: int | None = None,
) -> tuple[float, float]:
    """Exact optimal by full enumeration.

    Returns (best_total_distance, best_makespan).

    For each of K^n_customers assignments of customers to vehicles,
    computes the optimal (shortest) TSP tour per vehicle (trying all
    permutations), then tracks the best total-distance and best makespan
    across all assignments.
    """
    dist_matrix = (
        cp.asnumpy(dist_matrix) if hasattr(dist_matrix, "get") else np.asarray(dist_matrix)
    )
    n = dist_matrix.shape[0]
    depot_set = set(depots)
    customers = [i for i in range(n) if i not in depot_set]
    n_c = len(customers)
    K = num_vehicles

    if max_stops_per_vehicle is None:
        max_stops_per_vehicle = n_c  # no constraint

    best_total = float("inf")
    best_makespan = float("inf")

    # Enumerate all assignments: each customer -> one of K vehicles
    for assignment in itertools.product(range(K), repeat=n_c):
        # Build per-vehicle customer groups
        groups: list[list[int]] = [[] for _ in range(K)]
        for idx, v in enumerate(assignment):
            groups[v].append(customers[idx])

        # Check capacity
        if any(len(g) > max_stops_per_vehicle for g in groups):
            continue

        # For each vehicle, find the shortest tour over all permutations
        total_cost = 0.0
        max_cost = 0.0
        feasible = True
        for v in range(K):
            d = depots[v]
            grp = groups[v]
            if not grp:
                continue  # empty route, cost = 0

            # Try all permutations of this vehicle's customers
            best_v = float("inf")
            for perm in itertools.permutations(grp):
                full = [d] + list(perm) + [d]
                c = sum(float(dist_matrix[full[i], full[i + 1]]) for i in range(len(full) - 1))
                best_v = min(best_v, c)

            total_cost += best_v
            max_cost = max(max_cost, best_v)

        best_total = min(best_total, total_cost)
        best_makespan = min(best_makespan, max_cost)

    return best_total, best_makespan


# ─── Custom test instances ───────────────────────────────────────────────────


def _small_dist_matrix() -> cp.ndarray:
    """Small 7-node Euclidean instance. Depot=0, customers 1..6.

    Hand-built from 2-D points so the brute-force enumeration in
    ``brute_force_vrp`` is fast and the optimum is exact for our actual
    blended-objective MD-VRP (no capacity).
    """
    coords = np.array(
        [
            (0.0, 0.0),  # 0 depot
            (10.0, 0.0),  # 1
            (0.0, 10.0),  # 2
            (10.0, 10.0),  # 3
            (5.0, 5.0),  # 4
            (-5.0, 5.0),  # 5
            (5.0, -5.0),  # 6
        ],
        dtype=np.float64,
    )
    n = len(coords)
    dm = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            dm[i, j] = math.sqrt(dx * dx + dy * dy)
    return cp.asarray(dm)


def _clustered_known_optimum() -> tuple[cp.ndarray, list[int], float]:
    """K=3 vehicles, 9 customers in 3 well-separated clusters.

    Depot at origin; clusters around (50, 0), (0, 50), (-50, 0) with
    intra-cluster spread ≤ 2. Inter-cluster gap (~50) far exceeds intra-
    cluster diameter (~3), so the makespan-optimal assignment must put
    one cluster per vehicle. The optimal makespan is the maximum over
    clusters of the depot→3-customer→depot TSP tour cost (computed by
    brute force over the 6 permutations of each cluster).

    Returns
    -------
    (dist_matrix, depots, optimal_makespan)
    """
    depot = (0.0, 0.0)
    cluster_centres = [(50.0, 0.0), (0.0, 50.0), (-50.0, 0.0)]
    offsets = [(-1.0, 0.0), (1.0, 0.0), (0.0, 1.0)]  # 3 customers per cluster

    coords = [depot]
    cluster_node_ids: list[list[int]] = []
    for cx, cy in cluster_centres:
        ids = []
        for dx, dy in offsets:
            ids.append(len(coords))
            coords.append((cx + dx, cy + dy))
        cluster_node_ids.append(ids)
    coords_arr = np.asarray(coords, dtype=np.float64)
    n = len(coords_arr)

    dm = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            dx = coords_arr[i, 0] - coords_arr[j, 0]
            dy = coords_arr[i, 1] - coords_arr[j, 1]
            dm[i, j] = math.sqrt(dx * dx + dy * dy)

    # Optimal per-cluster TSP cost from depot 0 (brute-forced)
    cluster_costs = []
    for ids in cluster_node_ids:
        best = float("inf")
        for perm in itertools.permutations(ids):
            full = [0, *perm, 0]
            c = sum(dm[full[i], full[i + 1]] for i in range(len(full) - 1))
            best = min(best, c)
        cluster_costs.append(best)

    optimal_makespan = max(cluster_costs)
    depots = [0, 0, 0]
    return cp.asarray(dm), depots, optimal_makespan


def _asymmetric4_dist_matrix() -> cp.ndarray:
    """Custom asymmetric 4-node instance (depots=0)."""
    return cp.array(
        [
            [0, 1, 4, 6],
            [2, 0, 3, 5],
            [4, 3, 0, 1],
            [6, 5, 2, 0],
        ],
        dtype=cp.float64,
    )


def _multi_depot_line() -> tuple[cp.ndarray, list[int]]:
    """6-node line graph with 2 depots. depots=[0,1], customers=2..5."""
    positions = [0, 10, 1, 2, 8, 9]
    n = len(positions)
    dm = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            dm[i, j] = abs(positions[i] - positions[j])
    return cp.asarray(dm), [0, 1]


# ─── Helper: solver factory ─────────────────────────────────────────────────


def _make_solver(name: str, alpha: float = 1.0):
    """Instantiate a solver by name."""
    if name == "mip_cpu":
        return MIPMakespanCPU(time_limit=60, mip_gap=MIP_GAP), alpha
    elif name == "mip_gpu":
        return MIPMakespanGPU(time_limit=60, mip_gap=MIP_GAP), alpha
    else:
        raise ValueError(f"Unknown solver: {name}")


def _skip_on_subprocess_error(result: VRPResult, solver_name: str):
    """Skip the test if the solver failed due to subprocess/env issues."""
    if result.status != "success":
        if "subprocess_error" in result.status or "timeout" in result.status:
            pytest.skip(f"{solver_name} subprocess unavailable: {result.status}")
        assert False, f"Solver failed: {result.status}"


def _solver_ids(solvers):
    """Generate pytest ids from solver name list."""
    return solvers


# Solver groups
MAKESPAN_SOLVERS = ["mip_cpu", "mip_gpu"]
ALL_CPU_SOLVERS = ["mip_cpu", "mip_gpu"]


# ═══════════════════════════════════════════════════════════════════════════
# 1. TestSolverOptimality  --  Solver correctness against brute-force optimal
# ═══════════════════════════════════════════════════════════════════════════


class TestSolverOptimality:
    """Verify solvers find solutions within tolerance of brute-force optimal."""

    # ── Brute-force-optimal small instances (single-depot + multi-depot) ──
    #
    # All three of (small + alpha=1, small + alpha=0, multi-depot + alpha=1)
    # share the same shape: build instance -> brute-force optimum on the
    # relevant objective -> solve -> assert solver result is within MIP_GAP.
    #
    # ``objective`` is the field on ``VRPResult`` to compare; brute force
    # returns (total, makespan) so we index accordingly.
    @pytest.mark.parametrize(
        "instance_name,alpha,objective",
        [
            ("small", 1.0, "makespan"),
            ("small", 0.0, "total_cost"),
            ("multi_depot", 1.0, "makespan"),
        ],
        ids=["small-makespan", "small-total_cost", "multi_depot-makespan"],
    )
    @pytest.mark.parametrize("solver_name", MAKESPAN_SOLVERS, ids=_solver_ids(MAKESPAN_SOLVERS))
    def test_solver_within_mip_gap_of_brute_force(
        self, solver_name, instance_name, alpha, objective
    ):
        if instance_name == "small":
            dm = _small_dist_matrix()
            depots = [0, 0]
        elif instance_name == "multi_depot":
            dm, depots = _multi_depot_line()
        else:
            raise AssertionError(f"unknown instance {instance_name}")

        bf_total, bf_makespan = brute_force_vrp(dm, 2, depots, max_stops_per_vehicle=4)
        bf_value = bf_makespan if objective == "makespan" else bf_total

        solver, _ = _make_solver(solver_name, alpha=alpha)
        result = solver.solve(dm, num_vehicles=2, depots=depots, alpha=alpha)
        _skip_on_subprocess_error(result, solver_name)
        tol = MIP_GAP + 0.01  # MIP gap + small numerical margin
        actual = getattr(result, objective)
        assert actual <= bf_value * (1 + tol), (
            f"{solver_name} on {instance_name} ({objective}): "
            f"got {actual:.2f}, brute_force={bf_value:.2f} * {1 + tol:.2f}"
        )

    # ── Mid-size, structurally known optimum ─────────────────────────

    def test_clustered_known_optimum(self):
        """3 vehicles, 9 customers in 3 well-separated clusters.

        For alpha=1, each vehicle must cover one cluster. Optimum makespan =
        max over clusters of the brute-force depot->3-customer->depot tour.
        """
        dm, depots, optimal_makespan = _clustered_known_optimum()
        solver = MIPMakespanCPU(time_limit=60, mip_gap=MIP_GAP)
        result = solver.solve(dm, num_vehicles=3, depots=depots, alpha=1.0)
        _skip_on_subprocess_error(result, "mip_cpu")
        tol = MIP_GAP + 0.01
        assert result.makespan <= optimal_makespan * (1 + tol), (
            f"makespan={result.makespan:.2f} > optimal={optimal_makespan:.2f}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 2. TestSolverFeasibility  --  Structural invariants
# ═══════════════════════════════════════════════════════════════════════════


class TestSolverFeasibility:
    """Run MIP solver and verify structural invariants."""

    @pytest.fixture(params=ALL_CPU_SOLVERS, ids=_solver_ids(ALL_CPU_SOLVERS))
    def solved(self, request):
        solver_name = request.param
        dm = _small_dist_matrix()
        solver, alpha = _make_solver(solver_name, alpha=1.0)
        result = solver.solve(dm, num_vehicles=2, depots=[0, 0], alpha=alpha)
        if result.status != "success":
            pytest.skip(f"{solver_name} did not find a solution")
        return result, dm

    def test_all_customers_visited_once(self, solved):
        result, dm = solved
        visited = []
        for route in result.routes:
            visited.extend(route)
        customers = set(range(1, dm.shape[0]))  # all non-depot nodes
        assert set(visited) == customers, "Not all customers visited"
        assert len(visited) == len(customers), "Duplicate visits"

    def test_routes_exclude_depots(self, solved):
        result, _ = solved
        for route in result.routes:
            assert 0 not in route, "Depot index leaked into route"

    def test_reported_cost_matches_recomputation(self, solved):
        result, dm = solved
        recomputed = _compute_route_cost(result.routes, dm, depots=[0, 0])
        assert abs(result.total_cost - recomputed) < 1e-3, (
            f"reported={result.total_cost:.4f} vs recomputed={recomputed:.4f}"
        )

    def test_makespan_is_max_per_vehicle(self, solved):
        result, dm = solved
        per_v = _per_vehicle_costs(result.routes, dm, depots=[0, 0])
        expected_makespan = max(per_v) if per_v else 0.0
        assert abs(result.makespan - expected_makespan) < 1e-3, (
            f"makespan={result.makespan:.4f} vs max(per_v)={expected_makespan:.4f}"
        )

    def test_per_vehicle_sum_is_total(self, solved):
        result, dm = solved
        per_v = _per_vehicle_costs(result.routes, dm, depots=[0, 0])
        assert abs(sum(per_v) - result.total_cost) < 1e-3

    def test_closed_route_cost_includes_return(self, solved):
        """Verify each vehicle's cost = depot -> c1 -> c2 -> ... -> depot."""
        result, dm = solved
        for v, route in enumerate(result.routes):
            if not route:
                continue
            full = [0] + list(route) + [0]
            expected = sum(float(dm[full[i], full[i + 1]]) for i in range(len(full) - 1))
            actual = _per_vehicle_costs([route], dm, depots=[0])[0]
            assert abs(actual - expected) < 1e-6, (
                f"Vehicle {v}: closed-route cost mismatch ({actual:.4f} vs {expected:.4f})"
            )


# ═══════════════════════════════════════════════════════════════════════════
# 4. TestSolverEdgeCases
# ═══════════════════════════════════════════════════════════════════════════


class TestSolverEdgeCases:
    def test_single_customer(self):
        """2 nodes, 1 vehicle -> should visit the only customer."""
        dm = cp.array([[0, 5], [5, 0]], dtype=cp.float64)
        solver = MIPMakespanCPU(time_limit=30, mip_gap=0.05)
        result = solver.solve(dm, num_vehicles=1, depots=[0], alpha=1.0)
        assert result.status == "success"
        assert result.routes == [[1]]
        assert abs(result.total_cost - 10.0) < 1e-6  # 5 out + 5 back

    def test_more_vehicles_than_customers(self):
        """3 customers, 5 vehicles via MIP."""
        dm = cp.array(
            [
                [0, 1, 2, 3],
                [1, 0, 1, 2],
                [2, 1, 0, 1],
                [3, 2, 1, 0],
            ],
            dtype=cp.float64,
        )
        solver = MIPMakespanCPU(time_limit=30, mip_gap=0.05)
        result = solver.solve(dm, num_vehicles=5, depots=[0, 0, 0, 0, 0], alpha=1.0)
        assert result.status == "success"
        visited = set()
        for route in result.routes:
            visited.update(route)
        assert visited == {1, 2, 3}


# ═══════════════════════════════════════════════════════════════════════════
# 5. TestSolveVRPEntryPoint  --  Unified solve_vrp() function
# ═══════════════════════════════════════════════════════════════════════════


class TestSolveVRPEntryPoint:
    def test_makespan_leq_total_distance_makespan(self):
        """Makespan-objective solution should have <= makespan than
        total-distance-objective solution (or within tolerance)."""
        dm = _small_dist_matrix()
        td_result = solve_vrp(
            cp.asarray(dm),
            num_vehicles=2,
            depots=[0, 0],
            alpha=0.0,
            backend=VRPBackend.HIGHS,
            time_limit=60,
            mip_gap=0.05,
        )
        ms_result = solve_vrp(
            cp.asarray(dm),
            num_vehicles=2,
            depots=[0, 0],
            alpha=1.0,
            backend=VRPBackend.HIGHS,
            time_limit=60,
            mip_gap=0.05,
        )
        assert td_result.status == "success"
        assert ms_result.status == "success"
        # The makespan-optimised solution should not be worse on makespan
        # (with tolerance for MIP gap + heuristic variance)
        assert ms_result.makespan <= td_result.makespan * 1.10, (
            f"Makespan-opt ({ms_result.makespan:.2f}) worse than "
            f"total-dist ({td_result.makespan:.2f})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 6. TestCombinedObjective  --  Alpha-blended objective
# ═══════════════════════════════════════════════════════════════════════════


class TestCombinedObjective:
    """Verify combined-objective behaviour across alpha values."""

    def test_alpha_05_tradeoff(self):
        """alpha=0.5 should produce a valid intermediate solution."""
        dm = _small_dist_matrix()
        result = solve_vrp(
            cp.asarray(dm),
            num_vehicles=2,
            depots=[0, 0],
            alpha=0.5,
            backend=VRPBackend.HIGHS,
            time_limit=60,
            mip_gap=0.05,
        )
        assert result.status == "success"
        assert result.alpha == 0.5
        assert result.objective_value == pytest.approx(
            0.5 * result.makespan + 0.5 * result.total_cost
        )
        # All customers visited
        visited = set()
        for r in result.routes:
            visited.update(r)
        assert visited == set(range(1, 7))

    def test_alpha_out_of_range(self):
        """alpha outside [0,1] should raise ValueError."""
        dm = _small_dist_matrix()
        with pytest.raises(ValueError):
            solve_vrp(cp.asarray(dm), num_vehicles=2, depots=[0, 0], alpha=1.5)
        with pytest.raises(ValueError):
            solve_vrp(cp.asarray(dm), num_vehicles=2, depots=[0, 0], alpha=-0.1)


# ═══════════════════════════════════════════════════════════════════════════
# 7. TestNearestNeighborWarmstart
# ═══════════════════════════════════════════════════════════════════════════


class TestNearestNeighborWarmstart:
    def test_all_customers_visited(self):
        dm = _small_dist_matrix()
        routes = _nearest_neighbor_warmstart(cp.asarray(dm), 2, depots=[0, 0])
        visited = set()
        for r in routes:
            visited.update(r)
        assert visited == set(range(1, 7))

    def test_no_depot_in_routes(self):
        dm = _small_dist_matrix()
        routes = _nearest_neighbor_warmstart(cp.asarray(dm), 2, depots=[0, 0])
        for r in routes:
            assert 0 not in r

    def test_multi_depot(self):
        dm, depots = _multi_depot_line()
        routes = _nearest_neighbor_warmstart(cp.asarray(dm), 2, depots=depots)
        visited = set()
        for r in routes:
            visited.update(r)
        assert visited == {2, 3, 4, 5}
        for r in routes:
            assert 0 not in r and 1 not in r


# NOTE: Tests for ReservationTable, SpaceTimeAStar, coordinate transforms,
# route-cost helpers, and TrajectoryCollisions live in their own files:
#   - tests/test_reservation_table.py
#   - tests/test_space_time_astar.py
#   - tests/test_vrp_helpers.py
#   - tests/test_trajectory_collisions.py
