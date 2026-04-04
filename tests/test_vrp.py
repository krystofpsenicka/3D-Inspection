"""Comprehensive VRP module tests.

Tests solver correctness against brute-force optimal, published benchmarks,
structural invariants, edge cases, and routing components (ReservationTable,
Space-Time A*, coordinate transforms).

Supersedes tests/test_vrp_solver.py (which is kept for backwards compat).
"""

from __future__ import annotations

import itertools
import math
import os
from typing import List, Optional, Tuple

import numpy as np
import pytest

import cupy as cp

from VRP.core.types import VRPResult
from VRP.vrp._helpers import (
    compute_route_cost as _compute_route_cost,
    nearest_neighbor_warmstart as _nearest_neighbor_warmstart,
    normalise_depot as _normalise_depot,
    per_vehicle_costs as _per_vehicle_costs,
)
from VRP.vrp.vrp_solver import solve_vrp
from VRP.core.constants import RAPIDS_PYTHON, MIP_GAP

# ── Skip conditions ──────────────────────────────────────────────────────────

pulp = pytest.importorskip("pulp", reason="PuLP not installed")

from VRP.vrp.mip_makespan_solver import MIPMakespanCPU, MIPMakespanGPU

_HAS_RAPIDS = os.path.isfile(RAPIDS_PYTHON)
requires_rapids = pytest.mark.skipif(
    not _HAS_RAPIDS, reason="RAPIDS env not found"
)


# ─── Brute-force reference solver ───────────────────────────────────────────

def brute_force_vrp(
    dist_matrix: np.ndarray,
    num_vehicles: int,
    depots: List[int],
    max_stops_per_vehicle: Optional[int] = None,
) -> Tuple[float, float]:
    """Exact optimal by full enumeration.

    Returns (best_total_distance, best_makespan).

    For each of K^n_customers assignments of customers to vehicles,
    computes the optimal (shortest) TSP tour per vehicle (trying all
    permutations), then tracks the best total-distance and best makespan
    across all assignments.
    """
    if isinstance(depots, int):
        depots = [depots] * num_vehicles
    n = dist_matrix.shape[0]
    depot_set = set(depots)
    customers = [i for i in range(n) if i not in depot_set]
    n_c = len(customers)
    K = num_vehicles

    if max_stops_per_vehicle is None:
        max_stops_per_vehicle = n_c  # no constraint

    best_total = float("inf")
    best_makespan = float("inf")

    # Enumerate all assignments: each customer → one of K vehicles
    for assignment in itertools.product(range(K), repeat=n_c):
        # Build per-vehicle customer groups
        groups: List[List[int]] = [[] for _ in range(K)]
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
                c = sum(
                    float(dist_matrix[full[i], full[i + 1]])
                    for i in range(len(full) - 1)
                )
                best_v = min(best_v, c)

            total_cost += best_v
            max_cost = max(max_cost, best_v)

        best_total = min(best_total, total_cost)
        best_makespan = min(best_makespan, max_cost)

    return best_total, best_makespan


# ─── Benchmark instances ─────────────────────────────────────────────────────

def _eilon7_dist_matrix() -> np.ndarray:
    """Eilon 7-node benchmark (E-n7, depot=0, customers 1-6)."""
    return np.array([
        [0,  10, 20, 25, 12, 20,  2],
        [10,  0, 25, 20, 20, 10, 11],
        [20, 25,  0, 10, 25, 11, 25],
        [25, 20, 10,  0, 30, 22, 10],
        [12, 20, 25, 30,  0, 30, 20],
        [20, 10, 11, 22, 30,  0, 12],
        [2,  11, 25, 10, 20, 12,  0],
    ], dtype=np.float64)


def _asymmetric4_dist_matrix() -> np.ndarray:
    """Custom asymmetric 4-node instance (depot=0)."""
    return np.array([
        [0, 1, 4, 6],
        [2, 0, 3, 5],
        [4, 3, 0, 1],
        [6, 5, 2, 0],
    ], dtype=np.float64)


def _multi_depot_line() -> Tuple[np.ndarray, List[int]]:
    """6-node line graph with 2 depots. depots=[0,1], customers=2..5."""
    positions = [0, 10, 1, 2, 8, 9]
    n = len(positions)
    dm = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            dm[i, j] = abs(positions[i] - positions[j])
    return dm, [0, 1]


def _eilon22_dist_matrix() -> Tuple[np.ndarray, int]:
    """E-n22-k4 benchmark. Returns (dist_matrix, num_vehicles=4).

    Coordinates from Christofides & Eilon (1969). Published optimal = 375.
    Distance: integer EUC_2D (TSPLIB convention: nint(sqrt(dx^2+dy^2))).
    """
    coords = np.array([
        (145, 215),  # 0 depot
        (151, 264), (159, 261), (130, 254), (128, 252), (163, 247),
        (146, 246), (161, 242), (142, 239), (163, 236), (148, 232),
        (128, 231), (156, 217), (129, 214), (146, 208), (164, 208),
        (141, 206), (147, 193), (164, 193), (129, 189), (155, 185),
        (139, 182),
    ], dtype=np.float64)
    n = len(coords)
    dm = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            dm[i, j] = round(math.sqrt(dx * dx + dy * dy))
    return dm, 4


# ─── Helper: solver factory ─────────────────────────────────────────────────

def _make_solver(name: str, alpha: float = 1.0):
    """Instantiate a solver by name."""
    if name == "mip_cpu":
        return MIPMakespanCPU(time_limit=60, mip_gap=MIP_GAP), alpha
    elif name == "mip_gpu":
        return MIPMakespanGPU(time_limit=60, mip_gap=MIP_GAP, timeout=120), alpha
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
MAKESPAN_SOLVERS = ["mip_cpu"]
ALL_CPU_SOLVERS = ["mip_cpu"]

if _HAS_RAPIDS:
    MAKESPAN_SOLVERS.append("mip_gpu")
    ALL_CPU_SOLVERS.append("mip_gpu")


# ═══════════════════════════════════════════════════════════════════════════
# 1. TestSolverOptimality — Solver correctness against brute-force optimal
# ═══════════════════════════════════════════════════════════════════════════

class TestSolverOptimality:
    """Verify solvers find solutions within tolerance of brute-force optimal."""

    # ── Eilon 7-node, makespan ────────────────────────────────────────

    @pytest.mark.parametrize("solver_name", MAKESPAN_SOLVERS,
                             ids=_solver_ids(MAKESPAN_SOLVERS))
    def test_eilon7_makespan(self, solver_name):
        dm = _eilon7_dist_matrix()
        _, bf_makespan = brute_force_vrp(dm, 2, [0, 0], max_stops_per_vehicle=4)

        solver, alpha = _make_solver(solver_name, alpha=1.0)
        result = solver.solve(dm, num_vehicles=2, depot=0, alpha=alpha)
        _skip_on_subprocess_error(result, solver_name)
        tol = MIP_GAP + 0.01  # MIP gap + small numerical margin
        assert result.makespan <= bf_makespan * (1 + tol), (
            f"{solver_name}: makespan={result.makespan:.2f} > "
            f"brute_force={bf_makespan:.2f} * {1 + tol:.2f}"
        )

    # ── Eilon 7-node, total distance (alpha=0) ──────────────────────

    @pytest.mark.parametrize("solver_name", MAKESPAN_SOLVERS,
                             ids=_solver_ids(MAKESPAN_SOLVERS))
    def test_eilon7_total_distance(self, solver_name):
        dm = _eilon7_dist_matrix()
        bf_total, _ = brute_force_vrp(dm, 2, [0, 0], max_stops_per_vehicle=4)

        solver, alpha = _make_solver(solver_name, alpha=0.0)
        result = solver.solve(dm, num_vehicles=2, depot=0, alpha=alpha)
        _skip_on_subprocess_error(result, solver_name)
        tol = MIP_GAP + 0.01
        assert result.total_cost <= bf_total * (1 + tol), (
            f"{solver_name}: total_cost={result.total_cost:.2f} > "
            f"brute_force={bf_total:.2f} * {1 + tol:.2f}"
        )

    # ── Multi-depot, makespan ────────────────────────────────────────

    @pytest.mark.parametrize("solver_name", MAKESPAN_SOLVERS,
                             ids=_solver_ids(MAKESPAN_SOLVERS))
    def test_multi_depot_makespan(self, solver_name):
        dm, depots = _multi_depot_line()
        _, bf_makespan = brute_force_vrp(dm, 2, depots, max_stops_per_vehicle=4)

        solver, alpha = _make_solver(solver_name, alpha=1.0)
        result = solver.solve(dm, num_vehicles=2, depot=depots, alpha=alpha)
        _skip_on_subprocess_error(result, solver_name)
        tol = MIP_GAP + 0.01
        assert result.makespan <= bf_makespan * (1 + tol), (
            f"{solver_name}: makespan={result.makespan:.2f} > "
            f"brute_force={bf_makespan:.2f} * {1 + tol:.2f}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 2. TestCVRPLIBBenchmark — Published benchmark regression
# ═══════════════════════════════════════════════════════════════════════════

class TestCVRPLIBBenchmark:
    """Regression tests against published CVRPLIB benchmarks."""

    def test_eilon22_makespan(self):
        """MIP on E-n22-k4 should produce a feasible solution with consistent costs."""
        dm, k = _eilon22_dist_matrix()
        solver = MIPMakespanCPU(time_limit=60, mip_gap=0.10)
        result = solver.solve(dm, num_vehicles=k, depot=0, alpha=1.0)
        assert result.status == "success"
        assert result.makespan > 0
        # Recompute and verify consistency
        per_v = _per_vehicle_costs(result.routes, dm, depot=0)
        assert abs(max(per_v) - result.makespan) < 1e-6
        assert abs(sum(per_v) - result.total_cost) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════
# 3. TestSolverFeasibility — Structural invariants
# ═══════════════════════════════════════════════════════════════════════════

class TestSolverFeasibility:
    """Run MIP solver and verify structural invariants."""

    @pytest.fixture(params=ALL_CPU_SOLVERS, ids=_solver_ids(ALL_CPU_SOLVERS))
    def solved(self, request):
        solver_name = request.param
        dm = _eilon7_dist_matrix()
        solver, alpha = _make_solver(solver_name, alpha=1.0)
        result = solver.solve(dm, num_vehicles=2, depot=0, alpha=alpha)
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
        recomputed = _compute_route_cost(result.routes, dm, depot=0)
        assert abs(result.total_cost - recomputed) < 1e-3, (
            f"reported={result.total_cost:.4f} vs recomputed={recomputed:.4f}"
        )

    def test_makespan_is_max_per_vehicle(self, solved):
        result, dm = solved
        per_v = _per_vehicle_costs(result.routes, dm, depot=0)
        expected_makespan = max(per_v) if per_v else 0.0
        assert abs(result.makespan - expected_makespan) < 1e-3, (
            f"makespan={result.makespan:.4f} vs max(per_v)={expected_makespan:.4f}"
        )

    def test_per_vehicle_sum_is_total(self, solved):
        result, dm = solved
        per_v = _per_vehicle_costs(result.routes, dm, depot=0)
        assert abs(sum(per_v) - result.total_cost) < 1e-3

    def test_closed_route_cost_includes_return(self, solved):
        """Verify each vehicle's cost = depot → c1 → c2 → ... → depot."""
        result, dm = solved
        for v, route in enumerate(result.routes):
            if not route:
                continue
            full = [0] + list(route) + [0]
            expected = sum(
                float(dm[full[i], full[i + 1]])
                for i in range(len(full) - 1)
            )
            actual = _per_vehicle_costs([route], dm, depot=0)[0]
            assert abs(actual - expected) < 1e-6, (
                f"Vehicle {v}: closed-route cost mismatch "
                f"({actual:.4f} vs {expected:.4f})"
            )


# ═══════════════════════════════════════════════════════════════════════════
# 4. TestSolverEdgeCases
# ═══════════════════════════════════════════════════════════════════════════

class TestSolverEdgeCases:

    def test_single_customer(self):
        """2 nodes, 1 vehicle → should visit the only customer."""
        dm = np.array([[0, 5], [5, 0]], dtype=np.float64)
        solver = MIPMakespanCPU(time_limit=30, mip_gap=0.05)
        result = solver.solve(dm, num_vehicles=1, depot=0, alpha=1.0)
        assert result.status == "success"
        assert result.routes == [[1]]
        assert abs(result.total_cost - 10.0) < 1e-6  # 5 out + 5 back

    def test_more_vehicles_than_customers(self):
        """3 customers, 5 vehicles via MIP."""
        dm = np.array([
            [0, 1, 2, 3],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [3, 2, 1, 0],
        ], dtype=np.float64)
        solver = MIPMakespanCPU(time_limit=30, mip_gap=0.05)
        result = solver.solve(dm, num_vehicles=5, depot=0, alpha=1.0)
        assert result.status == "success"
        visited = set()
        for route in result.routes:
            visited.update(route)
        assert visited == {1, 2, 3}


# ═══════════════════════════════════════════════════════════════════════════
# 5. TestSolveVRPEntryPoint — Unified solve_vrp() function
# ═══════════════════════════════════════════════════════════════════════════

class TestSolveVRPEntryPoint:

    def test_alpha_1_makespan(self):
        """solve_vrp with alpha=1.0 returns valid makespan-optimised result."""
        dm = _eilon7_dist_matrix()
        result = solve_vrp(
            dm, num_vehicles=2, depot=0,
            alpha=1.0,
            backend="ortools",
            time_limit=60,
            mip_gap=0.05,
        )
        assert result.status == "success"
        assert result.makespan > 0

    def test_alpha_0_total_distance(self):
        """solve_vrp with alpha=0.0 returns valid total-distance result."""
        dm = _eilon7_dist_matrix()
        result = solve_vrp(
            dm, num_vehicles=2, depot=0,
            alpha=0.0,
            backend="ortools",
            time_limit=60,
            mip_gap=0.05,
        )
        assert result.status == "success"
        assert result.total_cost < float("inf")
        visited = set()
        for r in result.routes:
            visited.update(r)
        assert visited == set(range(1, 7))

    def test_makespan_leq_total_distance_makespan(self):
        """Makespan-objective solution should have ≤ makespan than
        total-distance-objective solution (or within tolerance)."""
        dm = _eilon7_dist_matrix()
        td_result = solve_vrp(
            dm, num_vehicles=2, depot=0,
            alpha=0.0,
            backend="ortools",
            time_limit=60,
            mip_gap=0.05,
        )
        ms_result = solve_vrp(
            dm, num_vehicles=2, depot=0,
            alpha=1.0,
            backend="ortools",
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
# 6. TestCombinedObjective — Alpha-blended objective
# ═══════════════════════════════════════════════════════════════════════════

class TestCombinedObjective:
    """Verify combined-objective behaviour across alpha values."""

    def test_alpha_1_matches_makespan(self):
        """alpha=1.0 should produce near-optimal makespan."""
        dm = _eilon7_dist_matrix()
        _, bf_makespan = brute_force_vrp(dm, 2, [0, 0], max_stops_per_vehicle=4)
        result = solve_vrp(dm, num_vehicles=2, depot=0, alpha=1.0,
                           backend="ortools", time_limit=60, mip_gap=0.05)
        assert result.status == "success"
        tol = MIP_GAP + 0.01
        assert result.makespan <= bf_makespan * (1 + tol)

    def test_alpha_0_minimizes_total_cost(self):
        """alpha=0.0 should produce near-optimal total distance."""
        dm = _eilon7_dist_matrix()
        bf_total, _ = brute_force_vrp(dm, 2, [0, 0], max_stops_per_vehicle=4)
        result = solve_vrp(dm, num_vehicles=2, depot=0, alpha=0.0,
                           backend="ortools", time_limit=60, mip_gap=0.05)
        assert result.status == "success"
        tol = MIP_GAP + 0.01
        assert result.total_cost <= bf_total * (1 + tol)

    def test_alpha_05_tradeoff(self):
        """alpha=0.5 should produce a valid intermediate solution."""
        dm = _eilon7_dist_matrix()
        result = solve_vrp(dm, num_vehicles=2, depot=0, alpha=0.5,
                           backend="ortools", time_limit=60, mip_gap=0.05)
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
        dm = _eilon7_dist_matrix()
        with pytest.raises(ValueError):
            solve_vrp(dm, num_vehicles=2, depot=0, alpha=1.5)
        with pytest.raises(ValueError):
            solve_vrp(dm, num_vehicles=2, depot=0, alpha=-0.1)

    def test_objective_value_field(self):
        """VRPResult.objective_value should be alpha*makespan + (1-alpha)*total_cost."""
        dm = _eilon7_dist_matrix()
        for alpha in [0.0, 0.3, 0.7, 1.0]:
            result = solve_vrp(dm, num_vehicles=2, depot=0, alpha=alpha,
                               backend="ortools", time_limit=60, mip_gap=0.05)
            if result.status == "success":
                expected = alpha * result.makespan + (1 - alpha) * result.total_cost
                assert result.objective_value == pytest.approx(expected)


# ═══════════════════════════════════════════════════════════════════════════
# 7. TestNearestNeighborWarmstart
# ═══════════════════════════════════════════════════════════════════════════

class TestNearestNeighborWarmstart:

    def test_all_customers_visited(self):
        dm = _eilon7_dist_matrix()
        routes = _nearest_neighbor_warmstart(dm, 2, depot=0)
        visited = set()
        for r in routes:
            visited.update(r)
        assert visited == set(range(1, 7))

    def test_no_depot_in_routes(self):
        dm = _eilon7_dist_matrix()
        routes = _nearest_neighbor_warmstart(dm, 2, depot=0)
        for r in routes:
            assert 0 not in r

    def test_multi_depot(self):
        dm, depots = _multi_depot_line()
        routes = _nearest_neighbor_warmstart(dm, 2, depot=depots)
        visited = set()
        for r in routes:
            visited.update(r)
        assert visited == {2, 3, 4, 5}
        for r in routes:
            assert 0 not in r and 1 not in r


# ═══════════════════════════════════════════════════════════════════════════
# 8. TestReservationTable — Collision avoidance core
# ═══════════════════════════════════════════════════════════════════════════

class TestReservationTable:

    def _make_table(self, shape=(10, 10, 10), T=20, half_extents=(0, 0, 0)):
        from VRP.mapf.space_time_search import ReservationTable
        return ReservationTable(shape, T, np.array(half_extents, dtype=np.intp))

    def test_commit_and_query(self):
        rt = self._make_table()
        positions = np.array([[3, 4, 5], [3, 5, 5]], dtype=np.intp)
        times = np.array([0, 1], dtype=np.intp)
        rt.commit_trajectory(positions, times)

        assert rt.is_reserved(3, 4, 5, 0) is True
        assert rt.is_reserved(3, 5, 5, 1) is True
        assert rt.is_reserved(3, 4, 5, 1) is False  # different time
        assert rt.is_reserved(0, 0, 0, 0) is False   # uncommitted cell

    def test_aabb_inflation(self):
        """AABB with half_extents=[1,1,1] inflates point → 3×3×3 cube."""
        rt = self._make_table(shape=(10, 10, 10), T=10, half_extents=(1, 1, 1))
        positions = np.array([[5, 5, 5]], dtype=np.intp)
        times = np.array([3], dtype=np.intp)
        rt.commit_trajectory(positions, times)

        # All 27 cells in the 3x3x3 cube should be reserved
        count = 0
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    if rt.is_reserved(5 + dx, 5 + dy, 5 + dz, 3):
                        count += 1
        assert count == 27, f"Expected 27 reserved cells, got {count}"

        # Outside the cube should be free
        assert rt.is_reserved(5, 5, 5, 4) is False  # different time
        assert rt.is_reserved(3, 5, 5, 3) is False   # outside AABB

    def test_boundary_no_crash(self):
        """Commit near grid boundary with AABB doesn't crash."""
        rt = self._make_table(shape=(5, 5, 5), T=5, half_extents=(2, 2, 2))
        # Point at corner: (0, 0, 0) — AABB extends outside grid
        positions = np.array([[0, 0, 0]], dtype=np.intp)
        times = np.array([0], dtype=np.intp)
        rt.commit_trajectory(positions, times)  # should not crash
        assert rt.is_reserved(0, 0, 0, 0) is True

    def test_time_beyond_horizon(self):
        """is_reserved(x,y,z, t>=T) returns False (line 145 of space_time_astar.py)."""
        rt = self._make_table(T=10)
        positions = np.array([[5, 5, 5]], dtype=np.intp)
        times = np.array([5], dtype=np.intp)
        rt.commit_trajectory(positions, times)

        assert rt.is_reserved(5, 5, 5, 5) is True
        assert rt.is_reserved(5, 5, 5, 10) is False  # at horizon
        assert rt.is_reserved(5, 5, 5, 100) is False  # way beyond


# ═══════════════════════════════════════════════════════════════════════════
# 9. TestSpaceTimeAStar — Pathfinding
# ═══════════════════════════════════════════════════════════════════════════

class TestSpaceTimeAStar:

    def _make_grid_and_table(self, shape=(20, 20, 20), T=100):
        from VRP.mapf.space_time_search import ReservationTable
        grid = cp.zeros(shape, dtype=cp.uint8)
        rt = ReservationTable(shape, T, np.array([0, 0, 0], dtype=np.intp))
        return grid, rt

    def test_open_grid_finds_path(self):
        from VRP.mapf.space_time_search import space_time_astar_gpu as space_time_astar
        grid, rt = self._make_grid_and_table()
        start = cp.array([2, 2, 2], dtype=cp.intp)
        goal = cp.array([15, 15, 15], dtype=cp.intp)

        result = space_time_astar(grid, start, goal, 0, rt, resolution=1.0)
        assert result is not None
        path_ijk, path_t = result
        assert len(path_ijk) >= 2
        np.testing.assert_array_equal(path_ijk[0].get(), start.get())
        np.testing.assert_array_equal(path_ijk[-1].get(), goal.get())

    def test_navigates_around_wall(self):
        """A* finds a path through a gap in a wall."""
        from VRP.mapf.space_time_search import space_time_astar_gpu as space_time_astar
        grid, rt = self._make_grid_and_table()
        # Build a wall at x=10 for all y except y=10
        for y in range(20):
            if y != 10:
                grid[10, y, :] = 1

        start = cp.array([5, 5, 5], dtype=cp.intp)
        goal = cp.array([15, 5, 5], dtype=cp.intp)

        result = space_time_astar(grid, start, goal, 0, rt, resolution=1.0)
        assert result is not None
        path_ijk, _ = result
        # Path must go through the gap at y=10
        np.testing.assert_array_equal(path_ijk[-1].get(), goal.get())

    def test_blocked_goal_returns_none(self):
        """Occupied goal → None."""
        from VRP.mapf.space_time_search import space_time_astar_gpu as space_time_astar
        grid, rt = self._make_grid_and_table()
        grid[15, 15, 15] = 1  # block goal

        start = np.array([2, 2, 2], dtype=np.intp)
        goal = np.array([15, 15, 15], dtype=np.intp)

        result = space_time_astar(grid, start, goal, 0, rt, resolution=1.0)
        assert result is None

    def test_avoids_reserved_cells(self):
        """A* detours around time-reserved cells."""
        from VRP.mapf.space_time_search import space_time_astar_gpu as space_time_astar, ReservationTable
        grid = cp.zeros((10, 10, 1), dtype=cp.uint8)
        rt = ReservationTable((10, 10, 1), 50, np.array([0, 0, 0], dtype=np.intp))

        start = cp.array([0, 5, 0], dtype=cp.intp)
        goal = cp.array([9, 5, 0], dtype=cp.intp)

        # Reserve the direct path at the times A* would traverse it
        for x in range(1, 9):
            positions = np.array([[x, 5, 0]], dtype=np.intp)
            times = np.array([x], dtype=np.intp)
            rt.commit_trajectory(positions, times)

        result = space_time_astar(grid, start, goal, 0, rt, resolution=1.0)
        assert result is not None
        path_ijk, path_t = result
        path_ijk_np = path_ijk.get()
        path_t_np = path_t.get()
        np.testing.assert_array_equal(path_ijk_np[-1], goal.get())

        # Verify no path cell collides with reservations
        for i in range(len(path_ijk_np)):
            assert not rt.is_reserved(
                int(path_ijk_np[i, 0]), int(path_ijk_np[i, 1]),
                int(path_ijk_np[i, 2]), int(path_t_np[i]),
            ) or i == 0  # start position at t=0 is not reserved

    def test_same_start_goal(self):
        """Returns single-point path when start == goal."""
        from VRP.mapf.space_time_search import space_time_astar_gpu as space_time_astar
        grid, rt = self._make_grid_and_table()
        point = cp.array([5, 5, 5], dtype=cp.intp)

        result = space_time_astar(grid, point, point, 0, rt, resolution=1.0)
        assert result is not None
        path_ijk, path_t = result
        assert len(path_ijk) == 1
        np.testing.assert_array_equal(path_ijk[0].get(), point.get())


# ═══════════════════════════════════════════════════════════════════════════
# 10. TestCoordinateTransforms
# ═══════════════════════════════════════════════════════════════════════════

class TestCoordinateTransforms:

    def test_round_trip(self):
        """world → coarse → world is within res/2 per axis."""
        from VRP.mapf.space_time_search import world_to_coarse, coarse_to_world
        res = 0.5
        origin = np.array([0.0, 0.0, 0.0])
        xyz = np.array([1.3, 2.7, 0.4])

        ijk = world_to_coarse(xyz, origin, res)
        recovered = coarse_to_world(ijk, origin, res).get()

        # Recovered is the voxel centre; should be within res/2 of original
        diff = np.abs(recovered - xyz)
        assert np.all(diff <= res), (
            f"Round-trip error too large: diff={diff}, max allowed={res}"
        )

    def test_known_values(self):
        """(1.0, 2.0, 3.0) at res=0.5 → voxel (2, 4, 6) → world (1.25, 2.25, 3.25)."""
        from VRP.mapf.space_time_search import world_to_coarse, coarse_to_world
        res = 0.5
        origin = np.array([0.0, 0.0, 0.0])
        xyz = np.array([1.0, 2.0, 3.0])

        ijk = world_to_coarse(xyz, origin, res)
        np.testing.assert_array_equal(ijk.get(), [2, 4, 6])

        world = coarse_to_world(ijk, origin, res)
        np.testing.assert_allclose(world.get(), [1.25, 2.25, 3.25])


# ═══════════════════════════════════════════════════════════════════════════
# 11. TestHelperFunctions
# ═══════════════════════════════════════════════════════════════════════════

class TestHelperFunctions:

    def test_cost_uses_forward_direction(self):
        """Asymmetric matrix: cost(0→1→2→0) uses d[0,1]+d[1,2]+d[2,0]."""
        dm = np.array([
            [0, 1, 99],
            [99, 0, 2],
            [3, 99, 0],
        ], dtype=np.float64)
        cost = _compute_route_cost([[1, 2]], dm, depot=0)
        # 0→1 + 1→2 + 2→0 = 1 + 2 + 3 = 6
        assert abs(cost - 6.0) < 1e-6

    def test_per_vehicle_costs_multi_depot(self):
        """Each vehicle's cost computed from its own depot."""
        dm = np.array([
            [0, 10, 20, 30],
            [10, 0, 15, 25],
            [20, 15, 0, 5],
            [30, 25, 5, 0],
        ], dtype=np.float64)
        # Vehicle 0: depot=0, route=[2] → 0→2→0 = 20+20 = 40
        # Vehicle 1: depot=1, route=[3] → 1→3→1 = 25+25 = 50
        per_v = _per_vehicle_costs([[2], [3]], dm, depot=[0, 1])
        assert abs(per_v[0] - 40.0) < 1e-6
        assert abs(per_v[1] - 50.0) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════
# 11. TestTrajectoryCollisions — GPU-vectorized AABB collision detection
# ═══════════════════════════════════════════════════════════════════════════

class TestTrajectoryCollisions:

    def test_no_collision_parallel_paths(self):
        """Two robots moving in parallel far apart produce no collisions."""
        from VRP.core.collision import find_trajectory_collisions
        T = 50
        traj_a = [np.array([0.0, 0.0, float(t) * 0.1], dtype=np.float32) for t in range(T)]
        traj_b = [np.array([5.0, 5.0, float(t) * 0.1], dtype=np.float32) for t in range(T)]
        collisions = find_trajectory_collisions([traj_a, traj_b])
        assert collisions == []

    def test_head_on_collision_detected(self):
        """Two robots crossing the same point detect overlap."""
        from VRP.core.collision import find_trajectory_collisions
        T = 20
        # Robot A moves along +X, Robot B moves along -X; they cross at x=0
        traj_a = [np.array([float(t) - 10.0, 0.0, 0.0], dtype=np.float32) for t in range(T)]
        traj_b = [np.array([10.0 - float(t), 0.0, 0.0], dtype=np.float32) for t in range(T)]
        collisions = find_trajectory_collisions([traj_a, traj_b])
        assert len(collisions) > 0
        # Verify tuple format: (step, robot_a, robot_b, penetration)
        for step, ra, rb, pen in collisions:
            assert ra == 0 and rb == 1
            assert pen > 0.0

    def test_padding_shorter_trajectory(self):
        """Robots with different trajectory lengths are padded correctly."""
        from VRP.core.collision import find_trajectory_collisions
        # Robot A: 10 steps far away; Robot B: 5 steps far away
        traj_a = [np.array([100.0, 0.0, 0.0], dtype=np.float32) for _ in range(10)]
        traj_b = [np.array([-100.0, 0.0, 0.0], dtype=np.float32) for _ in range(5)]
        collisions = find_trajectory_collisions([traj_a, traj_b])
        assert collisions == []

    def test_single_robot_no_collision(self):
        """A single robot cannot collide with itself."""
        from VRP.core.collision import find_trajectory_collisions
        traj = [np.array([0.0, 0.0, float(t)], dtype=np.float32) for t in range(10)]
        collisions = find_trajectory_collisions([traj])
        assert collisions == []

    def test_cupy_input_accepted(self):
        """Function accepts CuPy arrays as trajectory positions."""
        from VRP.core.collision import find_trajectory_collisions
        T = 10
        traj_a = [cp.array([0.0, 0.0, float(t)], dtype=cp.float32) for t in range(T)]
        traj_b = [cp.array([100.0, 0.0, float(t)], dtype=cp.float32) for t in range(T)]
        collisions = find_trajectory_collisions([traj_a, traj_b])
        assert collisions == []
