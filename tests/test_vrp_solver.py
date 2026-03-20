"""Tests for VRP solver pure functions."""
import numpy as np
import pytest
from VRP.solver.vrp_solver import (
    VRPResult,
    _normalise_depot,
    _compute_route_cost,
    _per_vehicle_costs,
)


class TestNormaliseDepot:
    def test_int_broadcast(self):
        assert _normalise_depot(0, 3) == [0, 0, 0]

    def test_list_passthrough(self):
        assert _normalise_depot([0, 1, 2], 3) == [0, 1, 2]


class TestComputeRouteCost:
    def test_known_cost(self):
        dm = np.array([
            [0, 1, 3, 7],
            [1, 0, 2, 6],
            [3, 2, 0, 4],
            [7, 6, 4, 0],
        ], dtype=np.float32)
        routes = [[1, 2]]  # depot=0 → 0→1→2→0 = 1+2+3 = 6
        cost = _compute_route_cost(routes, dm, depot=0)
        assert abs(cost - 6.0) < 1e-6

    def test_empty_route(self):
        dm = np.array([[0, 1], [1, 0]], dtype=np.float32)
        cost = _compute_route_cost([[]], dm, depot=0)
        assert cost == 0.0


class TestPerVehicleCosts:
    def test_sum_equals_total(self):
        dm = np.array([
            [0, 1, 3, 7],
            [1, 0, 2, 6],
            [3, 2, 0, 4],
            [7, 6, 4, 0],
        ], dtype=np.float32)
        routes = [[1, 2], [3]]
        total = _compute_route_cost(routes, dm, depot=0)
        per_v = _per_vehicle_costs(routes, dm, depot=0)
        assert abs(sum(per_v) - total) < 1e-6

    def test_empty_vehicle_zero(self):
        dm = np.array([[0, 1], [1, 0]], dtype=np.float32)
        per_v = _per_vehicle_costs([[]], dm, depot=0)
        assert per_v == [0.0]


class TestVRPResult:
    def test_construction(self):
        r = VRPResult(routes=[[1]], total_cost=5.0)
        assert r.solver == "unknown"
        assert r.status == "success"


class TestORToolsSolver:
    def test_tiny_problem(self):
        try:
            from VRP.solver.vrp_solver import ORToolsSolver
        except ImportError:
            pytest.skip("ortools not available")

        dm = np.array([
            [0, 1, 2, 3, 4],
            [1, 0, 1, 2, 3],
            [2, 1, 0, 1, 2],
            [3, 2, 1, 0, 1],
            [4, 3, 2, 1, 0],
        ], dtype=np.float32)
        solver = ORToolsSolver(time_limit=5)
        result = solver.solve(dm, num_vehicles=2, depot=0)
        if result.status == "success":
            # All non-depot nodes visited
            visited = set()
            for route in result.routes:
                visited.update(route)
            assert visited == {1, 2, 3, 4}
