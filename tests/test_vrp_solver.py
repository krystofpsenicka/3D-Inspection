"""Tests for VRP solver pure functions."""

import cupy as cp
import numpy as np
import pytest

from VRP.core.types import VRPResult
from VRP.vrp._helpers import (
    compute_route_cost as _compute_route_cost,
)
from VRP.vrp._helpers import (
    per_vehicle_costs as _per_vehicle_costs,
)


class TestComputeRouteCost:
    def test_known_cost(self):
        dm = cp.array(
            [
                [0, 1, 3, 7],
                [1, 0, 2, 6],
                [3, 2, 0, 4],
                [7, 6, 4, 0],
            ],
            dtype=cp.float32,
        )
        routes = [[1, 2]]  # depots=0 → 0→1→2→0 = 1+2+3 = 6
        cost = _compute_route_cost(routes, dm, depots=[0])
        assert abs(cost - 6.0) < 1e-6

    def test_empty_route(self):
        dm = cp.array([[0, 1], [1, 0]], dtype=cp.float32)
        cost = _compute_route_cost([[]], dm, depots=[0])
        assert cost == 0.0


class TestPerVehicleCosts:
    def test_sum_equals_total(self):
        dm = cp.array(
            [
                [0, 1, 3, 7],
                [1, 0, 2, 6],
                [3, 2, 0, 4],
                [7, 6, 4, 0],
            ],
            dtype=cp.float32,
        )
        routes = [[1, 2], [3]]
        total = _compute_route_cost(routes, dm, depots=[0, 0])
        per_v = _per_vehicle_costs(routes, dm, depots=[0, 0])
        assert abs(sum(per_v) - total) < 1e-6

    def test_empty_vehicle_zero(self):
        dm = cp.array([[0, 1], [1, 0]], dtype=cp.float32)
        per_v = _per_vehicle_costs([[]], dm, depots=[0])
        assert per_v == [0.0]


class TestVRPResult:
    def test_construction(self):
        r = VRPResult(routes=[[1]], total_cost=5.0)
        assert r.solver == "unknown"
        assert r.status == "success"


class TestNearestNeighborWarmstart:
    def test_tiny_problem(self):
        from VRP.vrp._helpers import nearest_neighbor_warmstart as _nearest_neighbor_warmstart

        dm = cp.array(
            [
                [0, 1, 2, 3, 4],
                [1, 0, 1, 2, 3],
                [2, 1, 0, 1, 2],
                [3, 2, 1, 0, 1],
                [4, 3, 2, 1, 0],
            ],
            dtype=cp.float32,
        )
        routes = _nearest_neighbor_warmstart(dm, num_vehicles=2, depots=[0, 0])
        # All non-depot nodes visited
        visited = set()
        for route in routes:
            visited.update(route)
        assert visited == {1, 2, 3, 4}
