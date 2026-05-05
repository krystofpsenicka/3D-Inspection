"""Tests for VRP coordinate transforms and per-route cost helpers."""

from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

from shared.occupancy_grid import OccupancyGrid
from VRP.vrp._helpers import (
    compute_route_cost as _compute_route_cost,
)
from VRP.vrp._helpers import (
    per_vehicle_costs as _per_vehicle_costs,
)


class TestCoordinateTransforms:
    def test_round_trip(self):
        """world -> voxel -> world is within res/2 per axis."""
        res = 0.5
        og = OccupancyGrid(
            grid=cp.zeros((100, 100, 100), dtype=cp.bool_),
            origin=cp.array([0.0, 0.0, 0.0], dtype=cp.float64),
            resolution=res,
        )
        xyz = cp.array([[1.3, 2.7, 0.4]], dtype=cp.float64)

        ijk = og.world_to_voxel(xyz)
        recovered = og.voxel_to_world(ijk).get()[0]

        diff = np.abs(recovered - xyz.get()[0])
        assert np.all(diff <= res), f"Round-trip error too large: diff={diff}, max allowed={res}"

    def test_known_values(self):
        """(1.0, 2.0, 3.0) at res=0.5 -> voxel (2, 4, 6) -> world (1.25, 2.25, 3.25)."""
        res = 0.5
        og = OccupancyGrid(
            grid=cp.zeros((100, 100, 100), dtype=cp.bool_),
            origin=cp.array([0.0, 0.0, 0.0], dtype=cp.float64),
            resolution=res,
        )
        xyz = cp.array([[1.0, 2.0, 3.0]], dtype=cp.float64)

        ijk = og.world_to_voxel(xyz)
        np.testing.assert_array_equal(ijk.get()[0], [2, 4, 6])

        world = og.voxel_to_world(ijk)
        np.testing.assert_allclose(world.get()[0], [1.25, 2.25, 3.25])


class TestRouteCostHelpers:
    def test_cost_uses_forward_direction(self):
        """Asymmetric matrix: cost(0->1->2->0) uses d[0,1]+d[1,2]+d[2,0]."""
        dm = cp.array(
            [
                [0, 1, 99],
                [99, 0, 2],
                [3, 99, 0],
            ],
            dtype=cp.float64,
        )
        cost = _compute_route_cost([[1, 2]], dm, depots=[0])
        # 0->1 + 1->2 + 2->0 = 1 + 2 + 3 = 6
        assert abs(cost - 6.0) < 1e-6

    def test_per_vehicle_costs_multi_depot(self):
        """Each vehicle's cost computed from its own depot."""
        dm = cp.array(
            [
                [0, 10, 20, 30],
                [10, 0, 15, 25],
                [20, 15, 0, 5],
                [30, 25, 5, 0],
            ],
            dtype=cp.float64,
        )
        # Vehicle 0: depots=0, route=[2] -> 0->2->0 = 20+20 = 40
        # Vehicle 1: depots=1, route=[3] -> 1->3->1 = 25+25 = 50
        per_v = _per_vehicle_costs([[2], [3]], dm, depots=[0, 1])
        assert abs(per_v[0] - 40.0) < 1e-6
        assert abs(per_v[1] - 50.0) < 1e-6
