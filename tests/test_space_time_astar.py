"""Tests for VRP/mapf/space_time_search.py:space_time_astar_gpu.

Covers open-grid pathfinding, navigation around walls, blocked-goal
handling, reservation-aware detours, and the start==goal edge case.
Split out of test_vrp.py.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from shared.occupancy_grid import OccupancyGrid
from VRP.mapf.reservation_table import ReservationTable
from VRP.mapf.space_time_search import space_time_astar_gpu as space_time_astar


def _make_og_and_table(shape=(20, 20, 20), T=100):
    grid = cp.zeros(shape, dtype=cp.bool_)
    og = OccupancyGrid(grid=grid, origin=cp.zeros(3, dtype=cp.float64), resolution=1.0)
    rt = ReservationTable(shape, T, robot_collision_radius=0.0)
    return og, rt


class TestSpaceTimeAStar:
    def test_open_grid_finds_path(self):
        og, rt = _make_og_and_table()
        start = cp.array([2, 2, 2], dtype=cp.intp)
        goal = cp.array([15, 15, 15], dtype=cp.intp)

        result = space_time_astar(og, start, goal, 0, rt)
        assert result is not None, "A* failed to return a path on an open grid"
        path_ijk, _ = result
        assert len(path_ijk) >= 2
        np.testing.assert_array_equal(path_ijk[0].get(), start.get())
        np.testing.assert_array_equal(path_ijk[-1].get(), goal.get())

    def test_navigates_around_wall(self):
        """A* finds a path through a gap in a wall."""
        og, rt = _make_og_and_table()
        # Build a wall at x=10 for all y except y=10
        for y in range(20):
            if y != 10:
                og.grid[10, y, :] = True

        start = cp.array([5, 5, 5], dtype=cp.intp)
        goal = cp.array([15, 5, 5], dtype=cp.intp)

        result = space_time_astar(og, start, goal, 0, rt)
        assert result is not None, "A* failed to find a path through the wall gap"
        path_ijk, _ = result
        np.testing.assert_array_equal(path_ijk[-1].get(), goal.get())

    def test_blocked_goal_returns_none(self):
        """Occupied goal -> None."""
        og, rt = _make_og_and_table()
        og.grid[15, 15, 15] = True

        start = cp.array([2, 2, 2], dtype=cp.intp)
        goal = cp.array([15, 15, 15], dtype=cp.intp)

        result = space_time_astar(og, start, goal, 0, rt)
        assert result is None

    def test_avoids_reserved_cells(self):
        """A* detours around time-reserved cells."""
        grid = cp.zeros((10, 10, 1), dtype=cp.bool_)
        og = OccupancyGrid(grid=grid, origin=cp.zeros(3, dtype=cp.float64), resolution=1.0)
        rt = ReservationTable((10, 10, 1), 50, robot_collision_radius=0.0)

        start = cp.array([0, 5, 0], dtype=cp.intp)
        goal = cp.array([9, 5, 0], dtype=cp.intp)

        # Reserve the direct path at the times A* would traverse it
        for x in range(1, 9):
            positions = cp.array([[x, 5, 0]], dtype=cp.intp)
            times = cp.array([x], dtype=cp.intp)
            rt.commit_trajectory(positions, times)

        result = space_time_astar(og, start, goal, 0, rt)
        assert result is not None
        path_ijk, path_t = result
        path_ijk_np = path_ijk.get()
        path_t_np = path_t.get()
        np.testing.assert_array_equal(path_ijk_np[-1], goal.get())

        # Verify no path cell collides with reservations
        for i in range(len(path_ijk_np)):
            assert (
                not rt.is_reserved(
                    int(path_ijk_np[i, 0]),
                    int(path_ijk_np[i, 1]),
                    int(path_ijk_np[i, 2]),
                    int(path_t_np[i]),
                )
                or i == 0
            )  # start position at t=0 is not reserved

    def test_same_start_goal(self):
        """Returns single-point path when start == goal."""
        og, rt = _make_og_and_table()
        point = cp.array([5, 5, 5], dtype=cp.intp)

        result = space_time_astar(og, point, point, 0, rt)
        assert result is not None, "A* failed when start == goal"
        path_ijk, _ = result
        assert len(path_ijk) == 1
        np.testing.assert_array_equal(path_ijk[0].get(), point.get())
