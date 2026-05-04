"""Tests for VRP/mapf/reservation_table.py.

Covers commit + query semantics, sphere inflation, grid-boundary safety,
and time-horizon enforcement. Split out of test_vrp.py for focus.
"""

from __future__ import annotations

import pytest

cp = pytest.importorskip("cupy")

from VRP.mapf.reservation_table import ReservationTable


def _make_table(shape=(10, 10, 10), T=20, radius=0.0):
    return ReservationTable(shape, T, robot_collision_radius=radius)


class TestReservationTable:
    def test_commit_and_query(self):
        rt = _make_table()
        positions = cp.array([[3, 4, 5], [3, 5, 5]], dtype=cp.intp)
        times = cp.array([0, 1], dtype=cp.intp)
        rt.commit_trajectory(positions, times)

        assert rt.is_reserved(3, 4, 5, 0) is True
        assert rt.is_reserved(3, 5, 5, 1) is True
        assert rt.is_reserved(3, 4, 5, 1) is False  # different time
        assert rt.is_reserved(0, 0, 0, 0) is False  # uncommitted cell

    def test_sphere_inflation(self):
        """Sphere with radius=1.75 covers all 27 cells in the 3x3x3 cube."""
        rt = _make_table(shape=(10, 10, 10), T=10, radius=1.75)
        positions = cp.array([[5, 5, 5]], dtype=cp.intp)
        times = cp.array([3], dtype=cp.intp)
        rt.commit_trajectory(positions, times)

        # All 27 cells in the 3x3x3 cube should be reserved
        # (corner distance = sqrt(3) ~ 1.73 < 1.75)
        count = 0
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    if rt.is_reserved(5 + dx, 5 + dy, 5 + dz, 3):
                        count += 1
        assert count == 27, f"Expected 27 reserved cells, got {count}"

        # Outside the sphere should be free
        assert rt.is_reserved(5, 5, 5, 4) is False  # different time
        assert rt.is_reserved(3, 5, 5, 3) is False  # outside sphere

    def test_boundary_no_crash(self):
        """Commit near grid boundary with inflation doesn't crash."""
        rt = _make_table(shape=(5, 5, 5), T=5, radius=2.5)
        # Point at corner: (0, 0, 0)  --  sphere extends outside grid
        positions = cp.array([[0, 0, 0]], dtype=cp.intp)
        times = cp.array([0], dtype=cp.intp)
        rt.commit_trajectory(positions, times)  # should not crash
        assert rt.is_reserved(0, 0, 0, 0) is True

    def test_time_beyond_horizon(self):
        """is_reserved(x,y,z, t>=T) returns False."""
        rt = _make_table(T=10)
        positions = cp.array([[5, 5, 5]], dtype=cp.intp)
        times = cp.array([5], dtype=cp.intp)
        rt.commit_trajectory(positions, times)

        assert rt.is_reserved(5, 5, 5, 5) is True
        assert rt.is_reserved(5, 5, 5, 10) is False  # at horizon
        assert rt.is_reserved(5, 5, 5, 100) is False  # way beyond
