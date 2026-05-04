"""Tests for VRP/core/geometry.py."""

from __future__ import annotations

import math

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from VRP.core.constants import CAMERA_OFFSET_FORWARD, CAMERA_OFFSET_UP
from VRP.core.geometry import compute_start_grid, viewpoints_to_robot_waypoints


class TestComputeStartGrid:
    def test_centered_above_mesh(self):
        bounds_min = np.array([-2.0, -2.0, -1.0])
        bounds_max = np.array([2.0, 2.0, 1.0])
        starts = compute_start_grid(
            num_robots=4,
            mesh_bounds_min=bounds_min,
            mesh_bounds_max=bounds_max,
            z_above=1.0,
            spacing=1.5,
        )
        starts_arr = np.stack(starts)
        # XY centroid should match the mesh's XY centre
        assert starts_arr[:, 0].mean() == pytest.approx(0.0, abs=1e-6)
        assert starts_arr[:, 1].mean() == pytest.approx(0.0, abs=1e-6)
        # Every Z should equal mesh_bounds_max[2] + z_above = 1 + 1 = 2
        np.testing.assert_allclose(starts_arr[:, 2], 2.0)

    @pytest.mark.parametrize("num_robots", [1, 5, 7, 16])
    def test_count_matches(self, num_robots):
        starts = compute_start_grid(
            num_robots=num_robots,
            mesh_bounds_min=np.array([-1.0, -1.0, -1.0]),
            mesh_bounds_max=np.array([1.0, 1.0, 1.0]),
        )
        assert len(starts) == num_robots


class TestViewpointsToRobotWaypoints:
    def test_offset_along_forward_x(self):
        """Forward = (1,0,0): body offset = -CAMERA_OFFSET_FORWARD on x."""
        positions = cp.array([[5.0, 0.0, 1.0]], dtype=cp.float32)
        rotmats = cp.zeros((1, 3, 3), dtype=cp.float32)
        rotmats[0, 0, 0] = 1.0  # forward = +x
        rotmats[0, 1, 1] = 1.0
        rotmats[0, 2, 2] = 1.0
        result = cp.asnumpy(viewpoints_to_robot_waypoints(positions, rotmats, set()))
        assert result[0, 0] == pytest.approx(5.0 - CAMERA_OFFSET_FORWARD, abs=1e-5)
        assert result[0, 1] == pytest.approx(0.0, abs=1e-5)
        assert result[0, 2] == pytest.approx(1.0 - CAMERA_OFFSET_UP, abs=1e-5)

    def test_offset_along_forward_y(self):
        """Forward = (0,1,0): body offset = -CAMERA_OFFSET_FORWARD on y."""
        positions = cp.array([[0.0, 5.0, 1.0]], dtype=cp.float32)
        rotmats = cp.zeros((1, 3, 3), dtype=cp.float32)
        rotmats[0, 1, 0] = 1.0  # forward = +y
        rotmats[0, 0, 1] = -1.0  # right = -x
        rotmats[0, 2, 2] = 1.0
        result = cp.asnumpy(viewpoints_to_robot_waypoints(positions, rotmats, set()))
        assert result[0, 0] == pytest.approx(0.0, abs=1e-5)
        assert result[0, 1] == pytest.approx(5.0 - CAMERA_OFFSET_FORWARD, abs=1e-5)
        assert result[0, 2] == pytest.approx(1.0 - CAMERA_OFFSET_UP, abs=1e-5)

    def test_home_indices_unchanged(self):
        """Home nodes must be returned with no offset applied."""
        positions = cp.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=cp.float32,
        )
        rotmats = cp.tile(cp.eye(3, dtype=cp.float32), (3, 1, 1))
        result = cp.asnumpy(viewpoints_to_robot_waypoints(positions, rotmats, {0, 2}))
        np.testing.assert_allclose(result[0], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(result[2], [7.0, 8.0, 9.0])
        # Index 1 is not a home -> identity rotation, forward = +x, up = +z
        assert result[1, 0] == pytest.approx(4.0 - CAMERA_OFFSET_FORWARD, abs=1e-5)
        assert result[1, 1] == pytest.approx(5.0, abs=1e-5)
        assert result[1, 2] == pytest.approx(6.0 - CAMERA_OFFSET_UP, abs=1e-5)
