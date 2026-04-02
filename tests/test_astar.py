"""Tests for CPU A* pathfinding."""
import numpy as np
import pytest
from VRP.core.astar import (
    astar_distance as _astar_single,
    astar_path as extract_astar_path,
)


class TestAstarSingle:
    def test_same_start_goal(self):
        grid = np.zeros((5, 5, 5), dtype=bool)
        d = _astar_single(grid, np.array([2, 2, 2]), np.array([2, 2, 2]), 1.0)
        assert d == 0.0

    def test_adjacent_face(self):
        grid = np.zeros((5, 5, 5), dtype=bool)
        d = _astar_single(grid, np.array([2, 2, 2]), np.array([2, 2, 3]), 1.0)
        assert abs(d - 1.0) < 1e-6

    def test_adjacent_diagonal(self):
        grid = np.zeros((5, 5, 5), dtype=bool)
        d = _astar_single(grid, np.array([2, 2, 2]), np.array([2, 3, 3]), 1.0)
        assert abs(d - np.sqrt(2)) < 1e-6

    def test_start_occupied(self):
        grid = np.zeros((5, 5, 5), dtype=bool)
        grid[2, 2, 2] = True
        d = _astar_single(grid, np.array([2, 2, 2]), np.array([3, 3, 3]), 1.0)
        assert d == np.inf

    def test_goal_occupied(self):
        grid = np.zeros((5, 5, 5), dtype=bool)
        grid[3, 3, 3] = True
        d = _astar_single(grid, np.array([2, 2, 2]), np.array([3, 3, 3]), 1.0)
        assert d == np.inf

    def test_through_corridor(self, corridor_og):
        """Path must go through the gap — distance >= Euclidean and finite."""
        start = np.array([2, 2, 2])
        goal = np.array([8, 8, 8])
        d = _astar_single(corridor_og.grid, start, goal, corridor_og.resolution)
        euclidean = np.linalg.norm(goal - start) * corridor_og.resolution
        assert d >= euclidean - 1e-6
        assert d < np.inf

    def test_no_path(self):
        """Grid walled off completely — should return inf."""
        grid = np.zeros((7, 7, 7), dtype=bool)
        grid[3, :, :] = True  # solid wall, no gap
        d = _astar_single(grid, np.array([1, 3, 3]), np.array([5, 3, 3]), 1.0)
        assert d == np.inf


class TestDistanceMatrixCPU:
    def test_symmetric(self):
        grid = np.zeros((10, 10, 10), dtype=bool)
        origin = np.zeros(3)
        pts = np.array([[1.5, 1.5, 1.5], [3.5, 3.5, 3.5], [7.5, 7.5, 7.5]])
        dm = compute_distance_matrix_cpu(grid, origin, 1.0, pts)
        assert np.allclose(dm, dm.T)

    def test_diagonal_zero(self):
        grid = np.zeros((10, 10, 10), dtype=bool)
        origin = np.zeros(3)
        pts = np.array([[1.5, 1.5, 1.5], [5.5, 5.5, 5.5]])
        dm = compute_distance_matrix_cpu(grid, origin, 1.0, pts)
        assert dm[0, 0] == 0.0
        assert dm[1, 1] == 0.0

    def test_triangle_inequality(self):
        grid = np.zeros((10, 10, 10), dtype=bool)
        origin = np.zeros(3)
        pts = np.array([[1.5, 1.5, 1.5], [3.5, 3.5, 3.5], [7.5, 7.5, 7.5]])
        dm = compute_distance_matrix_cpu(grid, origin, 1.0, pts)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    assert dm[i, j] <= dm[i, k] + dm[k, j] + 1e-6


class TestExtractAstarPath:
    def test_start_equals_goal(self, small_og):
        pt = small_og.voxel_to_world(np.array([0, 0, 0]))
        path = extract_astar_path(small_og, pt, pt)
        assert len(path) >= 1
        assert np.allclose(path[0], pt, atol=small_og.resolution)

    def test_path_endpoints(self, small_og):
        start = small_og.voxel_to_world(np.array([0, 0, 0]))
        goal = small_og.voxel_to_world(np.array([19, 19, 19]))
        path = extract_astar_path(small_og, start, goal)
        assert len(path) >= 2
        assert np.allclose(path[0], start, atol=small_og.resolution)
        assert np.allclose(path[-1], goal, atol=small_og.resolution)

    def test_all_points_free(self, small_og):
        start = small_og.voxel_to_world(np.array([0, 0, 0]))
        goal = small_og.voxel_to_world(np.array([19, 0, 0]))
        path = extract_astar_path(small_og, start, goal)
        for pt in path:
            assert small_og.is_free_world(pt), f"Point {pt} is not free"
