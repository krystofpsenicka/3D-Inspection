"""Tests for NPZ+JSON serialization of OccupancyGrid, SamplingOccupancyGrid, and ExecutionResult."""
import cupy as cp
import numpy as np
import pytest
from shared.occupancy_grid import OccupancyGrid
from visibility.sampling.utils.sampling_occupancy_grid import SamplingOccupancyGrid
from VRP.core.types import ExecutionResult
from VRP.core.serialization import save_solution, load_solution


class TestOccupancyGridSerialization:
    def test_npz_round_trip_all_fields(self, tmp_path):
        """SamplingOccupancyGrid round-trips raw_grid, filled_raw_grid, mesh_scale."""
        grid = cp.asarray(np.random.RandomState(42).random((8, 8, 8)) > 0.5)
        raw = cp.asarray(np.random.RandomState(43).random((8, 8, 8)) > 0.5)
        filled = cp.asarray(np.random.RandomState(44).random((8, 8, 8)) > 0.5)
        og = SamplingOccupancyGrid(
            grid=grid,
            origin=cp.array([1.0, 2.0, 3.0]),
            resolution=0.25,
            raw_grid=raw,
            filled_raw_grid=filled,
            mesh_scale=0.123,
        )
        path = str(tmp_path / "og.pkl")
        og.save(path)
        loaded = SamplingOccupancyGrid.load(path)
        assert cp.array_equal(loaded.grid, og.grid)
        assert cp.allclose(loaded.origin, og.origin)
        assert loaded.resolution == og.resolution
        assert cp.array_equal(loaded.raw_grid, og.raw_grid)
        assert cp.array_equal(loaded.filled_raw_grid, og.filled_raw_grid)
        assert loaded.mesh_scale == og.mesh_scale

    def test_npz_base_og_round_trip(self, tmp_path):
        """Base OccupancyGrid saves and loads only grid/origin/resolution."""
        og = OccupancyGrid(
            grid=cp.zeros((4, 4, 4), dtype=cp.bool_),
            origin=cp.zeros(3),
            resolution=1.0,
        )
        path = str(tmp_path / "og_base.pkl")
        og.save(path)
        loaded = OccupancyGrid.load(path)
        assert cp.array_equal(loaded.grid, og.grid)
        assert cp.allclose(loaded.origin, og.origin)
        assert loaded.resolution == og.resolution


class TestExecutionResultSerialization:
    def _make_exec_result(self, n_robots=2, steps=(10, 7)):
        """Create a test ExecutionResult with ragged trajectories."""
        rng = np.random.RandomState(42)
        positions = []
        velocities = []
        for i in range(n_robots):
            n_steps = steps[i] if i < len(steps) else 5
            positions.append([rng.randn(8).astype(np.float32) for _ in range(n_steps)])
            velocities.append([rng.randn(8).astype(np.float32) for _ in range(n_steps)])
        return ExecutionResult(
            all_traj_positions=positions,
            all_traj_velocities=velocities,
            all_waypoints=[[[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]], [[4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0]]],
            initial_positions=[np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])],
            joint_names=["x", "y", "z", "yaw", "pitch", "roll", "cam_yaw", "cam_pitch"],
            fail_counts=[0, 1],
        )

    def test_npz_round_trip(self, tmp_path):
        orig = self._make_exec_result()
        path = str(tmp_path / "exec.pkl")
        save_solution(orig, path)
        loaded = load_solution(path)

        assert len(loaded.all_traj_positions) == 2
        assert len(loaded.all_traj_positions[0]) == 10
        assert len(loaded.all_traj_positions[1]) == 7
        assert loaded.joint_names == orig.joint_names
        assert loaded.fail_counts == orig.fail_counts
        assert loaded.all_waypoints == orig.all_waypoints

        # Check array values match
        for r in range(2):
            for s in range(len(orig.all_traj_positions[r])):
                assert np.allclose(
                    loaded.all_traj_positions[r][s],
                    orig.all_traj_positions[r][s],
                )

    def test_empty_robot(self, tmp_path):
        """Robot with 0 trajectory steps."""
        result = ExecutionResult(
            all_traj_positions=[[], [np.zeros(8, dtype=np.float32)]],
            all_traj_velocities=[[], [np.zeros(8, dtype=np.float32)]],
            all_waypoints=[[], [[0, 0, 0, 1, 0, 0, 0]]],
            initial_positions=[np.zeros(3), np.ones(3)],
            joint_names=["a"],
            fail_counts=[0, 0],
        )
        path = str(tmp_path / "exec_empty.pkl")
        save_solution(result, path)
        loaded = load_solution(path)
        assert len(loaded.all_traj_positions[0]) == 0
        assert len(loaded.all_traj_positions[1]) == 1
