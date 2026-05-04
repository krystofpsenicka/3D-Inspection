"""Tests for VRP/utils/collision.py:find_trajectory_collisions.

GPU-vectorized AABB collision detection between robot trajectories.
Covers parallel paths, head-on collisions, padding for unequal lengths,
single-robot self-collision (impossible), penetration sign, and pair
uniqueness across multiple robots. Split out of test_vrp.py.
"""

from __future__ import annotations

import pytest

cp = pytest.importorskip("cupy")

from VRP.core.constants import ROBOT_RADIUS
from VRP.utils.collision import find_trajectory_collisions


class TestTrajectoryCollisions:
    def test_no_collision_parallel_paths(self):
        """Two robots moving in parallel far apart produce no collisions."""
        T = 50
        traj_a = cp.array([[0.0, 0.0, float(t) * 0.1] for t in range(T)], dtype=cp.float32)
        traj_b = cp.array([[5.0, 5.0, float(t) * 0.1] for t in range(T)], dtype=cp.float32)
        collisions = find_trajectory_collisions([traj_a, traj_b])
        assert collisions == []

    def test_head_on_collision_detected(self):
        """Two robots crossing the same point detect overlap."""
        T = 20
        # Robot A moves along +X, Robot B moves along -X; they cross at x=0
        traj_a = cp.array([[float(t) - 10.0, 0.0, 0.0] for t in range(T)], dtype=cp.float32)
        traj_b = cp.array([[10.0 - float(t), 0.0, 0.0] for t in range(T)], dtype=cp.float32)
        collisions = find_trajectory_collisions([traj_a, traj_b])
        assert len(collisions) > 0
        # Verify tuple format: (step, robot_a, robot_b, penetration)
        for _step, ra, rb, pen in collisions:
            assert ra == 0 and rb == 1
            assert pen > 0.0

    def test_padding_shorter_trajectory(self):
        """Robots with different trajectory lengths are padded correctly."""
        # Robot A: 10 steps far away; Robot B: 5 steps far away
        traj_a = cp.array([[100.0, 0.0, 0.0]] * 10, dtype=cp.float32)
        traj_b = cp.array([[-100.0, 0.0, 0.0]] * 5, dtype=cp.float32)
        collisions = find_trajectory_collisions([traj_a, traj_b])
        assert collisions == []

    def test_single_robot_no_collision(self):
        """A single robot cannot collide with itself."""
        traj = cp.array([[0.0, 0.0, float(t)] for t in range(10)], dtype=cp.float32)
        collisions = find_trajectory_collisions([traj])
        assert collisions == []

    def test_collision_penetration_sign(self):
        """Penetration depth = (2*ROBOT_RADIUS - distance), strictly positive on overlap."""
        # Two robots held a fixed distance r = ROBOT_RADIUS apart for T steps
        # (so 2r overlap = ROBOT_RADIUS).  Pure x-axis separation, no z motion.
        T = 5
        r = ROBOT_RADIUS  # (one radius apart in x)
        traj_a = cp.tile(cp.array([0.0, 0.0, 0.0], dtype=cp.float32), (T, 1))
        traj_b = cp.tile(cp.array([r, 0.0, 0.0], dtype=cp.float32), (T, 1))
        collisions = find_trajectory_collisions([traj_a, traj_b])
        assert len(collisions) == T  # one collision per step
        for _step, ra, rb, pen in collisions:
            assert ra == 0 and rb == 1
            # Expected penetration = 2*ROBOT_RADIUS - r = ROBOT_RADIUS
            assert pen == pytest.approx(ROBOT_RADIUS, abs=1e-4)

    def test_collision_pair_uniqueness(self):
        """3 robots overlapping at one step -> exactly 3 unique pairs (a<b)."""
        # 3 robots all at the origin for one step, then far apart
        traj_a = cp.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=cp.float32)
        traj_b = cp.array([[0.0, 0.0, 0.0], [200.0, 0.0, 0.0]], dtype=cp.float32)
        traj_c = cp.array([[0.0, 0.0, 0.0], [300.0, 0.0, 0.0]], dtype=cp.float32)
        collisions = find_trajectory_collisions([traj_a, traj_b, traj_c])
        # Step 0 collisions only, 3 unique pairs: (0,1) (0,2) (1,2)
        pairs = {(ra, rb) for _step, ra, rb, _pen in collisions}
        assert pairs == {(0, 1), (0, 2), (1, 2)}
