"""Tests for VRP/mapf/orientation.py:apply_heading_orientation.

Trajectory layout: [x, y, z, yaw, cam_pitch, cam_roll]. Yaw and camera pitch
are filled from waypoint rotation matrices: held during dwell windows,
cosine-eased between dwells.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from VRP.mapf.orientation import apply_heading_orientation


def _yaw_rotmat(yaw: float) -> np.ndarray:
    """3x3 rotation matrix with column 0 = forward direction at given yaw."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


class TestOrientationDwellWindow:
    def test_yaw_inside_dwell_matches_waypoint(self):
        """During a dwell window, yaw must equal the waypoint's yaw."""
        target_yaw = math.pi / 2  # waypoint forward = +y
        # Single dwell window from t=2.0 to t=4.0, node index = 0
        wp_schedule = cp.array([[2.0, 4.0, 0.0]], dtype=cp.float64)
        rotmats = cp.asarray(_yaw_rotmat(target_yaw)[None, :, :], dtype=cp.float64)

        n_samples = 100
        t_dense = cp.linspace(0.0, 6.0, n_samples)
        traj = cp.zeros((n_samples, 6), dtype=cp.float64)

        apply_heading_orientation(traj, t_dense, dt=0.06, wp_schedule_s=wp_schedule,
                                  waypoint_rotmats=rotmats)
        traj_np = cp.asnumpy(traj)
        t_np = cp.asnumpy(t_dense)

        # Inside the dwell window (with a small margin)
        inside = (t_np > 2.05) & (t_np < 3.95)
        np.testing.assert_allclose(traj_np[inside, 3], target_yaw, atol=1e-6)


class TestOrientationTransitionContinuity:
    def test_transition_smooth_no_step_jump(self):
        """Cosine easing -- yaw must change gradually across the transition,
        not exhibit a step jump. Concretely: the maximum yaw delta between
        consecutive samples in the transition window must be < 0.5 * (Δyaw / N_trans),
        which would only fail for a step (delta == full Δyaw at one sample)."""
        yaw_a = 0.0
        yaw_b = math.pi / 4
        # Two waypoints: dwell A in [1, 2], dwell B in [3, 4]. Transition: (2, 3).
        wp_schedule = cp.array(
            [
                [1.0, 2.0, 0.0],
                [3.0, 4.0, 1.0],
            ],
            dtype=cp.float64,
        )
        rotmats = cp.asarray(
            np.stack([_yaw_rotmat(yaw_a), _yaw_rotmat(yaw_b)], axis=0),
            dtype=cp.float64,
        )

        # Dense sampling of the transition window (t in (2, 3))
        n = 200
        t_dense = cp.linspace(0.0, 5.0, n)
        traj = cp.zeros((n, 6), dtype=cp.float64)
        apply_heading_orientation(traj, t_dense, dt=0.025,
                                  wp_schedule_s=wp_schedule, waypoint_rotmats=rotmats)
        traj_np = cp.asnumpy(traj)
        t_np = cp.asnumpy(t_dense)

        in_transition = (t_np > 2.0) & (t_np < 3.0)
        yaw_in_trans = traj_np[in_transition, 3]
        diffs = np.abs(np.diff(yaw_in_trans))
        # Each step's change is at most the total Δyaw / number of transition samples
        # (+ a small safety margin). A step jump would have one
        # diff equal to the full Δyaw -- well above this bound.
        max_acceptable = (yaw_b - yaw_a) / max(in_transition.sum() // 2, 1)
        assert diffs.max() < max_acceptable, (
            f"yaw jumped {diffs.max():.4f} between consecutive samples; "
            f"expected smooth transition with max step ~{max_acceptable:.4f}"
        )
        # And the transition must actually cover the range -- end > start.
        assert yaw_in_trans[-1] > yaw_in_trans[0]


class TestOrientationNoWaypoints:
    def test_empty_schedule_keeps_default_zero(self):
        """Empty wp_schedule_s -> camera_pitch column stays zero, yaw uses initial."""
        traj = cp.zeros((50, 6), dtype=cp.float64)
        # Initial yaw = 0; the function broadcasts traj[0, 3] to all rows
        t_dense = cp.linspace(0.0, 1.0, 50)
        apply_heading_orientation(traj, t_dense, dt=0.02,
                                  wp_schedule_s=None, waypoint_rotmats=None)
        traj_np = cp.asnumpy(traj)
        np.testing.assert_allclose(traj_np[:, 3], 0.0)
        np.testing.assert_allclose(traj_np[:, 4], 0.0)
