"""Tests for VRP/mapf/route_planner.py:plan_robot_route_st.

Most of the planner's components (Space-Time A*, ReservationTable) have
their own tests in test_vrp.py. The integration property worth verifying
here is that the full planner produces collision-free trajectories
end-to-end on a non-trivial obstacle.
"""

from __future__ import annotations

import pytest

cp = pytest.importorskip("cupy")

from VRP.mapf.reservation_table import ReservationTable
from VRP.mapf.route_planner import plan_robot_route_st


class TestPlanRobotRoute:
    def test_planned_route_is_collision_free(self, corridor_og):
        """Plan through 3 waypoints that force the robot through the wall gap.

        Every returned position must lie in a free voxel of the OG.
        """
        # Three waypoints; the path between them must traverse the gap at (5,5,5).
        waypoint_positions = cp.array(
            [
                [0.5, 5.5, 5.5],  # left of wall
                [9.5, 5.5, 5.5],  # right of wall (must cross gap)
                [0.5, 5.5, 0.5],  # back across (must cross gap again)
            ],
            dtype=cp.float64,
        )
        route = [0, 1, 2]

        # Reservation table for a single robot (no other commitments)
        reservation = ReservationTable(
            corridor_og.shape, max_time_steps=200, robot_collision_radius=0.0
        )

        world_xyz, _coarse_t, wp_schedule, _stats = plan_robot_route_st(
            corridor_og,
            reservation,
            route,
            waypoint_positions,
            dwell_s=1.0,
            fine_occupancy_grid=None,  # skip OMPL smoothing
            robot_radius=0.0,
        )

        assert len(world_xyz) > 0, "Planner returned an empty trajectory"

        # Every world position must be in a free voxel of the corridor OG
        free_mask = corridor_og.is_free_world_batch(world_xyz)
        n_collisions = int(cp.sum(~free_mask))
        assert n_collisions == 0, (
            f"{n_collisions}/{len(world_xyz)} planner samples are inside the wall"
        )

        # The schedule must contain a dwell window for each non-start route node
        assert len(wp_schedule) == len(route) - 1
