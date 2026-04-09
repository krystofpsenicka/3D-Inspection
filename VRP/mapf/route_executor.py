"""
VRP Planner – Route Executor (Priority-Based Sequential Planning)

Builds per-vehicle trajectories that are **collision-free by construction**.
Robots are planned one at a time in priority order (longest route first).
Each robot's route is planned via Space-Time A* on a coarse 4-D grid so it
avoids both static obstacles *and* previously committed robots'
trajectories.  The coarse space-time path is then densified at the replay
time-step (``TRAJ_DT``) and assigned heading orientation.

Pipeline
--------
1. Down-sample occupancy grid -> coarse grid for Space-Time A*.
2. Initialize a dense 4-D reservation table.
3. Sort robots by estimated route cost (longest first).
4. For each robot (priority order):
   a.  Plan each route leg with Space-Time A*, building a coarse
       time-stamped XYZ path.
   b.  Commit the trajectory to the reservation table so later robots
       avoid it.
5. Densify coarse paths to replay resolution and assign orientation.
6. Optionally run collision safety checks.

References
----------
Silver, D. (2005). Cooperative Pathfinding. AIIDE.
    Priority-based sequential planning framework: robots are planned
    one at a time, each committing its trajectory to a reservation
    table so that later robots avoid it.

Erdmann, M. & Lozano-Perez, T. (1987). On Multiple Moving Objects.
    Algorithmica, 2(1), 477-521.
    Foundational work on planning in configuration-space x time;
    the 4-D space-time grid with a reservation table derives from
    this lineage.

Zhou, Y. & Zeng, J. (2015). Massively Parallel A* Search on a GPU.
    AAAI.
    GPU-parallel frontier expansion via heuristic-guided threshold
    selection (expanding nodes where f <= f_min + delta).

Li, Z. et al. (2025). GPU-accelerated Conflict-based Search for
    Multi-agent Embodied Intelligence. Machine Intelligence Research.
    GPU-parallel frontier expansion for multi-agent pathfinding
    (GATSA algorithm).

Design note — why not Conflict-Based Search (CBS)?
---------------------------------------------------
CBS (Sharon et al., 2015) is an optimal MAPF solver: it searches a
conflict tree where each node represents a set of inter-agent
constraints, splitting on the first detected collision and re-planning
only the affected agent.  This guarantees the shortest-makespan
solution but at significant computational cost — CBS is exponential in
the number of conflicts, and each conflict-tree node triggers a full
single-agent A* re-plan.

In this application the priority-based approach is preferred because:

1. **Low robot density.**  A small fleet (typically 2-5 robots)
   operates in a large 3-D volume (~80 K free voxels, i.e. ~16 K
   voxels per robot at 5 agents).  Spatial conflicts are inherently
   sparse, so the optimality gap between prioritised planning and
   CBS is negligible in practice.

2. **Priority-order search already covers the gap.**  The executor
   tries multiple priority orderings (longest-first, shortest-first,
   most-conflicted-first, plus random permutations) and keeps the
   best result.  For 5 robots there are only 5! = 120 possible
   orderings; the default budget of 20 trials samples ~17 % of
   them, which is sufficient to find a near-optimal ordering.

3. **The real bottleneck is VRP assignment, not conflict resolution.**
   A suboptimal conflict resolution adds seconds of detour; a
   suboptimal waypoint-to-vehicle assignment adds minutes of extra
   travel.  Investing computation in CBS yields diminishing returns
   when the upstream VRP solution dominates total cost.

4. **CBS worst case is exponential.**  If robots frequently cross
   paths (e.g. star-shaped routes through a central corridor) the
   constraint tree can blow up.  The 4-D state space (x, y, z, t)
   with 26-connected + wait already has a large branching factor;
   CBS re-solves A* over this grid for every conflict-tree branch.
   Priority-based planning degrades gracefully under the same
   conditions.

CBS would become worthwhile at 15-30+ robots or in highly constrained
environments (narrow corridors, bottlenecks) where priority ordering
causes significant cascading delays.  At the current scale — a small
fleet in an open ship hull - the gain is not worth the complexity.

Sharon, G. et al. (2015). Conflict-Based Search for Optimal
    Multi-Agent Pathfinding. Artificial Intelligence, 219, 40-66.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

import cupy as cp
import numpy as np

from ..core.constants import (
    ROBOT_RADIUS,
    SPACE_TIME_DT,
    SPACE_TIME_DWELL_S,
    SPACE_TIME_MAX_HORIZON_S,
    SPACE_TIME_RESOLUTION,
    SPLINE_SAFETY_VOXELS,
    TRAJ_DT,
)
from ..core.types import ExecutionResult
from ..vrp._helpers import per_vehicle_costs as _per_vehicle_costs
from shared.grid_utils import downsample_occupancy_grid
from shared.occupancy_grid import OccupancyGrid
from .reservation_table import ReservationTable
from .route_planner import plan_robot_route_st
from .orientation import apply_heading_orientation

logger = logging.getLogger(__name__)


# ─── Executor ────────────────────────────────────────────────────────────────

class RouteExecutor:
    """Priority-Based Sequential Planner from Space-Time A* paths.

    Parameters
    ----------
    start_positions:
        Per-robot initial XYZ positions, each ``(3,)`` numpy array.
    og:
        Fine-resolution occupancy grid for collision checks.
    """

    def __init__(
        self,
        start_positions: List[np.ndarray],
        og: OccupancyGrid | None = None,
    ):
        self.num_robots = len(start_positions)
        self.start_positions = start_positions
        self.og = og

    def _plan_sequential(
        self,
        routes: List[List[int]],
        waypoint_positions: cp.ndarray,
        priority_order: List[int],
        coarse_og: OccupancyGrid,
        max_time_steps: int,
        collision_radius_vox: float,
        dwell_seconds: float,
    ):
        """Plan all robots sequentially in given priority order.

        Returns
        -------
        (robot_world_paths, robot_coarse_times, robot_waypoint_schedules,
         makespan_seconds, total_travel_seconds, per_robot_stats)
        """
        from ..core.types import PlanningStats

        num_robots = self.num_robots
        reservation = ReservationTable(
            coarse_og.shape, max_time_steps, collision_radius_vox,
        )

        robot_world_paths: List[Optional[cp.ndarray]] = [None] * num_robots
        robot_coarse_times: List[Optional[cp.ndarray]] = [None] * num_robots
        robot_waypoint_schedules: List[Optional[list]] = [None] * num_robots
        per_robot_stats: List[PlanningStats] = [
            PlanningStats() for _ in range(num_robots)
        ]

        for priority, robot_idx in enumerate(priority_order):
            route = routes[robot_idx]
            world_positions, coarse_time_steps, waypoint_schedule, stats = (
                plan_robot_route_st(
                    coarse_og,
                    reservation, route,
                    waypoint_positions,
                    dwell_s=dwell_seconds,
                    dt=SPACE_TIME_DT,
                    fine_occupancy_grid=self.og,
                    robot_radius=ROBOT_RADIUS,
                )
            )
            robot_world_paths[robot_idx] = world_positions
            robot_coarse_times[robot_idx] = coarse_time_steps
            robot_waypoint_schedules[robot_idx] = waypoint_schedule
            per_robot_stats[robot_idx] = stats

        # Per-vehicle travel times
        last_steps = cp.array([
            float(robot_coarse_times[i][-1])
            if robot_coarse_times[i] is not None and len(robot_coarse_times[i]) > 0
            else 0.0
            for i in range(num_robots)
        ])
        per_vehicle_seconds = (last_steps * SPACE_TIME_DT).tolist()
        makespan_seconds = float(last_steps.max() * SPACE_TIME_DT) if num_robots > 0 else 0.0
        total_travel_seconds = float(last_steps.sum() * SPACE_TIME_DT)

        return (
            robot_world_paths, robot_coarse_times, robot_waypoint_schedules,
            makespan_seconds, total_travel_seconds, per_robot_stats,
        )

    # ──────────────────────────────────────────────────────────────────

    def execute(
        self,
        routes: List[List[int]],
        waypoint_positions: cp.ndarray,
        waypoint_rotmats: cp.ndarray,
        home_indices: set,
        dist_matrix: cp.ndarray,
        alpha: float = 1.0,
        dwell_seconds: float = SPACE_TIME_DWELL_S,
        n_priority_trials: int = 20,
        run_collision_checks: bool = False,
    ) -> ExecutionResult:
        """Build collision-free trajectories via Priority-Based Sequential
        Planning and return an :class:`ExecutionResult`.

        Parameters
        ----------
        waypoint_positions : cp.ndarray (N, 3)
            World-frame positions for all VRP nodes (homes + inspection).
        waypoint_rotmats : cp.ndarray (N, 3, 3)
            Rotation matrices for all VRP nodes (identity for homes).
        dist_matrix : cp.ndarray (N, N)
            Distance matrix used for route cost estimation.
        alpha : float
            Objective blending in [0, 1]. 1.0 = pure makespan,
            0.0 = pure total travel time. Must match the VRP objective.
        n_priority_trials : int
            Total budget of priority orderings to try.
        run_collision_checks : bool
            If True, run environment and inter-robot collision checks
            on the densified trajectories before returning.
        """
        num_robots = self.num_robots

        initial_positions = [pos.copy() for pos in self.start_positions]

        # ── 1. Coarse grid + reservation table ────────────────────────
        if self.og is not None:
            coarse_og = downsample_occupancy_grid(
                self.og, coarse_res=SPACE_TIME_RESOLUTION,
            )
        else:
            logger.warning("No occupancy grid; collision avoidance with "
                           "static obstacles is disabled.")
            coarse_og = OccupancyGrid(
                grid=cp.zeros((10, 10, 10), dtype=cp.bool_),
                origin=cp.zeros(3, dtype=cp.float64),
                resolution=SPACE_TIME_RESOLUTION,
            )

        max_time_steps = max(
            1, int(math.ceil(SPACE_TIME_MAX_HORIZON_S / SPACE_TIME_DT))
        )
        collision_radius_vox = ROBOT_RADIUS / coarse_og.resolution + SPLINE_SAFETY_VOXELS

        logger.info("[RouteExecutor] Coarse grid %s  T=%d  collision_radius_vox=%.2f",
                    coarse_og.shape, max_time_steps, collision_radius_vox)

        # ── 2. Estimate route costs for priority ordering ────────────
        # Routes are [home, c1, ..., ck, home]; strip depots for helper
        customer_routes = [route[1:-1] for route in routes]
        depot_indices = [route[0] for route in routes]
        route_costs = _per_vehicle_costs(customer_routes, dist_matrix, depot_indices)

        default_order = sorted(
            range(num_robots), key=lambda i: -route_costs[i]
        )
        logger.info("[RouteExecutor] Default priority (longest first): %s  "
                    "costs=%s",
                    default_order,
                    [f"{route_costs[i]:.1f}" for i in default_order])

        def _combined_objective(makespan: float, total_time: float) -> float:
            return alpha * makespan + (1 - alpha) * total_time

        # ── 3. Try priority orderings, keep best ─────────────────────
        def _run_trial(order: List[int]) -> tuple:
            result = self._plan_sequential(
                routes, waypoint_positions,
                order, coarse_og,
                max_time_steps, collision_radius_vox, dwell_seconds,
            )
            makespan = result[3]
            total_time = result[4]
            objective = _combined_objective(makespan, total_time)
            return result, objective

        best_result, best_objective = _run_trial(default_order)
        best_order = default_order
        best_stats = best_result[5]
        logger.info("[RouteExecutor] Default order (longest-first) "
                    "objective: %.1f  makespan: %.1f s",
                    best_objective, best_result[3])

        trials_used = 1

        if n_priority_trials > trials_used:
            reverse_order = list(reversed(default_order))
            trial_result, trial_objective = _run_trial(reverse_order)
            logger.info("[RouteExecutor] Shortest-first order "
                        "objective: %.1f  makespan: %.1f s",
                        trial_objective, trial_result[3])
            if trial_objective < best_objective:
                best_result = trial_result
                best_objective = trial_objective
                best_order = reverse_order
                best_stats = trial_result[5]
            trials_used += 1

        if n_priority_trials > trials_used:
            conflict_scores = [
                s.wait_steps + s.astar_failures * 100
                for s in best_stats
            ]
            conflict_order = sorted(
                range(num_robots), key=lambda i: -conflict_scores[i],
            )
            if conflict_order != default_order:
                trial_result, trial_objective = _run_trial(conflict_order)
                logger.info("[RouteExecutor] Most conflicted order %s "
                            "objective: %.1f  makespan: %.1f s",
                            conflict_order, trial_objective, trial_result[3])
                if trial_objective < best_objective:
                    best_result = trial_result
                    best_objective = trial_objective
                    best_order = conflict_order
                    best_stats = trial_result[5]
            trials_used += 1

        if n_priority_trials > trials_used:
            conflict_scores = [
                s.wait_steps + s.astar_failures * 100
                for s in best_stats
            ]
            least_conflict_order = sorted(
                range(num_robots), key=lambda i: conflict_scores[i],
            )
            if least_conflict_order != default_order:
                trial_result, trial_objective = _run_trial(
                    least_conflict_order,
                )
                logger.info("[RouteExecutor] Least conflicted order %s "
                            "objective: %.1f  makespan: %.1f s",
                            least_conflict_order, trial_objective,
                            trial_result[3])
                if trial_objective < best_objective:
                    best_result = trial_result
                    best_objective = trial_objective
                    best_order = least_conflict_order
                    best_stats = trial_result[5]
            trials_used += 1

        for trial in range(trials_used, n_priority_trials):
            random_order = list(np.random.permutation(num_robots))
            trial_result, trial_objective = _run_trial(random_order)
            logger.info("[RouteExecutor] Random trial %d order %s "
                        "objective: %.1f  makespan: %.1f s",
                        trial + 1, random_order,
                        trial_objective, trial_result[3])
            if trial_objective < best_objective:
                best_result = trial_result
                best_objective = trial_objective
                best_order = random_order
                best_stats = trial_result[5]

        logger.info("[RouteExecutor] Best priority order: %s  "
                    "objective=%.1f  makespan=%.1f s  (%d trials)",
                    best_order, best_objective, best_result[3],
                    n_priority_trials)

        (robot_world_paths, robot_coarse_times, robot_waypoint_schedules,
         _, _, _per_robot_stats) = best_result

        for robot_idx in best_order:
            world_positions = robot_world_paths[robot_idx]
            coarse_times = robot_coarse_times[robot_idx]
            if world_positions is not None and len(world_positions) > 0:
                logger.info("  Robot %d: %d coarse samples, t=[%d..%d] "
                            "(%.1f s)",
                            robot_idx, len(world_positions),
                            int(coarse_times[0]), int(coarse_times[-1]),
                            float(coarse_times[-1]) * SPACE_TIME_DT)

        # ── 4. Densify to replay resolution and assign orientation ────
        # All computation stays on GPU; per-robot CuPy arrays
        all_trajectories_gpu: List[cp.ndarray] = []
        all_velocities_gpu: List[cp.ndarray] = []
        fail_counts = [0] * num_robots

        for robot_idx in range(num_robots):
            world_positions_gpu = robot_world_paths[robot_idx]
            coarse_time_steps_gpu = robot_coarse_times[robot_idx]

            waypoint_schedule = robot_waypoint_schedules[robot_idx] or []
            waypoint_schedule_seconds = [
                (int(ts) * SPACE_TIME_DT, int(te) * SPACE_TIME_DT, node)
                for ts, te, node in waypoint_schedule
            ]

            if world_positions_gpu is None or len(world_positions_gpu) == 0:
                fail_counts[robot_idx] = len(routes[robot_idx]) - 1
                home_trajectory = cp.zeros((50, 6), dtype=cp.float32)
                home_trajectory[:, :3] = cp.asarray(self.start_positions[robot_idx])
                all_trajectories_gpu.append(home_trajectory)
                all_velocities_gpu.append(cp.zeros((50, 6), dtype=cp.float32))
                continue

            time_seconds = coarse_time_steps_gpu.astype(cp.float64) * SPACE_TIME_DT
            t_start = float(time_seconds[0])
            t_end = float(time_seconds[-1])
            duration = t_end - t_start

            if duration < 1e-6:
                trajectory = cp.zeros((1, 6), dtype=cp.float64)
                trajectory[0, :3] = world_positions_gpu[-1]
                dense_time_samples = cp.array([t_start])
            else:
                num_dense_steps = max(2, int(math.ceil(duration / TRAJ_DT)))
                dense_time_samples = cp.linspace(t_start, t_end, num_dense_steps)
                unique_times, unique_indices = cp.unique(
                    time_seconds, return_index=True,
                )
                unique_indices = cp.sort(unique_indices)
                unique_times = time_seconds[unique_indices]
                unique_positions = world_positions_gpu[unique_indices]
                interpolated_positions = cp.column_stack([
                    cp.interp(dense_time_samples, unique_times, unique_positions[:, d])
                    for d in range(3)
                ])
                trajectory = cp.zeros((num_dense_steps, 6), dtype=cp.float64)
                trajectory[:, :3] = interpolated_positions

            apply_heading_orientation(
                trajectory,
                t_dense=dense_time_samples,
                dt=TRAJ_DT,
                wp_schedule_s=waypoint_schedule_seconds,
                waypoint_rotmats=waypoint_rotmats,
            )

            velocities = cp.zeros_like(trajectory)
            if len(trajectory) > 1:
                velocities[:-1] = cp.diff(trajectory, axis=0) / TRAJ_DT

            all_trajectories_gpu.append(trajectory.astype(cp.float32))
            all_velocities_gpu.append(velocities.astype(cp.float32))

        # ── 5. Build per-robot waypoint list ──────────────────────────
        waypoint_positions_np = cp.asnumpy(waypoint_positions)
        all_waypoints = [
            [waypoint_positions_np[node].tolist() for node in route]
            for route in routes
        ]

        # ── 6. Optional collision safety checks (on GPU) ─────────────
        if run_collision_checks:
            from ..core.collision import (
                find_environment_collisions,
                find_trajectory_collisions,
            )

            if self.og is not None:
                env_collision_counts = find_environment_collisions(
                    all_trajectories_gpu, self.og,
                )
                for robot_idx, count in enumerate(env_collision_counts):
                    if count > 0:
                        logger.warning(
                            "[RouteExecutor] Robot %d: %d / %d trajectory "
                            "steps collide with fine OG.",
                            robot_idx, count, len(all_trajectories_gpu[robot_idx]),
                        )

            inter_robot_collisions = find_trajectory_collisions(
                all_trajectories_gpu,
            )
            if inter_robot_collisions:
                logger.warning(
                    "[RouteExecutor] %d inter-robot collision events detected.",
                    len(inter_robot_collisions),
                )

        # ── 7. Transfer to CPU for ExecutionResult ────────────────────
        all_trajectory_positions: List[List[np.ndarray]] = []
        all_trajectory_velocities: List[List[np.ndarray]] = []
        for robot_idx in range(num_robots):
            traj_np = cp.asnumpy(all_trajectories_gpu[robot_idx])
            vel_np = cp.asnumpy(all_velocities_gpu[robot_idx])
            all_trajectory_positions.append(list(traj_np))
            all_trajectory_velocities.append(list(vel_np))

        actual_per_vehicle_times = [
            len(all_trajectories_gpu[i]) * TRAJ_DT for i in range(num_robots)
        ]
        actual_makespan = (
            max(actual_per_vehicle_times) if actual_per_vehicle_times else 0.0
        )
        logger.info("[RouteExecutor] Actual makespan: %.1f s  per_vehicle: %s",
                    actual_makespan,
                    [f"{t:.1f}" for t in actual_per_vehicle_times])

        return ExecutionResult(
            all_traj_positions=all_trajectory_positions,
            all_traj_velocities=all_trajectory_velocities,
            all_waypoints=all_waypoints,
            initial_positions=initial_positions,
            fail_counts=fail_counts,
            actual_makespan=actual_makespan,
            actual_per_vehicle_times=actual_per_vehicle_times,
        )
