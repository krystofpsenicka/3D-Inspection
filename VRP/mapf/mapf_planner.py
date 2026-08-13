"""
VRP Planner - MAPF Planner (Priority-Based Sequential Planning)

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
"""

from __future__ import annotations

import itertools
import logging
import math
import random

import cupy as cp
import numpy as np

from shared.grid_utils import downsample_occupancy_grid
from shared.occupancy_grid import OccupancyGrid

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
from .orientation import apply_heading_orientation
from .committed_motion import CommittedMotion
from .route_planner import plan_robot_route_st

logger = logging.getLogger(__name__)


def _perm_to_index(perm: list) -> int:
    """Factoradic (Lehmer code) index of a permutation of ``range(n)``."""
    n = len(perm)
    available = list(range(n))
    idx = 0
    for i, v in enumerate(perm):
        j = available.index(v)
        available.pop(j)
        idx += j * math.factorial(n - 1 - i)
    return idx


def _index_to_perm(idx: int, n: int) -> list:
    """Permutation of ``range(n)`` corresponding to a factoradic index."""
    available = list(range(n))
    perm: list = []
    for i in range(n, 0, -1):
        f = math.factorial(i - 1)
        j, idx = divmod(idx, f)
        perm.append(available.pop(j))
    return perm


def _per_leg_travel_times(wp_schedule, dt):
    """Travel-only time per leg (dwell excluded), in seconds.

    wp_schedule: list of (t_dwell_start, t_dwell_end, node_idx) in coarse
    timesteps, in route order, one entry per non-start node.
    Each leg's travel time is the gap between the end of the previous
    dwell (or t=0 for the first leg) and the start of this leg's dwell.
    Returns list[float] in seconds, one entry per wp_schedule entry.
    """
    times, prev_end = [], 0
    for t_dwell_start, t_dwell_end, _ in wp_schedule:
        times.append(max(0.0, (t_dwell_start - prev_end) * dt))
        prev_end = t_dwell_end
    return times


# ─── Executor ────────────────────────────────────────────────────────────────


class MultiAgentPathPlanner:
    """Priority-Based Sequential Planner from Space-Time A* paths.

    Parameters
    ----------
    start_positions:
        Per-robot initial XYZ positions, each ``(3,)`` CuPy or numpy array.
    og:
        Fine-resolution occupancy grid for collision checks.
    """

    def __init__(
        self,
        start_positions: list[cp.ndarray] | list[np.ndarray],
        og: OccupancyGrid | None = None,
    ):
        self.num_robots = len(start_positions)
        self.start_positions = [cp.asarray(p, dtype=cp.float32) for p in start_positions]
        self.og = og

    def _plan_sequential(
        self,
        routes: list[list[int]],
        waypoint_positions: cp.ndarray,
        priority_order: list[int],
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
        # Continuous-time inter-robot avoidance (CCBS-style). Static obstacles
        # stay on coarse_og; collision_radius_vox is no longer used for
        # inter-robot separation (the geometric test uses the true 2*radius).
        committed = CommittedMotion(max_time_steps, ROBOT_RADIUS)

        robot_world_paths: list[cp.ndarray | None] = [None] * num_robots
        robot_coarse_times: list[cp.ndarray | None] = [None] * num_robots
        robot_waypoint_schedules: list[list | None] = [None] * num_robots
        per_robot_stats: list[PlanningStats] = [PlanningStats() for _ in range(num_robots)]

        for _priority, robot_idx in enumerate(priority_order):
            route = routes[robot_idx]
            world_positions, coarse_time_steps, waypoint_schedule, stats = plan_robot_route_st(
                coarse_og,
                committed,
                route,
                waypoint_positions,
                dwell_s=dwell_seconds,
                dt=SPACE_TIME_DT,
                fine_occupancy_grid=self.og,
                robot_radius=ROBOT_RADIUS,
            )
            robot_world_paths[robot_idx] = world_positions
            robot_coarse_times[robot_idx] = coarse_time_steps
            robot_waypoint_schedules[robot_idx] = waypoint_schedule
            per_robot_stats[robot_idx] = stats

        # Per-vehicle travel times
        last_steps = cp.array(
            [
                float(robot_coarse_times[i][-1])
                if robot_coarse_times[i] is not None and len(robot_coarse_times[i]) > 0
                else 0.0
                for i in range(num_robots)
            ]
        )
        makespan_seconds = float(last_steps.max() * SPACE_TIME_DT) if num_robots > 0 else 0.0
        total_travel_seconds = float(last_steps.sum() * SPACE_TIME_DT)

        return (
            robot_world_paths,
            robot_coarse_times,
            robot_waypoint_schedules,
            makespan_seconds,
            total_travel_seconds,
            per_robot_stats,
        )

    # ──────────────────────────────────────────────────────────────────

    def execute(
        self,
        routes: list[list[int]],
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
                self.og,
                coarse_res=SPACE_TIME_RESOLUTION,
            )
        else:
            logger.warning(
                "No occupancy grid; collision avoidance with static obstacles is disabled."
            )
            coarse_og = OccupancyGrid(
                grid=cp.zeros((10, 10, 10), dtype=cp.bool_),
                origin=cp.zeros(3, dtype=cp.float64),
                resolution=SPACE_TIME_RESOLUTION,
            )

        max_time_steps = max(1, int(math.ceil(SPACE_TIME_MAX_HORIZON_S / SPACE_TIME_DT)))
        collision_radius_vox = ROBOT_RADIUS / coarse_og.resolution + SPLINE_SAFETY_VOXELS

        logger.info(
            "[MultiAgentPathPlanner] Coarse grid %s  T=%d  collision_radius_vox=%.2f",
            coarse_og.shape,
            max_time_steps,
            collision_radius_vox,
        )

        # ── 2. Estimate route costs for priority ordering ────────────
        # Routes are [home, c1, ..., ck, home]; strip depots for helper
        customer_routes = [route[1:-1] for route in routes]
        depot_indices = [route[0] for route in routes]
        route_costs = _per_vehicle_costs(customer_routes, dist_matrix, depot_indices)

        default_order = sorted(range(num_robots), key=lambda i: -route_costs[i])
        logger.info(
            "[MultiAgentPathPlanner] Default priority (longest first): %s  costs=%s",
            default_order,
            [f"{route_costs[i]:.1f}" for i in default_order],
        )

        def _combined_objective(makespan: float, total_time: float) -> float:
            return alpha * makespan + (1 - alpha) * total_time

        # ── 3. Try priority orderings, keep best ─────────────────────
        # Each ordering is planned collision-free by construction via the A*'s
        # continuous-time move filter; on the rare geometry where a residual
        # slips through we still *measure* it exactly and prefer an ordering with
        # none. COLLISION_PENALTY dominates any objective difference so a
        # verified-collision-free ordering always wins when one exists.
        from .committed_motion import continuous_collision_report
        COLLISION_PENALTY = 1.0e6

        def _run_trial(order: list[int]) -> tuple:
            result = self._plan_sequential(
                routes,
                waypoint_positions,
                order,
                coarse_og,
                max_time_steps,
                collision_radius_vox,
                dwell_seconds,
            )
            makespan = result[3]
            total_time = result[4]
            n_coll, _ = continuous_collision_report(
                result[0], result[1], max_time_steps, ROBOT_RADIUS
            )
            objective = n_coll * COLLISION_PENALTY + _combined_objective(makespan, total_time)
            return result, objective

        def _update_best(order, trial_result, trial_objective):
            """Return updated (best_result, best_objective, best_order,
            best_stats) if *trial_objective* improves on the current best."""
            nonlocal best_result, best_objective, best_order, best_stats
            if trial_objective < best_objective:
                best_result = trial_result
                best_objective = trial_objective
                best_order = order
                best_stats = trial_result[5]

        total_perms = math.factorial(num_robots)

        if total_perms <= n_priority_trials:
            # ── Exhaustive: try every permutation ────────────────────
            best_result = best_objective = best_order = best_stats = None
            for perm in itertools.permutations(range(num_robots)):
                order = list(perm)
                trial_result, trial_objective = _run_trial(order)
                if best_result is None or trial_objective < best_objective:
                    best_result = trial_result
                    best_objective = trial_objective
                    best_order = order
                    best_stats = trial_result[5]
            logger.info(
                "[MultiAgentPathPlanner] Exhaustive search (%d orderings)  "
                "best order: %s  objective=%.1f  makespan=%.1f s",
                total_perms,
                best_order,
                best_objective,
                best_result[3],
            )
        else:
            # ── Non-exhaustive: deterministic + unique random ────────
            best_result, best_objective = _run_trial(default_order)
            best_order = default_order
            best_stats = best_result[5]
            logger.info(
                "[MultiAgentPathPlanner] Default order (longest-first) "
                "objective: %.1f  makespan: %.1f s",
                best_objective,
                best_result[3],
            )

            tried_orderings: set = {tuple(default_order)}
            trials_used = 1

            if n_priority_trials > trials_used:
                reverse_order = list(reversed(default_order))
                trial_result, trial_objective = _run_trial(reverse_order)
                logger.info(
                    "[MultiAgentPathPlanner] Shortest-first order "
                    "objective: %.1f  makespan: %.1f s",
                    trial_objective,
                    trial_result[3],
                )
                _update_best(reverse_order, trial_result, trial_objective)
                tried_orderings.add(tuple(reverse_order))
                trials_used += 1

            if n_priority_trials > trials_used:
                conflict_scores = [s.wait_steps + s.astar_failures * 100 for s in best_stats]
                conflict_order = sorted(
                    range(num_robots),
                    key=lambda i: -conflict_scores[i],
                )
                if tuple(conflict_order) not in tried_orderings:
                    trial_result, trial_objective = _run_trial(conflict_order)
                    logger.info(
                        "[MultiAgentPathPlanner] Most conflicted order %s "
                        "objective: %.1f  makespan: %.1f s",
                        conflict_order,
                        trial_objective,
                        trial_result[3],
                    )
                    _update_best(conflict_order, trial_result, trial_objective)
                    tried_orderings.add(tuple(conflict_order))
                trials_used += 1

            if n_priority_trials > trials_used:
                conflict_scores = [s.wait_steps + s.astar_failures * 100 for s in best_stats]
                least_conflict_order = sorted(
                    range(num_robots),
                    key=lambda i: conflict_scores[i],
                )
                if tuple(least_conflict_order) not in tried_orderings:
                    trial_result, trial_objective = _run_trial(
                        least_conflict_order,
                    )
                    logger.info(
                        "[MultiAgentPathPlanner] Least conflicted order %s "
                        "objective: %.1f  makespan: %.1f s",
                        least_conflict_order,
                        trial_objective,
                        trial_result[3],
                    )
                    _update_best(least_conflict_order, trial_result, trial_objective)
                    tried_orderings.add(tuple(least_conflict_order))
                trials_used += 1

            # ── Unique random orderings via oversampling ──────
            remaining = n_priority_trials - trials_used
            if remaining > 0:
                tried_indices = {_perm_to_index(list(t)) for t in tried_orderings}
                candidates = random.sample(
                    range(total_perms),
                    remaining + len(tried_indices),
                )
                random_indices = [c for c in candidates if c not in tried_indices][:remaining]
                random_orderings = [_index_to_perm(idx, num_robots) for idx in random_indices]
                for trial_num, random_order in enumerate(
                    random_orderings,
                    start=trials_used + 1,
                ):
                    trial_result, trial_objective = _run_trial(random_order)
                    logger.info(
                        "[MultiAgentPathPlanner] Random trial %d order %s "
                        "objective: %.1f  makespan: %.1f s",
                        trial_num,
                        random_order,
                        trial_objective,
                        trial_result[3],
                    )
                    _update_best(random_order, trial_result, trial_objective)

            logger.info(
                "[MultiAgentPathPlanner] Best priority order: %s  "
                "objective=%.1f  makespan=%.1f s  (%d trials)",
                best_order,
                best_objective,
                best_result[3],
                n_priority_trials,
            )

        (
            robot_world_paths,
            robot_coarse_times,
            robot_waypoint_schedules,
            _,
            _,
            _per_robot_stats,
        ) = best_result

        # Exact continuous-time inter-robot collision status of the committed
        # coarse polylines (ground truth: the dense replay is their linear
        # interpolation). Distinct from the index-aligned dense metric.
        _true_pairs, _true_min_sep = continuous_collision_report(
            robot_world_paths, robot_coarse_times, max_time_steps, ROBOT_RADIUS
        )
        if _true_pairs > 0:
            logger.warning(
                "[MAPF] selected plan still has %d colliding pair(s) (min_sep=%.3f m) - "
                "no collision-free ordering found among %d trials",
                _true_pairs, _true_min_sep, n_priority_trials,
            )
        else:
            logger.info(
                "[MAPF] TRUE continuous inter-robot collisions: pairs=0  min_sep=%.3f m",
                _true_min_sep,
            )

        for robot_idx in best_order:
            world_positions = robot_world_paths[robot_idx]
            coarse_times = robot_coarse_times[robot_idx]
            if world_positions is not None and len(world_positions) > 0:
                logger.info(
                    "  Robot %d: %d coarse samples, t=[%d..%d] (%.1f s)",
                    robot_idx,
                    len(world_positions),
                    int(coarse_times[0]),
                    int(coarse_times[-1]),
                    float(coarse_times[-1]) * SPACE_TIME_DT,
                )

        # ── 4. Densify to replay resolution and assign orientation ────
        # All computation stays on GPU; per-robot CuPy arrays
        all_trajectories_gpu: list[cp.ndarray] = []
        all_velocities_gpu: list[cp.ndarray] = []
        fail_counts = [0] * num_robots

        for robot_idx in range(num_robots):
            world_positions_gpu = robot_world_paths[robot_idx]
            coarse_time_steps_gpu = robot_coarse_times[robot_idx]

            waypoint_schedule = robot_waypoint_schedules[robot_idx] or []
            if waypoint_schedule:
                wp_sched_arr = cp.array(waypoint_schedule, dtype=cp.float64)
                wp_schedule_gpu = cp.empty_like(wp_sched_arr)
                wp_schedule_gpu[:, 0] = wp_sched_arr[:, 0] * SPACE_TIME_DT
                wp_schedule_gpu[:, 1] = wp_sched_arr[:, 1] * SPACE_TIME_DT
                wp_schedule_gpu[:, 2] = wp_sched_arr[:, 2]
            else:
                wp_schedule_gpu = None

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
                    time_seconds,
                    return_index=True,
                )
                unique_indices = cp.sort(unique_indices)
                unique_times = time_seconds[unique_indices]
                unique_positions = world_positions_gpu[unique_indices]
                interpolated_positions = cp.column_stack(
                    [
                        cp.interp(dense_time_samples, unique_times, unique_positions[:, d])
                        for d in range(3)
                    ]
                )
                trajectory = cp.zeros((num_dense_steps, 6), dtype=cp.float64)
                trajectory[:, :3] = interpolated_positions

            apply_heading_orientation(
                trajectory,
                t_dense=dense_time_samples,
                dt=TRAJ_DT,
                wp_schedule_s=wp_schedule_gpu,
                waypoint_rotmats=waypoint_rotmats,
            )

            velocities = cp.zeros_like(trajectory)
            if len(trajectory) > 1:
                velocities[:-1] = cp.diff(trajectory, axis=0) / TRAJ_DT

            all_trajectories_gpu.append(trajectory.astype(cp.float32))
            all_velocities_gpu.append(velocities.astype(cp.float32))

        # ── 5. Build per-robot waypoint list (pos + quaternion) ────────
        from scipy.spatial.transform import Rotation

        waypoint_positions_np = cp.asnumpy(waypoint_positions)
        waypoint_rotmats_np = cp.asnumpy(waypoint_rotmats)
        all_waypoints = []
        for route in routes:
            route_wps = []
            for node in route:
                pos = waypoint_positions_np[node].tolist()
                quat_xyzw = Rotation.from_matrix(waypoint_rotmats_np[node]).as_quat()
                # Isaac Sim expects [qw, qx, qy, qz]
                quat_wxyz = [
                    float(quat_xyzw[3]),
                    float(quat_xyzw[0]),
                    float(quat_xyzw[1]),
                    float(quat_xyzw[2]),
                ]
                route_wps.append(pos + quat_wxyz)
            all_waypoints.append(route_wps)

        # ── 6. Optional collision safety checks (on GPU) ─────────────
        if run_collision_checks:
            from ..core.collision import (
                find_environment_collisions,
                find_trajectory_collisions,
            )

            if self.og is not None:
                env_collision_counts = find_environment_collisions(
                    all_trajectories_gpu,
                    self.og,
                )
                for robot_idx, count in enumerate(env_collision_counts):
                    if count > 0:
                        logger.warning(
                            "[MultiAgentPathPlanner] Robot %d: %d / %d trajectory "
                            "steps collide with fine OG.",
                            robot_idx,
                            count,
                            len(all_trajectories_gpu[robot_idx]),
                        )

            inter_robot_collisions = find_trajectory_collisions(
                all_trajectories_gpu,
            )
            if inter_robot_collisions:
                logger.warning(
                    "[MultiAgentPathPlanner] %d inter-robot collision events detected.",
                    len(inter_robot_collisions),
                )

        # ── 7. Transfer to CPU for ExecutionResult ────────────────────
        all_trajectory_positions: list[np.ndarray] = []
        all_trajectory_velocities: list[np.ndarray] = []
        for robot_idx in range(num_robots):
            traj_np = cp.asnumpy(all_trajectories_gpu[robot_idx])
            vel_np = cp.asnumpy(all_velocities_gpu[robot_idx])
            all_trajectory_positions.append(traj_np)
            all_trajectory_velocities.append(vel_np)

        actual_per_vehicle_times = [
            len(all_trajectories_gpu[i]) * TRAJ_DT for i in range(num_robots)
        ]
        actual_makespan = max(actual_per_vehicle_times) if actual_per_vehicle_times else 0.0
        logger.info(
            "[MultiAgentPathPlanner] Actual makespan: %.1f s  per_vehicle: %s",
            actual_makespan,
            [f"{t:.1f}" for t in actual_per_vehicle_times],
        )

        actual_per_leg_times = [
            _per_leg_travel_times(
                robot_waypoint_schedules[i] or [],
                SPACE_TIME_DT,
            )
            for i in range(num_robots)
        ]

        initial_positions_np = [cp.asnumpy(p) for p in initial_positions]

        return ExecutionResult(
            all_traj_positions=all_trajectory_positions,
            all_traj_velocities=all_trajectory_velocities,
            all_waypoints=all_waypoints,
            initial_positions=initial_positions_np,
            fail_counts=fail_counts,
            actual_makespan=actual_makespan,
            actual_per_vehicle_times=actual_per_vehicle_times,
            actual_per_leg_times=actual_per_leg_times,
        )
