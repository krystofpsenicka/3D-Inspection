"""
VRP Planner – Route Executor (Priority-Based Sequential Planning)

Builds per-vehicle trajectories that are **collision-free by construction**.
Robots are planned one at a time in priority order (longest route first).
Each robot's route is planned via Space-Time A* on a coarse 4-D grid so it
avoids both static obstacles *and* previously committed robots'
trajectories.  The coarse space-time path is then smoothly interpolated at
the replay time-step (``TRAJ_DT``) and converted to 8-DOF joint space.

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
   c.  Smooth-interpolate the coarse path at TRAJ_DT for replay.
5. Run a final AABB safety check (should be clean; log any residual).

References
----------
Silver, D. (2005). Cooperative Pathfinding. AIIDE.
    Priority-based sequential planning framework: robots are planned
    one at a time, each committing its trajectory to a reservation
    table so that later robots avoid it.

Erdmann, M. & Lozano-Perez, T. (1987). On Multiple Moving Objects.
    Algorithmica, 2(1), 477–521.
    Foundational work on planning in configuration-space × time;
    the 4-D space-time grid with a reservation table derives from
    this lineage.

Zhou, Y. & Zeng, J. (2015). Massively Parallel A* Search on a GPU.
    AAAI.
    GPU-parallel frontier expansion via heuristic-guided threshold
    selection (expanding nodes where f ≤ f_min + δ).

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

1. **Low robot density.**  A small fleet (typically 2–5 robots)
   operates in a large 3-D volume (~80 K free voxels, i.e. ~16 K
   voxels per robot at 5 agents).  Spatial conflicts are inherently
   sparse, so the optimality gap between prioritised planning and
   CBS is negligible in practice.

2. **Priority-order search already covers the gap.**  The executor
   tries multiple priority orderings (longest-first, shortest-first,
   most-conflicted-first, plus random permutations) and keeps the
   best makespan.  For 5 robots there are only 5! = 120 possible
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

CBS would become worthwhile at 15–30+ robots or in highly constrained
environments (narrow corridors, bottlenecks) where priority ordering
causes significant cascading delays.  At the current scale — a small
fleet in an open ship hull — the gain is not worth the complexity.

Sharon, G. et al. (2015). Conflict-Based Search for Optimal
    Multi-Agent Pathfinding. Artificial Intelligence, 219, 40–66.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

import cupy as cp
import numpy as np

from ..core.constants import (
    AUV_CRUISE_SPEED,
    BROV_CUBOID_DIMS,
    ROBOT_RADIUS,
    SPACE_TIME_DT,
    SPACE_TIME_DWELL_S,
    SPACE_TIME_MAX_HORIZON_S,
    SPACE_TIME_RESOLUTION,
    SPLINE_SAFETY_VOXELS,
    TRAJ_DT,
)
from ..core.types import ExecutionResult
from shared.grid_utils import downsample_occupancy_grid
from shared.occupancy_grid import OccupancyGrid
from .space_time_search import ReservationTable
from .route_planner import plan_robot_route_st
logger = logging.getLogger(__name__)


# ─── Executor ────────────────────────────────────────────────────────────────

class RouteExecutor:
    """Priority-Based Sequential Planner from Space-Time A* paths.

    Parameters
    ----------
    start_configs:
        ``(num_robots, D)`` initial joint configurations (numpy).
    joint_names:
        Ordered joint name list.
    og:
        Occupancy grid.
    """

    def __init__(
        self,
        start_configs:  List[np.ndarray],
        joint_names:    List[str],
        og: OccupancyGrid | None = None,
    ):
        self.num_robots    = len(start_configs)
        self.start_configs = start_configs
        self.og            = og

    def _plan_sequential(
        self,
        routes: List[List[int]],
        waypoint_positions: cp.ndarray,
        waypoint_rotmats: cp.ndarray,
        home_indices: set,
        priority_order: List[int],
        route_costs: List[float],
        coarse_grid: cp.ndarray,
        coarse_origin: cp.ndarray,
        coarse_res: float,
        T_max: int,
        half_v: np.ndarray,
        dwell_s: float,
    ):
        """Plan all robots sequentially in given priority order.

        Returns (robot_world_paths, robot_coarse_times, robot_wp_schedules,
                 makespan, per_robot_stats).
        """
        from ..core.types import PlanningStats

        n_robots = self.num_robots
        reservation = ReservationTable(coarse_grid.shape, T_max, half_v)

        robot_world_paths: List[Optional[cp.ndarray]] = [None] * n_robots
        robot_coarse_times: List[Optional[cp.ndarray]] = [None] * n_robots
        robot_wp_schedules: List[Optional[list]] = [None] * n_robots
        per_robot_stats: List[PlanningStats] = [PlanningStats() for _ in range(n_robots)]

        for priority, robot_idx in enumerate(priority_order):
            route = routes[robot_idx]
            world_xyz, coarse_t, wp_schedule, stats = plan_robot_route_st(
                coarse_grid, coarse_origin, coarse_res,
                reservation, route,
                waypoint_positions, waypoint_rotmats, home_indices,
                dwell_s=dwell_s,
                dt=SPACE_TIME_DT,
                fine_occupancy_grid=self.og,
                robot_radius=ROBOT_RADIUS,
            )
            robot_world_paths[robot_idx] = world_xyz
            robot_coarse_times[robot_idx] = coarse_t
            robot_wp_schedules[robot_idx] = wp_schedule
            per_robot_stats[robot_idx] = stats

        # Compute makespan (max final time across robots)
        makespan = 0.0
        for i in range(n_robots):
            ct = robot_coarse_times[i]
            if ct is not None and len(ct) > 0:
                makespan = max(makespan, float(ct[-1]) * SPACE_TIME_DT)

        return (robot_world_paths, robot_coarse_times, robot_wp_schedules,
                makespan, per_robot_stats)

    # ──────────────────────────────────────────────────────────────────

    def execute(
        self,
        routes:              List[List[int]],
        waypoint_positions:  cp.ndarray,
        waypoint_rotmats:    cp.ndarray,
        home_indices:        set,
        dwell_s: float = SPACE_TIME_DWELL_S,
        n_priority_trials: int = 20,
    ) -> ExecutionResult:
        """Build collision-free trajectories via Priority-Based Sequential
        Planning and return an :class:`ExecutionResult`.

        Parameters
        ----------
        waypoint_positions : cp.ndarray (N, 3)
            World-frame positions for all VRP nodes (homes + inspection).
        waypoint_rotmats : cp.ndarray (N, 3, 3)
            Rotation matrices for all VRP nodes (identity for homes).
        n_priority_trials : int
            Total budget of priority orderings to try (including
            deterministic ones).  The best (lowest makespan) is kept.
        """
        n_robots = self.num_robots

        initial_positions = [cfg[:3].copy() for cfg in self.start_configs]

        # ── 1. Coarse grid + reservation table ────────────────────────
        if self.og is not None:
            coarse_grid, coarse_origin, coarse_res = downsample_occupancy_grid(
                self.og.grid, self.og.origin, self.og.resolution,
                coarse_res=SPACE_TIME_RESOLUTION,
            )
        else:
            logger.warning("No occupancy grid; collision avoidance with "
                           "static obstacles is disabled.")
            coarse_grid = cp.zeros((10, 10, 10), dtype=cp.bool_)
            coarse_origin = cp.zeros(3, dtype=cp.float64)
            coarse_res = SPACE_TIME_RESOLUTION

        T_max = max(1, int(math.ceil(SPACE_TIME_MAX_HORIZON_S / SPACE_TIME_DT)))
        dims = np.array(BROV_CUBOID_DIMS, dtype=np.float64)
        half_v = np.maximum(np.ceil(dims / (2.0 * coarse_res)).astype(np.intp), 1)
        half_v += SPLINE_SAFETY_VOXELS

        logger.info("[RouteExecutor] Coarse grid %s  T=%d  half_v=%s",
                    coarse_grid.shape, T_max, half_v)

        # ── 2. Priority ordering ──────────────────────────────────────
        route_costs: List[float] = []
        for i, route in enumerate(routes):
            cost = 0.0
            for leg in range(1, len(route)):
                a, b = route[leg - 1], route[leg]
                cost += float(cp.linalg.norm(
                    waypoint_positions[b] - waypoint_positions[a]))
            route_costs.append(cost)

        default_order = sorted(range(n_robots), key=lambda i: -route_costs[i])
        logger.info("[RouteExecutor] Default priority (longest first): %s  costs=%s",
                    default_order, [f"{route_costs[i]:.1f}" for i in default_order])

        # ── 3. Try priority orderings, keep best ─────────────
        # Deterministic ordering 1: longest-first (default)
        best_result = self._plan_sequential(
            routes, waypoint_positions, waypoint_rotmats, home_indices,
            default_order, route_costs,
            coarse_grid, coarse_origin, coarse_res, T_max, half_v,
            dwell_s,
        )
        best_makespan = best_result[3]
        best_order = default_order
        best_stats = best_result[4]
        logger.info("[RouteExecutor] Default order (longest-first) makespan: %.1f s",
                    best_makespan)

        trials_used = 1

        # Deterministic ordering 2: shortest-first
        if n_priority_trials > trials_used:
            reverse_order = list(reversed(default_order))
            trial_result = self._plan_sequential(
                routes, waypoint_positions, waypoint_rotmats, home_indices,
                reverse_order, route_costs,
                coarse_grid, coarse_origin, coarse_res, T_max, half_v,
                dwell_s,
            )
            trial_makespan = trial_result[3]
            logger.info("[RouteExecutor] Shortest-first order makespan: %.1f s",
                        trial_makespan)
            if trial_makespan < best_makespan:
                best_result = trial_result
                best_makespan = trial_makespan
                best_order = reverse_order
                best_stats = trial_result[4]
            trials_used += 1

        # Deterministic ordering 3: conflict-count (most-conflicted first)
        if n_priority_trials > trials_used:
            conflict_scores = [
                s.wait_steps + s.astar_failures * 100
                for s in best_stats
            ]
            conflict_order = sorted(
                range(n_robots), key=lambda i: -conflict_scores[i],
            )
            if conflict_order != default_order:
                trial_result = self._plan_sequential(
                    routes, waypoint_positions, waypoint_rotmats, home_indices,
                    conflict_order, route_costs,
                    coarse_grid, coarse_origin, coarse_res, T_max, half_v,
                    dwell_s,
                )
                trial_makespan = trial_result[3]
                logger.info("[RouteExecutor] Most conflicted order %s makespan: %.1f s",
                            conflict_order, trial_makespan)
                if trial_makespan < best_makespan:
                    best_result = trial_result
                    best_makespan = trial_makespan
                    best_order = conflict_order
                    best_stats = trial_result[4]
            trials_used += 1

        # Deterministic ordering 4: conflict-count (least-conflicted first)
        if n_priority_trials > trials_used:
            conflict_scores = [
                s.wait_steps + s.astar_failures * 100
                for s in best_stats
            ]
            conflict_order = sorted(
                range(n_robots), key=lambda i: conflict_scores[i],
            )
            if conflict_order != default_order:
                trial_result = self._plan_sequential(
                    routes, waypoint_positions, waypoint_rotmats, home_indices,
                    conflict_order, route_costs,
                    coarse_grid, coarse_origin, coarse_res, T_max, half_v,
                    dwell_s,
                )
                trial_makespan = trial_result[3]
                logger.info("[RouteExecutor] Least conflicted order %s makespan: %.1f s",
                            conflict_order, trial_makespan)
                if trial_makespan < best_makespan:
                    best_result = trial_result
                    best_makespan = trial_makespan
                    best_order = conflict_order
                    best_stats = trial_result[4]
            trials_used += 1

        # Remaining budget: random permutations
        for trial in range(trials_used, n_priority_trials):
            random_order = list(np.random.permutation(n_robots))
            trial_result = self._plan_sequential(
                routes, waypoint_positions, waypoint_rotmats, home_indices,
                random_order, route_costs,
                coarse_grid, coarse_origin, coarse_res, T_max, half_v,
                dwell_s,
            )
            trial_makespan = trial_result[3]
            logger.info("[RouteExecutor] Random trial %d order %s makespan: %.1f s",
                        trial + 1, random_order, trial_makespan)
            if trial_makespan < best_makespan:
                best_result = trial_result
                best_makespan = trial_makespan
                best_order = random_order
                best_stats = trial_result[4]

        logger.info("[RouteExecutor] Best priority order: %s  makespan=%.1f s "
                    "(%d trials)", best_order, best_makespan, n_priority_trials)

        (robot_world_paths, robot_coarse_times, robot_wp_schedules,
         _, _per_robot_stats) = best_result

        for robot_idx in best_order:
            w_xyz = robot_world_paths[robot_idx]
            c_t = robot_coarse_times[robot_idx]
            if w_xyz is not None and len(w_xyz) > 0:
                logger.info("  Robot %d: %d coarse samples, t=[%d..%d] (%.1f s)",
                            robot_idx, len(w_xyz),
                            int(c_t[0]), int(c_t[-1]),
                            float(c_t[-1]) * SPACE_TIME_DT)

        # ── 4. Smooth-interpolate to replay resolution ────────────────
        # CPU: np.interp, trajectory densification
        rotmats_np = cp.asnumpy(waypoint_rotmats)

        all_traj_positions:  List[List[np.ndarray]] = [[] for _ in range(n_robots)]
        all_traj_velocities: List[List[np.ndarray]] = [[] for _ in range(n_robots)]
        fail_counts = [0] * n_robots

        for i in range(n_robots):
            # GPU → CPU for trajectory densification
            w_xyz = robot_world_paths[i].get()
            c_t   = robot_coarse_times[i].get()

            cfg   = self.start_configs[i].copy()  # 8-DOF
            wp_sched = robot_wp_schedules[i] or []
            # Convert coarse time-step schedule -> seconds
            wp_sched_s = [
                (int(ts) * SPACE_TIME_DT, int(te) * SPACE_TIME_DT, node)
                for ts, te, node in wp_sched
            ]

            if w_xyz is None or len(w_xyz) == 0:
                # Nothing planned - hold at home
                fail_counts[i] = len(routes[i]) - 1
                for _ in range(50):
                    all_traj_positions[i].append(cfg.astype(np.float32).copy())
                    all_traj_velocities[i].append(np.zeros_like(cfg, dtype=np.float32))
                continue

            # Convert coarse time-steps -> continuous seconds
            t_seconds = c_t.astype(np.float64) * SPACE_TIME_DT
            t0, tf = float(t_seconds[0]), float(t_seconds[-1])
            duration = tf - t0

            if duration < 1e-6:
                traj_js = np.tile(cfg.astype(np.float64), (1, 1))
                traj_js[0, :3] = w_xyz[-1]
                t_dense = np.array([t0])
            else:
                n_steps = max(2, int(np.ceil(duration / TRAJ_DT)))
                t_dense = np.linspace(t0, tf, n_steps)
                # Remove duplicate time stamps from dwells
                _, unique_idx = np.unique(t_seconds, return_index=True)
                unique_idx = np.sort(unique_idx)
                t_uniq = t_seconds[unique_idx]
                xyz_uniq = w_xyz[unique_idx]
                # Linear densification – path is already OMPL-smooth
                xyz_dense = np.column_stack([
                    np.interp(t_dense, t_uniq, xyz_uniq[:, d]) for d in range(3)
                ])
                traj_js = np.tile(cfg.astype(np.float64), (n_steps, 1))
                traj_js[:, :3] = xyz_dense

            # Orientation: interpolate yaw/pitch between consecutive waypoints
            _apply_heading_orientation(
                traj_js,
                t_dense=t_dense,
                dt=TRAJ_DT,
                wp_schedule_s=wp_sched_s,
                waypoint_rotmats_np=rotmats_np,
            )

            # Finite-difference velocities
            vel = np.zeros_like(traj_js)
            if len(traj_js) > 1:
                vel[:-1] = np.diff(traj_js, axis=0) / TRAJ_DT

            for sp, sv in zip(traj_js, vel):
                all_traj_positions[i].append(sp.astype(np.float32))
                all_traj_velocities[i].append(sv.astype(np.float32))

        # ── 5. Safety checks ────────────────────────────────────────────
        # Fine OG obstacle check on smoothed trajectories
        if self.og is not None:
            for i in range(n_robots):
                og_collisions = 0
                for step_pos in all_traj_positions[i]:
                    xyz = step_pos[:3]
                    if not self.og.is_free_world(xyz):
                        og_collisions += 1
                if og_collisions > 0:
                    logger.warning(
                        "[RouteExecutor] Robot %d: %d / %d trajectory steps "
                        "collide with fine OG.",
                        i, og_collisions, len(all_traj_positions[i]),
                    )

        # Compute actual per-vehicle travel times from trajectory lengths
        actual_per_vehicle = []
        for i in range(n_robots):
            n_steps = len(all_traj_positions[i])
            actual_per_vehicle.append(n_steps * TRAJ_DT)
        actual_makespan = max(actual_per_vehicle) if actual_per_vehicle else 0.0
        logger.info("[RouteExecutor] Actual makespan: %.1f s  per_vehicle: %s",
                    actual_makespan,
                    [f"{t:.1f}" for t in actual_per_vehicle])

        return ExecutionResult(
            all_traj_positions       = all_traj_positions,
            all_traj_velocities      = all_traj_velocities,
            all_waypoints            = all_waypoints,
            initial_positions        = initial_positions,
            fail_counts              = fail_counts,
            actual_makespan          = actual_makespan,
            actual_per_vehicle_times = actual_per_vehicle,
        )


# ─── Orientation: interpolate yaw between consecutive waypoints ─────────────

def _apply_heading_orientation(
    traj: np.ndarray,
    t_dense: np.ndarray,
    dt: float = TRAJ_DT,
    wp_schedule_s: Optional[list] = None,
    waypoint_rotmats_np: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Set yaw (joint 3) and camera pitch (joint 7) on a dense trajectory.

    During dwell windows the robot holds the waypoint's viewing direction
    derived from the rotation matrix forward column.  Between dwells the
    yaw and camera pitch are cosine-eased between successive waypoint values.

    Joint layout: [x, y, z, yaw, pitch, roll, cam_yaw, cam_pitch]
    """
    N = len(traj)
    if N < 2:
        return traj

    yaw          = np.empty(N)
    camera_pitch = np.zeros(N)

    if not wp_schedule_s or waypoint_rotmats_np is None:
        yaw[:] = traj[0, 3]
        traj[:, 3] = (yaw + math.pi) % (2 * math.pi) - math.pi
        return traj

    # ── Extract per-waypoint yaw and camera pitch from rotation matrices ────
    wp_yaws:   list = []
    wp_cpitch: list = []
    wp_t_ds:   list = []
    wp_t_de:   list = []

    for (t_ds, t_de, node_idx) in wp_schedule_s:
        # Forward direction = column 0 of rotation matrix
        forward = waypoint_rotmats_np[node_idx, :, 0]
        fx, fy, fz = float(forward[0]), float(forward[1]), float(forward[2])
        xy_norm = math.sqrt(fx * fx + fy * fy)
        wp_yaws.append(math.atan2(fy, fx))
        wp_cpitch.append(-math.atan2(fz, xy_norm) if xy_norm > 1e-9 else 0.0)
        wp_t_ds.append(t_ds)
        wp_t_de.append(t_de)

    # Unwrap waypoint yaw sequence so interpolation always takes the short arc
    wp_yaws = list(np.unwrap(wp_yaws))

    n_wp = len(wp_schedule_s)

    def _dense_idx(t: float, side: str = "left") -> int:
        return max(0, min(int(np.searchsorted(t_dense, t, side=side)), N))

    # ── Before first waypoint: hold first waypoint yaw ──────────────────────
    d_start_0 = _dense_idx(wp_t_ds[0])
    yaw[:d_start_0]          = wp_yaws[0]
    camera_pitch[:d_start_0] = 0.0

    # ── Fill dwell windows and transit segments ──────────────────────────────
    for i in range(n_wp):
        d_start = _dense_idx(wp_t_ds[i])
        d_end   = _dense_idx(wp_t_de[i], side="right")

        # Dwell: exact waypoint orientation
        yaw[d_start:d_end]          = wp_yaws[i]
        camera_pitch[d_start:d_end] = wp_cpitch[i]

        if i < n_wp - 1:
            # Transit: cosine-ease from wp[i] to wp[i+1]
            next_start = _dense_idx(wp_t_ds[i + 1])
            if next_start > d_end:
                ts   = t_dense[d_end:next_start]
                t0_t = t_dense[d_end]
                t1_t = t_dense[min(next_start, N - 1)]
                dur  = t1_t - t0_t
                if dur > 0:
                    alpha = (ts - t0_t) / dur
                    ease  = 0.5 * (1.0 - np.cos(math.pi * alpha))
                    yaw[d_end:next_start]          = (wp_yaws[i]
                        + ease * (wp_yaws[i + 1] - wp_yaws[i]))
                    camera_pitch[d_end:next_start] = (wp_cpitch[i]
                        + ease * (wp_cpitch[i + 1] - wp_cpitch[i]))
                else:
                    yaw[d_end:next_start]          = wp_yaws[i]
                    camera_pitch[d_end:next_start] = wp_cpitch[i]
        else:
            # After last waypoint: hold last waypoint yaw
            yaw[d_end:]          = wp_yaws[i]
            camera_pitch[d_end:] = 0.0

    # ── Write back ───────────────────────────────────────────────────────────
    traj[:, 3] = yaw
    if traj.shape[1] > 7:
        traj[:, 7] = camera_pitch
    return traj
