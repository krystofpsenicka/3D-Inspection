"""
VRP Planner – Pipeline Orchestrator

:class:`VRPPipeline` wires all modules together in the correct order:

1. Build / load occupancy grid from the wreck mesh.
2. Load robot config; determine depot position (retract config XYZ).
   Load inspection waypoints and prepend the depot as node 0.
3. Compute GPU distance matrix over all nodes (depot + inspection).
4. Solve VRP (cuOpt GPU → OR-Tools CPU fallback); wrap routes with depot.
5. Execute trajectories per robot (cuRobo robot-unaware + RRT* warm-start).
   Post-planning AABB collision resolution via hold-step insertion.
6. (Optional) Replay in Isaac Sim.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np

from ..core.constants import (
    ASSETS_PATH,
    CONFIGS_PATH,
    MIP_GAP,
    MIP_TIME_LIMIT,
    RAPIDS_PYTHON,
    ROBOT_RADIUS,
    VRP_ALPHA,
    VRP_FEEDBACK_ITERATIONS,
    VRP_FEEDBACK_THRESHOLD,
)
from ..core.types import ExecutionResult, PipelineConfig, VRPResult
from ..core.distance_matrix import build_route_path_cache, compute_distance_matrix
from ..core.occupancy_grid import OccupancyGrid, build_occupancy_grid, get_mesh_world_bounds
from ..mapf.route_executor import RouteExecutor
from ..core.robot_config import load_local_robot_config
from ..vrp.vrp_solver import solve_vrp
from ..core.waypoint_loader import load_waypoints

logger = logging.getLogger(__name__)

def _compute_start_grid(
    num_robots: int,
    mesh_bounds_min: np.ndarray,
    mesh_bounds_max: np.ndarray,
    z_above: float = 1.0,
    spacing: float = 1.5,
):
    """Return per-robot start positions above the mesh centre.

    The ``num_robots`` start positions are arranged in a rectangular grid
    centred on the XY centre of the mesh bounding box, ``z_above`` metres
    above the mesh top, with ``spacing`` metres between adjacent robots.

    Depot Z is anchored to the *mesh* top (not the occupancy grid top) so
    that the value is stable and can be used to extend the grid bounds
    before voxelisation.

    Parameters
    ----------
    num_robots:
        Number of robots to place.
    mesh_bounds_min:
        World-frame lower bound of the scaled+posed mesh ``(3,)``.
    mesh_bounds_max:
        World-frame upper bound of the scaled+posed mesh ``(3,)``.
    z_above:
        Clearance above the mesh top (metres).
    spacing:
        Distance between adjacent robot grid cells (metres).

    Returns
    -------
    start_positions : List[np.ndarray]
        Per-robot ``(3,)`` start XYZ, indexed 0 … num_robots-1.
    """
    import math

    # Centre XY of the mesh bounding box
    cx = float((mesh_bounds_min[0] + mesh_bounds_max[0]) / 2.0)
    cy = float((mesh_bounds_min[1] + mesh_bounds_max[1]) / 2.0)
    # Z: mesh top + clearance (stable anchor independent of grid size)
    z  = float(mesh_bounds_max[2] + z_above)

    cols = math.ceil(math.sqrt(num_robots))
    rows = math.ceil(num_robots / cols)

    start_positions: List[np.ndarray] = []
    for idx in range(num_robots):
        r = idx // cols
        c = idx % cols
        x = cx + (c - (cols - 1) / 2.0) * spacing
        y = cy + (r - (rows - 1) / 2.0) * spacing
        start_positions.append(np.array([x, y, z], dtype=np.float32))

    return start_positions


class VRPPipeline:
    """End-to-end collision-free VRP pipeline.

    Parameters
    ----------
    cfg:
        :class:`PipelineConfig` describing the full run.
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    # ──────────────────────────────────────────────────────────────────

    def run(self) -> ExecutionResult:
        """Execute all pipeline stages and return trajectory buffers."""
        cfg = self.cfg
        logger.info("=" * 60)
        logger.info("VRP Pipeline  (%d robots)", cfg.num_robots)
        logger.info("=" * 60)

        # ── Stage 1: Occupancy grid ───────────────────────────────────
        logger.info("[1/5] Building occupancy grid …")

        # ── Stage 2: Robot config + depot positions (computed BEFORE grid) ─
        # Depot Z is anchored to the mesh top so the grid can be extended
        # to include the depots in one voxelisation pass, giving accurate
        # A* costs and path-cache trajectories to/from depot nodes.
        logger.info("[2/5] Loading robot config and waypoints …")

        robot_cfg   = load_local_robot_config("brov.yml")
        j_names     = robot_cfg["kinematics"]["cspace"]["joint_names"]
        default_cfg = robot_cfg["kinematics"]["cspace"]["retract_config"]

        # Probe mesh bounds (fast — no voxelisation) so depot Z is stable.
        logger.info("      Probing mesh bounds for depot placement …")
        mesh_bounds_min, mesh_bounds_max = get_mesh_world_bounds()
        logger.info("      Mesh world bounds: min=%s  max=%s",
                    mesh_bounds_min.round(2), mesh_bounds_max.round(2))

        robot_start_xyzs = _compute_start_grid(
            cfg.num_robots, mesh_bounds_min, mesh_bounds_max
        )
        for i, p in enumerate(robot_start_xyzs):
            logger.info("      Robot %d home XYZ: %s", i, p)

        # Build grid sized to cover both the mesh and all depot positions.
        og: OccupancyGrid = build_occupancy_grid(
            extra_free_points=np.array(robot_start_xyzs, dtype=np.float32),
        )
        logger.info("      Grid shape: %s  resolution: %.2fm",
                    og.grid.shape, og.resolution)

        # Sanity-check: every depot must be a valid, free voxel now.
        for i, xyz in enumerate(robot_start_xyzs):
            valid = og.is_valid_voxel(og.world_to_voxel(xyz))
            free  = og.is_free_world(xyz)
            if not valid or not free:
                logger.warning(
                    "      Robot %d depot (%s) valid=%s free=%s – "
                    "depot may clip an obstacle or grid boundary.",
                    i, xyz, valid, free
                )

        inspection_wps = np.array(
            load_waypoints(
                source      = cfg.waypoint_source,
                n_random    = cfg.n_random_waypoints,
                og          = og,
                random_seed = cfg.random_seed,
            ),
            dtype=np.float32,
        )  # (N, 7)
        N = len(inspection_wps)
        logger.info("      %d inspection waypoints loaded.", N)

        # Home nodes occupy indices 0..K-1; inspection nodes occupy K..K+N-1.
        K = cfg.num_robots
        home_poses = np.array(
            [[*xyz, 1.0, 0.0, 0.0, 0.0] for xyz in robot_start_xyzs],
            dtype=np.float32,
        )  # (K, 7)
        waypoints_world = np.vstack([home_poses, inspection_wps])  # (K+N, 7)
        home_indices = list(range(K))   # per-robot depot indices for VRP
        logger.info("      Total VRP nodes: %d (%d homes + %d inspection)",
                    len(waypoints_world), K, N)

        # ── Stage 3: Distance matrix (depot + inspection waypoints) ────────
        M = len(waypoints_world)
        logger.info("[3/5] Computing %dx%d distance matrix (cuGraph) …", M, M)
        dist_matrix = compute_distance_matrix(og, waypoints_world[:, :3])
        logger.info("      Distance matrix computed.  max_dist=%.2fm",
                    float(np.max(dist_matrix[np.isfinite(dist_matrix)])))

        # ── Build start configs (shared across feedback iterations) ────
        start_configs: List[np.ndarray] = []
        for i in range(cfg.num_robots):
            s    = list(default_cfg)
            xyz  = robot_start_xyzs[i]
            s[0] = float(xyz[0])
            s[1] = float(xyz[1])
            s[2] = float(xyz[2])
            start_configs.append(np.array(s, dtype=np.float32))

        # ── Stage 4+5: VRP solve ↔ path planning feedback loop ──────────
        current_dist = dist_matrix.copy()
        max_iters = max(1, cfg.feedback_iterations)
        exec_result = None

        for iteration in range(max_iters):
            # ── Stage 4: VRP solve ───────────────────────────────────
            logger.info("[4/5] Solving VRP (%s, alpha=%.2f, iter=%d/%d) …",
                        cfg.solver_backend, cfg.alpha,
                        iteration + 1, max_iters)
            vrp_result: VRPResult = solve_vrp(
                dist_matrix    = current_dist,
                num_vehicles   = cfg.num_robots,
                depot          = home_indices,
                alpha          = cfg.alpha,
                backend        = cfg.solver_backend,
                rapids_python  = cfg.rapids_python,
                time_limit     = cfg.mip_time_limit,
                gpu_timeout    = cfg.gpu_timeout,
                mip_gap        = cfg.mip_gap,
            )
            logger.info("      VRP status=%s  total_cost=%.2f  makespan=%.2f  "
                        "solver=%s",
                        vrp_result.status, vrp_result.total_cost,
                        vrp_result.makespan, vrp_result.solver)
            logger.info("      Routes (pre-home): %s", vrp_result.routes)

            if not any(vrp_result.routes):
                logger.error("VRP produced empty routes – aborting.")
                raise RuntimeError(f"VRP failed: {vrp_result.status}")

            # Wrap each route with the robot's own home node as start/end.
            routes = [
                [home_indices[i]] + list(r) + [home_indices[i]]
                for i, r in enumerate(vrp_result.routes)
            ]
            logger.info("      Routes (with homes): %s", routes)

            # Build A* sub-waypoint cache for every segment used by routes.
            logger.info("      Building route path cache (A* sub-waypoints) …")
            path_cache = build_route_path_cache(
                og, waypoints_world[:, :3], routes,
            )

            # ── Stage 5: Trajectory execution ────────────────────────
            logger.info("[5/5] Generating trajectories (iter=%d/%d) …",
                        iteration + 1, max_iters)

            executor = RouteExecutor(
                start_configs = start_configs,
                joint_names   = j_names,
                og            = og,
            )
            exec_result = executor.execute(
                routes          = routes,
                waypoints_world = waypoints_world,
                path_cache      = path_cache,
            )

            # ── Check feedback convergence ───────────────────────────
            vrp_makespan = vrp_result.makespan
            actual_makespan = exec_result.actual_makespan

            if vrp_makespan < 1e-6:
                logger.info("      VRP makespan near zero; skipping feedback.")
                break

            ratio = abs(actual_makespan - vrp_makespan) / vrp_makespan
            logger.info("      Feedback: VRP makespan=%.1f  actual=%.1f  "
                        "ratio=%.2f  threshold=%.2f",
                        vrp_makespan, actual_makespan, ratio,
                        cfg.feedback_threshold)

            if ratio <= cfg.feedback_threshold:
                logger.info("      Feedback converged (within %.0f%%).",
                            cfg.feedback_threshold * 100)
                break

            if iteration < max_iters - 1:
                # Update distance matrix with actual travel times where
                # the actual cost significantly exceeds the estimate.
                logger.info("      Updating distance matrix for re-solve …")
                updated = 0
                for v, route in enumerate(routes):
                    for leg in range(1, len(route)):
                        a, b = route[leg - 1], route[leg]
                        est = float(current_dist[a, b])
                        # Estimate actual leg cost from trajectory
                        # (proportional share of vehicle's total time)
                        v_total = exec_result.actual_per_vehicle_times[v]
                        n_legs = max(1, len(route) - 1)
                        actual_leg = v_total / n_legs
                        if actual_leg > est * (1 + cfg.feedback_threshold):
                            current_dist[a, b] = actual_leg
                            current_dist[b, a] = actual_leg
                            updated += 1
                logger.info("      Updated %d distance matrix entries.", updated)

        logger.info("=" * 60)
        logger.info("Pipeline complete.  Fail counts: %s  Actual makespan: %.1f s",
                    exec_result.fail_counts, exec_result.actual_makespan)
        logger.info("=" * 60)

        # ── Optional: save solution for later Isaac Sim replay ────────
        if cfg.save_solution_path:
            from ..core.serialization import save_solution
            save_solution(exec_result, cfg.save_solution_path)
            logger.info("Solution saved to: %s", cfg.save_solution_path)

        # ── Optional: Isaac Sim replay (same-env mode) ────────────────
        if cfg.replay_in_isaac and not cfg.save_solution_path:
            from ._isaac_replay import replay_in_isaac_sim
            replay_in_isaac_sim(exec_result, headless=cfg.headless)

        return exec_result
