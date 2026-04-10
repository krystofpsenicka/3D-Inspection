"""
VRP Feedback Orchestrator
=========================

:class:`VRPFeedbackOrchestrator` wires all modules together:

1. Probe mesh bounds and compute depot positions.
2. Assemble VRP nodes (depots + inspection waypoints).
3. Build occupancy grid covering all robot positions.
4. Compute GPU distance matrix over all nodes.
5. Solve VRP (cuOpt GPU or HiGHS CPU).
6. Execute trajectories per robot (Space-Time A* + OMPL smoothing).

Stages 5-6 run inside a feedback loop: if the actual makespan
diverges from the VRP estimate, the distance matrix is updated
and re-solved.
"""

from __future__ import annotations

import logging
from typing import List

import cupy as cp
import numpy as np

from .constants import (
    INFLATION_VOXELS,
    MESH_PATH,
    MESH_POSE,
    MESH_TARGET_LENGTH,
    ROBOT_RADIUS,
    VOXEL_RESOLUTION,
)
from .types import ExecutionResult, PipelineConfig, VRPResult
from shared.types import Side
from .distance_matrix import compute_distance_matrix
from .geometry import compute_start_grid, viewpoints_to_robot_waypoints
from shared.occupancy_grid import OccupancyGrid
from shared.mesh_loader import load_and_transform_mesh
from shared.grid_builder_utils import build_occupancy_grid
from ..mapf.route_executor import MultiAgentPathPlanner
from ..vrp.vrp_solver import solve_vrp

logger = logging.getLogger(__name__)


class VRPFeedbackOrchestrator:
    """End-to-end collision-free VRP pipeline with feedback.

    Parameters
    ----------
    cfg:
        :class:`PipelineConfig` describing the full run.
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    # ──────────────────────────────────────────────────────────────────

    def run(
        self,
        insp_positions: cp.ndarray,
        insp_rotmats: cp.ndarray,
    ) -> ExecutionResult:
        """Execute all pipeline stages and return trajectories.

        Parameters
        ----------
        insp_positions:
            (N, 3) CuPy — inspection waypoint world-frame positions.
        insp_rotmats:
            (N, 3, 3) CuPy — inspection waypoint rotation matrices.
        """
        cfg = self.cfg
        logger.info("=" * 60)
        logger.info("VRP Feedback Orchestrator  (%d robots, side=%s)",
                    cfg.num_robots, cfg.side.value)
        logger.info("=" * 60)

        # ── Stage 1: Probe mesh bounds + depot positions ─────────────
        logger.info("[1/6] Probing mesh bounds and computing depot positions …")

        mesh = load_and_transform_mesh(MESH_PATH, MESH_TARGET_LENGTH, MESH_POSE)
        mesh_bounds_min = np.asarray(mesh.bounds[0], dtype=float)
        mesh_bounds_max = np.asarray(mesh.bounds[1], dtype=float)
        logger.info("      Mesh world bounds: min=%s  max=%s",
                    mesh_bounds_min.round(2), mesh_bounds_max.round(2))

        robot_start_xyzs = compute_start_grid(
            cfg.num_robots, mesh_bounds_min, mesh_bounds_max
        )
        for i, p in enumerate(robot_start_xyzs):
            logger.info("      Robot %d home XYZ: %s", i, p)

        # ── Stage 2: Assemble VRP nodes ──────────────────────────────
        logger.info("[2/6] Assembling VRP nodes …")

        N = len(insp_positions)
        K = cfg.num_robots
        logger.info("      %d inspection waypoints received.", N)

        home_positions = cp.array(
            [[float(xyz[0]), float(xyz[1]), float(xyz[2])]
             for xyz in robot_start_xyzs],
            dtype=cp.float64,
        )
        home_rotmats = cp.tile(cp.eye(3, dtype=cp.float64), (K, 1, 1))

        all_positions = cp.vstack([home_positions, insp_positions])
        all_rotmats = cp.concatenate([home_rotmats, insp_rotmats])
        home_indices = list(range(K))
        logger.info("      Total VRP nodes: %d (%d homes + %d inspection)",
                    len(all_positions), K, N)

        robot_positions_gpu = viewpoints_to_robot_waypoints(
            all_positions, all_rotmats, set(home_indices),
        )

        # ── Stage 3: Occupancy grid ─────────────────────────────────
        logger.info("[3/6] Building occupancy grid …")

        # All robot positions (waypoints + depots) as extra free space
        extra_free = cp.asnumpy(robot_positions_gpu).astype(np.float32)
        extra_margin = max(3, int(np.ceil(ROBOT_RADIUS / VOXEL_RESOLUTION)))
        og: OccupancyGrid = build_occupancy_grid(
            mesh=mesh,
            padding=1.0,
            inflation_voxels=INFLATION_VOXELS,
            resolution=VOXEL_RESOLUTION,
            fill_interior=(cfg.side == Side.OUTSIDE),
            extra_free_points=extra_free,
            extra_margin_voxels=extra_margin,
        )
        logger.info("      Grid shape: %s  resolution: %.2fm  fill_interior=%s",
                    og.grid.shape, og.resolution,
                    cfg.side == Side.OUTSIDE)

        # Sanity-check: every depot must be a valid, free voxel.
        for i, xyz in enumerate(robot_start_xyzs):
            valid = og.is_valid_voxel(og.world_to_voxel(xyz))
            free = og.is_free_world(xyz)
            if not valid or not free:
                logger.warning(
                    "      Robot %d depot (%s) valid=%s free=%s – "
                    "depot may clip an obstacle or grid boundary.",
                    i, xyz, valid, free
                )

        # ── Stage 4: Distance matrix ────────────────────────────────
        M = len(all_positions)
        logger.info("[4/6] Computing %dx%d distance matrix (cuGraph) …", M, M)
        dist_matrix = compute_distance_matrix(og, robot_positions_gpu)
        logger.info("      Distance matrix computed.  max_dist=%.2fm",
                    float(cp.max(dist_matrix[cp.isfinite(dist_matrix)])))

        # ── Build start positions (shared across feedback iterations) ──
        start_positions: List[cp.ndarray] = [
            cp.asarray(xyz, dtype=cp.float32) for xyz in robot_start_xyzs
        ]

        # ── Stages 5+6: VRP solve ↔ path planning feedback loop ─────
        current_dist = dist_matrix.copy()
        max_iters = max(1, cfg.feedback_iterations)
        exec_result = None

        for iteration in range(max_iters):
            # ── Stage 5: VRP solve ───────────────────────────────────
            logger.info("[5/6] Solving VRP (%s, alpha=%.2f, iter=%d/%d) …",
                        cfg.solver_backend, cfg.alpha,
                        iteration + 1, max_iters)
            vrp_result: VRPResult = solve_vrp(
                dist_matrix=current_dist,
                num_vehicles=cfg.num_robots,
                depots=home_indices,
                alpha=cfg.alpha,
                backend=cfg.solver_backend,
                rapids_python=cfg.rapids_python,
                time_limit=cfg.mip_time_limit,
                gpu_timeout=cfg.gpu_timeout,
                mip_gap=cfg.mip_gap,
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

            # ── Stage 6: Trajectory execution ────────────────────────
            logger.info("[6/6] Generating trajectories (iter=%d/%d) …",
                        iteration + 1, max_iters)

            planner = MultiAgentPathPlanner(
                start_positions=start_positions,
                og=og,
            )
            exec_result = planner.execute(
                routes=routes,
                waypoint_positions=robot_positions_gpu,
                waypoint_rotmats=all_rotmats,
                home_indices=set(home_indices),
                dist_matrix=current_dist,
                alpha=cfg.alpha,
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
                # Vectorized distance matrix update using CuPy
                logger.info("      Updating distance matrix for re-solve …")
                all_a, all_b, all_actual = [], [], []
                for v, route in enumerate(routes):
                    n_legs = max(1, len(route) - 1)
                    actual_leg = exec_result.actual_per_vehicle_times[v] / n_legs
                    for leg in range(1, len(route)):
                        all_a.append(route[leg - 1])
                        all_b.append(route[leg])
                        all_actual.append(actual_leg)

                a_idx = cp.array(all_a, dtype=cp.intp)
                b_idx = cp.array(all_b, dtype=cp.intp)
                actual_costs = cp.array(all_actual, dtype=current_dist.dtype)
                est = current_dist[a_idx, b_idx]
                mask = actual_costs > est * (1.0 + cfg.feedback_threshold)
                current_dist[a_idx[mask], b_idx[mask]] = actual_costs[mask]
                current_dist[b_idx[mask], a_idx[mask]] = actual_costs[mask]
                updated = int(mask.sum())
                logger.info("      Updated %d distance matrix entries.", updated)

        logger.info("=" * 60)
        logger.info("Orchestrator complete.  Fail counts: %s  Actual makespan: %.1f s",
                    exec_result.fail_counts, exec_result.actual_makespan)
        logger.info("=" * 60)

        # ── Optional: save solution for later Isaac Sim replay ────────
        if cfg.save_solution_path:
            from .serialization import save_solution
            save_solution(exec_result, cfg.save_solution_path)
            logger.info("Solution saved to: %s", cfg.save_solution_path)

        # ── Optional: Isaac Sim replay ────────────────
        if cfg.replay_in_isaac:
            from ..scripts._isaac_replay import replay_in_isaac_sim
            replay_in_isaac_sim(exec_result, headless=cfg.headless)

        return exec_result
