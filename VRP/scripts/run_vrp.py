#!/usr/bin/env python3
"""
VRP Planner - CLI Entry Point

Usage examples
--------------

Random 5 waypoints, 2 AUVs, save solution:
    python -m VRP.scripts.run_vrp --num_robots 2 --random_waypoints 5 --save_solution solution.pkl

Inside inspection (routes traverse mesh interior):
    python -m VRP.scripts.run_vrp --num_robots 2 --random_waypoints 5 --side inside
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Ensure the repository root is on PYTHONPATH so `VRP` package resolves
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import cupy as cp
import numpy as np
import open3d as o3d

from shared.mesh_loader import load_and_transform_mesh
from shared.surface_sampler import SurfacePointSampler
from shared.types import Side
from visibility.sampling import WeightedViewpointSampler
from visibility.sampling.utils.sampling_grid_builder import build_sampling_occupancy_grid
from VRP.core.constants import MESH_PATH, MESH_POSE, MESH_TARGET_LENGTH, ROBOT_RADIUS
from VRP.core.distance_matrix import compute_distance_matrix
from VRP.core.geometry import compute_start_grid
from VRP.core.types import ExecutionResult, VRPBackend, VRPResult
from VRP.mapf.mapf_planner import MultiAgentPathPlanner
from VRP.utils.collision import find_trajectory_collisions
from VRP.vrp.vrp_solver import solve_vrp

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collision-free multi-AUV VRP planner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Robot / simulation ─────────────────────────────────────────────
    p.add_argument("--num_robots", "-n", type=int, default=2, help="Number of AUV robots.")
    p.add_argument(
        "--side",
        choices=["outside", "inside"],
        default="outside",
        help="Inspection side: 'outside' (routes around mesh) or 'inside' (routes inside mesh).",
    )

    # ── Waypoint source ────────────────────────────────────────────────
    p.add_argument(
        "--random_waypoints",
        type=int,
        default=5,
        metavar="N",
        help="Sample N random collision-free waypoints.",
    )

    p.add_argument("--random_seed", type=int, default=42, help="Random seed for waypoint sampling.")

    # ── VRP solver ────────────────────────────────────────────────────
    p.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Objective blending: 1.0=pure makespan, 0.0=pure "
        "total distance, 0.5=balanced trade-off.",
    )
    p.add_argument(
        "--solver",
        choices=["cuopt", "highs"],
        default="highs",
        help="MIP backend: 'cuopt' (GPU) or 'highs' (CPU).",
    )
    p.add_argument(
        "--mip_time_limit", type=int, default=120, help="MIP solver time budget (seconds)."
    )
    p.add_argument(
        "--mip_gap",
        type=float,
        default=0.05,
        help="MIP solver relative optimality gap (0.05 = 5%%).",
    )

    # ── Solution persistence ──────────────────────────────────
    p.add_argument(
        "--save_solution",
        type=str,
        default="vrp_solution.pkl",
        metavar="PATH",
        help="Save the planned ExecutionResult to PATH (.pkl).",
    )

    # ── Logging ───────────────────────────────────────────────────────
    p.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging.")

    return p.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
    )

    side = Side(args.side)
    K = args.num_robots
    _FRUSTUM_FAR = 6.0
    _NUM_SURFACE_POINTS = 200_000
    _RESOLUTION = 0.20

    # ── 1. Load mesh ───────────────────────────────────────────────���─
    logger.info("Loading mesh …")
    mesh = load_and_transform_mesh(MESH_PATH, MESH_TARGET_LENGTH, MESH_POSE)
    mesh_bounds_min = np.asarray(mesh.bounds[0], dtype=float)
    mesh_bounds_max = np.asarray(mesh.bounds[1], dtype=float)

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    o3d_mesh.compute_vertex_normals()

    # ── 2. Build occupancy grid (reused for sampling + pathfinding) ──
    logger.info("Building occupancy grid (res=%.2f) …", _RESOLUTION)
    og = build_sampling_occupancy_grid(
        mesh=o3d_mesh,
        frustum_far=_FRUSTUM_FAR,
        min_clearance=2 * ROBOT_RADIUS,
        resolution=_RESOLUTION,
    )

    # ── 3. Sample viewpoints ─────────────────────────────────────────
    logger.info("Building WeightedViewpointSampler …")
    pts_np, norms_np = SurfacePointSampler().sample(
        o3d_mesh,
        _NUM_SURFACE_POINTS,
        seed=args.random_seed,
    )
    sampler = WeightedViewpointSampler(
        o3d_mesh,
        cp.asarray(pts_np, dtype=cp.float32),
        cp.asarray(norms_np, dtype=cp.float32),
        _FRUSTUM_FAR,
        collision_radius=ROBOT_RADIUS,
        occupancy_grid=og,
    )

    cp.random.seed(args.random_seed)
    pos_gpu, rot_gpu = sampler.sample(args.random_waypoints, side=side)
    insp_positions = cp.asnumpy(pos_gpu).astype(np.float32)
    insp_rotmats = cp.asnumpy(rot_gpu).astype(np.float32)
    logger.info("Sampled %d viewpoints (side=%s)", args.random_waypoints, side.value)

    # Free sampler (keep og for pathfinding)
    del sampler, pts_np, norms_np, o3d_mesh, mesh
    cp.get_default_memory_pool().free_all_blocks()

    # ── 4. Assemble VRP nodes (homes + inspection waypoints) ─────────
    robot_start_xyzs = compute_start_grid(K, mesh_bounds_min, mesh_bounds_max)
    home_positions = np.array(
        [[float(xyz[0]), float(xyz[1]), float(xyz[2])] for xyz in robot_start_xyzs],
        dtype=np.float32,
    )
    home_rotmats = np.tile(np.eye(3, dtype=np.float32), (K, 1, 1))
    all_positions = np.vstack([home_positions, insp_positions])
    all_rotmats = np.concatenate([home_rotmats, insp_rotmats])
    home_indices = list(range(K))

    # ── 5. Distance matrix ───────────────────────────────────────────
    logger.info("Computing %dx%d distance matrix …", len(all_positions), len(all_positions))
    dist_matrix = compute_distance_matrix(og, cp.asarray(all_positions))

    # ── 6. Solve VRP ─────────────────────────────────────────────────
    logger.info("Solving VRP (%s, alpha=%.2f) …", args.solver, args.alpha)
    vrp_result: VRPResult = solve_vrp(
        dist_matrix=dist_matrix,
        num_vehicles=K,
        depots=home_indices,
        alpha=args.alpha,
        backend=VRPBackend(args.solver),
        time_limit=args.mip_time_limit,
        mip_gap=args.mip_gap,
    )
    logger.info(
        "VRP status=%s  total_cost=%.2f  makespan=%.2f",
        vrp_result.status,
        vrp_result.total_cost,
        vrp_result.makespan,
    )

    if not any(vrp_result.routes):
        logger.error("VRP produced empty routes – aborting.")
        sys.exit(1)

    routes = [
        [home_indices[i]] + list(r) + [home_indices[i]] for i, r in enumerate(vrp_result.routes)
    ]
    logger.info("Routes: %s", routes)

    # ── 7. Execute trajectories (MAPF) ───────────────────────────────
    logger.info("Generating trajectories …")
    start_positions = [np.array(xyz, dtype=np.float32) for xyz in robot_start_xyzs]
    wp_pos_gpu = cp.asarray(all_positions, dtype=cp.float32)
    wp_rot_gpu = cp.asarray(all_rotmats, dtype=cp.float32)

    planner = MultiAgentPathPlanner(start_positions=start_positions, og=og)
    result: ExecutionResult = planner.execute(
        routes=routes,
        waypoint_positions=wp_pos_gpu,
        waypoint_rotmats=wp_rot_gpu,
        home_indices=set(home_indices),
        dist_matrix=dist_matrix,
        alpha=args.alpha,
    )

    residual_collisions = len(find_trajectory_collisions(result.all_traj_positions))

    # ── 8. Save solution ─────────────────────────────────────────────
    if args.save_solution:
        from VRP.utils.serialization import save_solution

        save_solution(result, args.save_solution)
        logger.info("Solution saved to: %s", args.save_solution)

    # ── 9. Summary ───────────────────────────────────────────────────
    total_wps = sum(len(r) for r in result.all_waypoints)
    total_fail = sum(result.fail_counts)
    total_steps = max(len(t) for t in result.all_traj_positions) if result.all_traj_positions else 0

    print("\n" + "=" * 60)
    print("VRP planning complete")
    print(f"  Robots             : {K}")
    print(f"  Waypoints          : {total_wps} total  ({total_fail} failed)")
    print(f"  Trajectory steps   : {total_steps}")
    print(f"  Residual collisions: {residual_collisions}")
    print(f"  VRP makespan       : {vrp_result.makespan:.1f} m")
    print(f"  VRP total cost     : {vrp_result.total_cost:.1f} m")
    if args.save_solution:
        print(f"  Solution saved     : {args.save_solution}")
    print("=" * 60)


if __name__ == "__main__":
    main()
