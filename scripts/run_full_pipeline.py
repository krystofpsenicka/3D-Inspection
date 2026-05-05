#!/usr/bin/env python3
"""
Full Inspection Pipeline - Computation Script
==============================================

Runs the complete 3D-Inspection -> VRP pipeline end-to-end and saves all
intermediate data into a directory so the visualisation script can replay the
process in Isaac Sim phase-by-phase.

Pipeline stages
---------------
1. Load & transform mesh (duke_of_lancaster_uk_clipped.glb -> 50 m, VRP pose).
2. Sample 200 K surface points (pointcloud) and estimate outward normals.
3. Generate 1 500 candidate viewpoints outside the mesh.
4. Compute raycast visibility for every candidate (BVH + KD-tree frustum
   culling).
5. Greedy set-cover optimisation at 95 % target coverage.
6. Convert selected viewpoints to VRP waypoints [x,y,z,qw,qx,qy,qz].
7. Build occupancy grid (50 m mesh), compute distance matrix, solve VRP.
8. Execute routes via Space-Time A* -> ``ExecutionResult``.
9. Save everything via ``VRP.utils.serialization.save_pipeline``.

Usage
-----
::

    python -m scripts.run_full_pipeline                          # outputs/full_pipeline
    python -m scripts.run_full_pipeline --output my_dir/         # custom directory
    python -m scripts.run_full_pipeline --num_robots 3           # 3 AUVs
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import cupy as cp
import numpy as np

# ── Ensure repo root is on sys.path ──────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# The GLB file stores vertices in Y-up convention (glTF standard).
# Isaac Sim's GLB->USD converter implicitly prepends a Y-up->Z-up rotation
# (+90 deg about X) before any user xformOps, but trimesh loads raw coordinates.
# To get the same orientation from trimesh we compose:
#   R_combined = R_mesh_pose(180 degX) @ R_y2z(+90 degX) = R_x(270 deg) = R_x(-90 deg)
#   Quaternion [qw,qx,qy,qz] = [cos(-45 deg), sin(-45 deg), 0, 0]
import math as _math

from shared.types import Side
from VRP.core import constants as _vrp_cfg

_SQRT2_2 = _math.sqrt(2.0) / 2.0
_CORRECTED_MESH_POSE = list(_vrp_cfg.MESH_POSE[:3]) + [
    _SQRT2_2,
    -_SQRT2_2,
    0.0,
    0.0,
]

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the full 3D-Inspection → VRP pipeline and save data "
        "for Isaac Sim visualisation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output",
        "-o",
        default=os.path.join(REPO_ROOT, "outputs", "full_pipeline"),
        help="Directory to write pipeline artefacts (overwritten on each run).",
    )
    p.add_argument("--num_robots", "-n", type=int, default=5, help="Number of AUV robots for VRP.")
    p.add_argument(
        "--mesh_target_length",
        type=float,
        default=50.0,
        help="Target length (m) of the ship along its longest axis.",
    )
    p.add_argument(
        "--num_surface_points",
        type=int,
        default=200_000,
        help="Number of surface points to sample on the mesh.",
    )
    p.add_argument(
        "--num_candidates",
        type=int,
        default=1500,
        help="Number of candidate viewpoints to generate.",
    )
    p.add_argument(
        "--target_coverage",
        type=float,
        default=0.95,
        help="Greedy set cover target coverage (0–1).",
    )
    p.add_argument("--frustum_near", type=float, default=0.1, help="Frustum near plane (m).")
    p.add_argument("--frustum_far", type=float, default=6.0, help="Frustum far plane (m).")
    p.add_argument(
        "--frustum_fov_deg", type=float, default=40.0, help="Frustum vertical FOV (degrees)."
    )
    p.add_argument(
        "--frustum_aspect", type=float, default=1.0, help="Frustum aspect ratio (width/height)."
    )
    p.add_argument(
        "--solver", choices=["cuopt", "highs"], default="cuopt", help="MIP solver backend."
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Objective blending: 1.0=makespan, 0.0=total distance.",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--curvature_weighting",
        action="store_true",
        help="Enable curvature-weighted sampling (bias toward complex regions).",
    )
    p.add_argument(
        "--resample_fraction",
        type=float,
        default=0.0,
        help="Fraction of candidates generated via targeted resampling "
        "(0.0 = disabled, 0.25 = 25%% targeted).",
    )
    p.add_argument(
        "--resampling_strategy",
        choices=["random", "optimal"],
        default="optimal",
        help="Targeted resampling strategy: 'random' (proximity-weighted) or 'optimal' (CMA-ES).",
    )
    p.add_argument(
        "--k_coverage",
        type=int,
        default=1,
        help="Coverage redundancy: sample until each point is covered "
        "by at least k viewpoints (Glorieux 2020).",
    )
    p.add_argument(
        "--side",
        choices=["outside", "inside"],
        default="outside",
        help="Inspection side: 'outside' (routes around mesh) or 'inside' (routes inside mesh).",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
    )
    np.random.seed(args.seed)
    cp.random.seed(args.seed)

    side = Side(args.side)

    MESH_TARGET_LENGTH = args.mesh_target_length
    MESH_PATH = _vrp_cfg.MESH_PATH
    # Use the rotation-corrected pose (trimesh Y-up -> Z-up adjustment).
    MESH_POSE = _CORRECTED_MESH_POSE

    logger.info("=" * 70)
    logger.info("FULL INSPECTION PIPELINE")
    logger.info("  Mesh          : %s", MESH_PATH)
    logger.info("  Mesh length   : %.1f m", MESH_TARGET_LENGTH)
    logger.info("  Mesh pose     : %s", MESH_POSE)
    logger.info("  Surface pts   : %d", args.num_surface_points)
    logger.info("  Candidates    : %d", args.num_candidates)
    logger.info("  Coverage      : %.0f%%", args.target_coverage * 100)
    logger.info("  Robots        : %d", args.num_robots)
    logger.info("  Output dir    : %s", args.output)
    logger.info(
        "  Frustum       : near=%.2f far=%.1f fov=%.0f° aspect=%.1f",
        args.frustum_near,
        args.frustum_far,
        args.frustum_fov_deg,
        args.frustum_aspect,
    )
    logger.info("=" * 70)

    t0 = time.perf_counter()

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1 - Load & transform mesh
    # ══════════════════════════════════════════════════════════════════════
    logger.info("[1/9] Loading and transforming mesh …")
    import open3d as o3d

    from shared.mesh_loader import load_and_transform_mesh

    raw_tm = load_and_transform_mesh(MESH_PATH, MESH_TARGET_LENGTH, MESH_POSE)
    mesh_scale = raw_tm.metadata.get("scale_factor", 1.0)

    mesh_bounds_min = raw_tm.bounds[0]
    mesh_bounds_max = raw_tm.bounds[1]
    logger.info(
        "  Mesh scale=%.6f  bounds_min=%s  bounds_max=%s",
        mesh_scale,
        mesh_bounds_min.round(2),
        mesh_bounds_max.round(2),
    )

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(raw_tm.vertices))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(raw_tm.faces))
    o3d_mesh.compute_vertex_normals()

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2 - Sample surface points
    # ══════════════════════════════════════════════════════════════════════
    logger.info("[2/9] Sampling %d surface points …", args.num_surface_points)
    from shared.surface_sampler import SurfacePointSampler

    surface_sampler = SurfacePointSampler()
    target_points_np, normals_np = surface_sampler.sample(
        o3d_mesh,
        args.num_surface_points,
        seed=args.seed,
    )
    target_points = cp.asarray(target_points_np, dtype=cp.float32)
    normals = cp.asarray(normals_np, dtype=cp.float32)
    if side == Side.INSIDE:
        normals = -normals
        logger.info("  Normals negated for inside inspection.")
    logger.info("  Sampled %d points.  Normal estimation done.", len(target_points))

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2b - Build surface-only occupancy grid for sampling
    # ══════════════════════════════════════════════════════════════════════
    logger.info("[2b/9] Building surface-only occupancy grid for sampling …")
    from visibility.sampling.utils.sampling_grid_builder import build_sampling_occupancy_grid

    sampling_min_clearance = 2 * _vrp_cfg.ROBOT_RADIUS
    sampling_og, _, _ = build_sampling_occupancy_grid(
        mesh=o3d_mesh,
        frustum_far=args.frustum_far,
        min_clearance=sampling_min_clearance,
        resolution=_vrp_cfg.VOXEL_RESOLUTION,
    )
    logger.info(
        "  Sampling OG shape: %s  res=%.2f m  free=%d",
        sampling_og.grid.shape,
        sampling_og.resolution,
        sampling_og.num_free,
    )

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 3 - Sampler + visibility query
    # ══════════════════════════════════════════════════════════════════════
    logger.info("[3/9] Generating %d candidate viewpoints …", args.num_candidates)
    if args.curvature_weighting:
        logger.info("  Curvature weighting: ENABLED")
    if args.resample_fraction > 0:
        logger.info(
            "  Targeted resampling: ENABLED (fraction=%.2f, strategy=%s)",
            args.resample_fraction,
            args.resampling_strategy,
        )

    from visibility.sampling import (
        CMAESBackend,
        OptimizingSampler,
        TargetedViewpointSampler,
    )

    sampler = TargetedViewpointSampler(
        mesh=o3d_mesh,
        target_points=target_points,
        normals=normals,
        frustum_far=args.frustum_far,
        collision_radius=_vrp_cfg.ROBOT_RADIUS,
        occupancy_grid=sampling_og,
    )

    logger.info("[4/9] Building raycast visibility query …")

    from visibility.core.types import FrustumParams, OptimizationResult
    from visibility.visibility.raycast_cuda import RaycastingVisibilityQueryCuda

    frustum_params = FrustumParams(
        fov_y=np.deg2rad(args.frustum_fov_deg),
        aspect=args.frustum_aspect,
        near=args.frustum_near,
        far=args.frustum_far,
    )

    raycast_query = RaycastingVisibilityQueryCuda(
        mesh=o3d_mesh,
        target_points=target_points,
        normals=normals,
        frustum_params=frustum_params,
    )

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 5 - Greedy set cover
    # ══════════════════════════════════════════════════════════════════════
    if args.resample_fraction > 0:
        n_targeted = int(args.num_candidates * args.resample_fraction)
        n_uniform = args.num_candidates - n_targeted

        logger.info(
            "[3–5/9] Sampling %d uniform + %d targeted (%s) …",
            n_uniform,
            n_targeted,
            args.resampling_strategy,
        )
        pos_gpu, rot_gpu = sampler.sample(
            cp.arange(len(target_points)),
            n_uniform,
            side=side,
            curvature_weighting=args.curvature_weighting,
        )
        V, _ = raycast_query.compute_visibility_batch(pos_gpu, rot_gpu)

        coverage_count_gpu = V.astype(cp.int32).sum(axis=0)
        under_k_mask = coverage_count_gpu < args.k_coverage
        uncovered = cp.where(under_k_mask)[0]

        if len(uncovered) > 0 and n_targeted > 0:
            if args.resampling_strategy == "optimal":
                opt_sampler = OptimizingSampler(
                    mesh=o3d_mesh,
                    target_points=target_points,
                    normals=normals,
                    frustum_far=args.frustum_far,
                    collision_radius=_vrp_cfg.ROBOT_RADIUS,
                    occupancy_grid=sampling_og,
                    backend=CMAESBackend(),
                    random_sampler=sampler,
                )
                opt_pos_gpu, opt_rot_gpu, _ = opt_sampler.sample_optimized(
                    n_targeted,
                    coverage_count_gpu,
                    raycast_query,
                    existing_pos_gpu=pos_gpu,
                    existing_rot_gpu=rot_gpu,
                    k_coverage=args.k_coverage,
                )
                if len(opt_pos_gpu) > 0:
                    V_new, _ = raycast_query.compute_visibility_batch(opt_pos_gpu, opt_rot_gpu)
                    V = cp.concatenate([V, V_new])
                    pos_gpu = cp.concatenate([pos_gpu, opt_pos_gpu])
                    rot_gpu = cp.concatenate([rot_gpu, opt_rot_gpu])
            else:
                targeted_pos_gpu, targeted_rot_gpu = sampler.sample(
                    uncovered,
                    n_targeted,
                    side=side,
                    curvature_weighting=args.curvature_weighting,
                    visibility_query=raycast_query,
                    k_coverage=args.k_coverage,
                    coverage_count_gpu=coverage_count_gpu,
                )
                if len(targeted_pos_gpu) > 0:
                    V_new, _ = raycast_query.compute_visibility_batch(
                        targeted_pos_gpu, targeted_rot_gpu
                    )
                    V = cp.concatenate([V, V_new])
                    pos_gpu = cp.concatenate([pos_gpu, targeted_pos_gpu])
                    rot_gpu = cp.concatenate([rot_gpu, targeted_rot_gpu])

        logger.info("  Total candidates after resampling: %d", len(pos_gpu))
    else:
        pos_gpu, rot_gpu = sampler.sample(
            cp.arange(len(target_points)),
            num_candidates=args.num_candidates,
            side=side,
            curvature_weighting=args.curvature_weighting,
        )
        logger.info("  Generated %d candidates.", len(pos_gpu))
        V, _ = raycast_query.compute_visibility_batch(pos_gpu, rot_gpu)

    logger.info(
        "[5/9] Running greedy set cover at %.0f%% (%d candidates) …",
        args.target_coverage * 100,
        len(pos_gpu),
    )
    from visibility.set_cover import LazyGreedySetCover

    pos_np = cp.asnumpy(pos_gpu)
    rot_np = cp.asnumpy(rot_gpu)
    V_np = cp.asnumpy(V)
    optimizer = LazyGreedySetCover(len(target_points), pos_np, rot_np, V_np)
    opt_result: OptimizationResult = optimizer.optimize(
        target_coverage=args.target_coverage,
        max_viewpoints=1000,
    )
    logger.info(
        "  Selected %d viewpoints.  Coverage=%.2f%%  Time=%.1f s",
        opt_result.num_viewpoints,
        opt_result.total_coverage * 100,
        opt_result.optimization_time,
    )

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 6 - Convert selected viewpoints to VRP waypoints
    # ══════════════════════════════════════════════════════════════════════
    logger.info("[6/9] Preparing %d selected viewpoints for VRP …", opt_result.num_viewpoints)

    selected_positions = opt_result.positions  # (K, 3) CuPy
    selected_rotmats = opt_result.rotations  # (K, 3, 3) CuPy
    logger.info("  Selected positions shape: %s", selected_positions.shape)

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 7 - Occupancy grid + depot positions + distance matrix
    # ══════════════════════════════════════════════════════════════════════
    logger.info("[7/9] Building occupancy grid & distance matrix …")

    from VRP.core.distance_matrix import compute_distance_matrix
    from VRP.core.geometry import compute_start_grid as _compute_start_grid

    mesh_bmin = mesh_bounds_min
    mesh_bmax = mesh_bounds_max
    robot_start_xyzs = _compute_start_grid(args.num_robots, mesh_bmin, mesh_bmax)
    for i, p in enumerate(robot_start_xyzs):
        logger.info("  Robot %d depot: %s", i, p)

    home_positions = cp.array(
        [[float(xyz[0]), float(xyz[1]), float(xyz[2])] for xyz in robot_start_xyzs],
        dtype=cp.float32,
    )
    waypoint_positions = cp.vstack([home_positions, selected_positions.astype(cp.float32)])

    from shared.grid_builder_utils import build_occupancy_grid as _build_routing_og

    _all_waypoints_np = cp.asnumpy(waypoint_positions).astype(np.float32)
    extra_margin = max(3, int(np.ceil(_vrp_cfg.ROBOT_RADIUS / _vrp_cfg.VOXEL_RESOLUTION)))
    og = _build_routing_og(
        mesh=raw_tm,
        padding=1.0,
        inflation_voxels=_vrp_cfg.INFLATION_VOXELS,
        resolution=_vrp_cfg.VOXEL_RESOLUTION,
        fill_interior=(side == Side.OUTSIDE),
        complement_fill=(side == Side.INSIDE),
        extra_free_points=_all_waypoints_np,
        extra_margin_voxels=extra_margin,
    )
    logger.info(
        "  Routing OG shape: %s  res=%.2f m  fill=%s complement=%s",
        og.grid.shape,
        og.resolution,
        side == Side.OUTSIDE,
        side == Side.INSIDE,
    )

    K = args.num_robots
    home_rotmats = cp.tile(cp.eye(3, dtype=cp.float32), (K, 1, 1))

    waypoint_rotmats = cp.concatenate([home_rotmats, selected_rotmats.astype(cp.float32)])
    home_indices = list(range(K))
    N_insp = opt_result.num_viewpoints
    logger.info("  VRP nodes: %d (%d homes + %d inspection)", len(waypoint_positions), K, N_insp)

    from shared.grid_utils import downsample_occupancy_grid

    _MAX_FREE_VOXELS = 5_000_000

    def _pick_distmatrix_og(fine_og, max_free: int):
        free_count = int((~fine_og.grid).sum())
        logger.info(
            "  Dist-matrix OG candidate: factor=1 res=%.2fm grid=%s free=%d",
            fine_og.resolution,
            fine_og.grid.shape,
            free_count,
        )
        if free_count <= max_free:
            return fine_og, 1, free_count
        for factor in range(2, 32):
            target_res = fine_og.resolution * factor
            coarse_og = downsample_occupancy_grid(fine_og, target_res)
            free_count = int((~coarse_og.grid).sum())
            logger.info(
                "  Dist-matrix OG candidate: factor=%d res=%.2fm grid=%s free=%d",
                factor,
                coarse_og.resolution,
                coarse_og.shape,
                free_count,
            )
            if free_count <= max_free:
                return coarse_og, factor, free_count
        return coarse_og, 31, int((~coarse_og.grid).sum())

    _dm_og, _dm_factor, _dm_free = _pick_distmatrix_og(og, _MAX_FREE_VOXELS)
    logger.info(
        "  Using factor=%d (%.2f m) for dist-matrix (%d free voxels).",
        _dm_factor,
        _dm_og.resolution,
        _dm_free,
    )
    dist_matrix = compute_distance_matrix(_dm_og, waypoint_positions)
    logger.info(
        "  Distance matrix computed.  max=%.2f m",
        float(cp.max(dist_matrix[cp.isfinite(dist_matrix)])),
    )

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 8 - Solve VRP + execute routes
    # ══════════════════════════════════════════════════════════════════════
    logger.info("[8/9] Solving VRP (%s) and executing routes …", args.solver)

    from VRP.core.types import ExecutionResult, VRPBackend, VRPResult
    from VRP.mapf.mapf_planner import MultiAgentPathPlanner
    from VRP.vrp.vrp_solver import solve_vrp

    vrp_backend = VRPBackend(args.solver)
    vrp_result: VRPResult = solve_vrp(
        dist_matrix=dist_matrix,
        num_vehicles=args.num_robots,
        depots=home_indices,
        alpha=args.alpha,
        backend=vrp_backend,
    )
    logger.info(
        "  VRP status=%s  cost=%.2f  solver=%s",
        vrp_result.status,
        vrp_result.total_cost,
        vrp_result.solver,
    )

    routes = [
        [home_indices[i]] + list(r) + [home_indices[i]] for i, r in enumerate(vrp_result.routes)
    ]
    logger.info("  Routes (with homes): %s", routes)

    start_positions = [cp.asarray(xyz, dtype=cp.float32) for xyz in robot_start_xyzs]
    executor = MultiAgentPathPlanner(start_positions=start_positions, og=og)
    exec_result: ExecutionResult = executor.execute(
        routes=routes,
        waypoint_positions=waypoint_positions,
        waypoint_rotmats=waypoint_rotmats,
        home_indices=set(home_indices),
        dist_matrix=dist_matrix,
        alpha=args.alpha,
    )
    logger.info("  Execution done.  Fail counts: %s", exec_result.fail_counts)

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 9 - Build per-robot waypoint mapping + save everything
    # ══════════════════════════════════════════════════════════════════════
    logger.info("[9/9] Saving pipeline data to %s …", args.output)

    robot_inspection_wp_indices: list[list[int]] = []
    for i, route in enumerate(vrp_result.routes):
        insp_idxs = [int(node - K) for node in route if node >= K]
        robot_inspection_wp_indices.append(insp_idxs)

    pipeline_data = {
        # Mesh
        "mesh_scale": mesh_scale,
        "mesh_pose": list(MESH_POSE),
        "mesh_target_length": MESH_TARGET_LENGTH,
        "mesh_path": MESH_PATH,
        "mesh_bounds_min": mesh_bounds_min,
        "mesh_bounds_max": mesh_bounds_max,
        # Pointcloud
        "target_points": cp.asnumpy(target_points),
        "normals": cp.asnumpy(normals),
        # Candidates (full set + full visibility matrix)
        "all_positions": cp.asnumpy(pos_gpu),
        "all_rotmats": cp.asnumpy(rot_gpu),
        "full_visibility_map": cp.asnumpy(V),
        # Frustum
        "frustum_params": {
            "fov_deg": args.frustum_fov_deg,
            "fov_y_rad": float(np.deg2rad(args.frustum_fov_deg)),
            "aspect": args.frustum_aspect,
            "near": args.frustum_near,
            "far": args.frustum_far,
        },
        # Set cover
        "optimization_result": opt_result,
        "selected_positions": cp.asnumpy(selected_positions),
        "selected_rotmats": cp.asnumpy(selected_rotmats),
        # VRP
        "vrp_routes": [list(r) for r in vrp_result.routes],
        "vrp_routes_with_homes": [list(r) for r in routes],
        "num_robots": args.num_robots,
        "home_indices": home_indices,
        "robot_start_xyzs": [np.asarray(xyz, dtype=np.float32) for xyz in robot_start_xyzs],
        "robot_inspection_wp_indices": robot_inspection_wp_indices,
        "vrp_status": vrp_result.status,
        "vrp_total_cost": float(vrp_result.total_cost),
        "vrp_makespan": float(vrp_result.makespan),
        "vrp_solver": vrp_result.solver,
        "alpha": args.alpha,
        # Trajectories
        "exec_result": exec_result,
        # CLI snapshot
        "args": vars(args),
    }

    from VRP.utils.serialization import save_pipeline

    save_pipeline(pipeline_data, args.output)

    elapsed = time.perf_counter() - t0
    logger.info("=" * 70)
    logger.info("Pipeline complete in %.1f s.  Saved to: %s", elapsed, args.output)
    logger.info("  Pointcloud   : %d points", len(target_points))
    logger.info("  Candidates   : %d", len(pos_gpu))
    logger.info("  Selected VPs : %d", opt_result.num_viewpoints)
    logger.info("  Coverage     : %.2f%%", opt_result.total_coverage * 100)
    logger.info("  Robots       : %d", args.num_robots)
    logger.info("  Traj steps   : %d", max(len(t) for t in exec_result.all_traj_positions))
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
