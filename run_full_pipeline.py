#!/usr/bin/env python3
"""
Full Inspection Pipeline – Computation Script
==============================================

Runs the complete 3D-Inspection → VRP pipeline end-to-end and saves all
intermediate data so the visualisation script can replay the process in
Isaac Sim phase-by-phase.

Pipeline stages
---------------
1. Load & transform mesh (duke_of_lancaster_uk_clipped.glb → 50 m, VRP pose).
2. Sample 200 K surface points (pointcloud) and estimate outward normals.
3. Generate 1 500 candidate viewpoints outside the mesh.
4. Compute raycast visibility for every candidate (BVH + KD-tree frustum
   culling).
5. Greedy set-cover optimisation at 95 % target coverage.
6. Convert selected viewpoints to VRP waypoints [x,y,z,qw,qx,qy,qz].
7. Build occupancy grid (50 m mesh), compute distance matrix, solve VRP.
8. Execute routes via Space-Time A* → ``ExecutionResult``.
9. Pickle everything to ``pipeline_data.pkl``.

Usage
-----
::

    python run_full_pipeline.py                         # defaults
    python run_full_pipeline.py --output my_data.pkl    # custom output
    python run_full_pipeline.py --num_robots 3          # 3 AUVs
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import asdict
from time import time as get_time

import cupy as cp
import numpy as np

# ── Ensure repo root is on sys.path ──────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ── Also make visibility package importable ──────────────────────────────────
VISIBILITY_DIR = os.path.join(REPO_ROOT, "visibility")
if VISIBILITY_DIR not in sys.path:
    sys.path.insert(0, os.path.dirname(VISIBILITY_DIR))

from VRP.core import constants as _vrp_cfg

# The GLB file stores vertices in Y-up convention (glTF standard).
# Isaac Sim's GLB→USD converter implicitly prepends a Y-up→Z-up rotation
# (+90° about X) before any user xformOps, but trimesh loads raw coordinates.
# To get the same orientation from trimesh we compose:
#   R_combined = R_mesh_pose(180°X) @ R_y2z(+90°X) = R_x(270°) = R_x(-90°)
#   Quaternion [qw,qx,qy,qz] = [cos(-45°), sin(-45°), 0, 0]
import math as _math
_SQRT2_2 = _math.sqrt(2.0) / 2.0
_CORRECTED_MESH_POSE = list(_vrp_cfg.MESH_POSE[:3]) + [
    _SQRT2_2, -_SQRT2_2, 0.0, 0.0,
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
    p.add_argument("--output", "-o", default="pipeline_data.pkl",
                   help="Path for the output pickle file.")
    p.add_argument("--num_robots", "-n", type=int, default=5,
                   help="Number of AUV robots for VRP.")
    p.add_argument("--mesh_target_length", type=float, default=50.0,
                   help="Target length (m) of the ship along its longest axis.")
    p.add_argument("--num_surface_points", type=int, default=200_000,
                   help="Number of surface points to sample on the mesh.")
    p.add_argument("--num_candidates", type=int, default=1500,
                   help="Number of candidate viewpoints to generate.")
    p.add_argument("--target_coverage", type=float, default=0.95,
                   help="Greedy set cover target coverage (0–1).")
    p.add_argument("--frustum_near", type=float, default=0.1,
                   help="Frustum near plane (m).")
    p.add_argument("--frustum_far", type=float, default=6.0,
                   help="Frustum far plane (m).")
    p.add_argument("--frustum_fov_deg", type=float, default=40.0,
                   help="Frustum vertical FOV (degrees).")
    p.add_argument("--frustum_aspect", type=float, default=1.0,
                   help="Frustum aspect ratio (width/height).")
    p.add_argument("--solver", choices=["auto", "cuopt", "ortools"],
                   default="cuopt", help="MIP solver backend.")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Objective blending: 1.0=makespan, 0.0=total distance.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--curvature_weighting", action="store_true",
                   help="Enable curvature-weighted sampling (bias toward complex regions).")
    p.add_argument("--resample_fraction", type=float, default=0.5,
                   help="Fraction of candidates generated via targeted resampling "
                        "(0.0 = disabled, 0.25 = 25%% targeted). Default: 0.0")
    p.add_argument("--resampling_strategy", choices=["random", "optimal"],
                   default="optimal",
                   help="Targeted resampling strategy: 'random' (proximity-weighted) "
                        "or 'optimal' (CMA-ES).")
    p.add_argument("--k_coverage", type=int, default=1,
                   help="Coverage redundancy: sample until each point is covered "
                        "by at least k viewpoints (Glorieux 2020). Default: 1.")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _save_pipeline_data(pipeline_data: dict, path: str) -> None:
    """Save pipeline output as NPZ (arrays) + JSON (metadata).

    ExecutionResult is saved separately via :func:`VRP.utils.save_solution`.
    """
    import json as _json
    from VRP.core.serialization import save_solution

    base = path.rsplit(".", 1)[0] if "." in path else path
    os.makedirs(os.path.dirname(os.path.abspath(base)) or ".", exist_ok=True)

    # Save ExecutionResult separately
    save_solution(pipeline_data["exec_result"], base + "_exec")

    # Collect numpy arrays
    arrays = {}
    json_meta = {}
    skip_keys = {"exec_result", "optimization_result"}
    for k, v in pipeline_data.items():
        if k in skip_keys:
            continue
        if isinstance(v, np.ndarray):
            arrays[k] = v
        elif isinstance(v, list) and v and isinstance(v[0], np.ndarray):
            arrays[k] = np.array(v)
        else:
            json_meta[k] = v

    # Save optimization_result fields we need
    opt = pipeline_data.get("optimization_result")
    if opt is not None:
        json_meta["optimization_result"] = {
            "selected_indices": opt.selected_indices if hasattr(opt, "selected_indices") else [],
            "total_coverage": opt.total_coverage,
            "num_viewpoints": opt.num_viewpoints,
            "per_vp_coverage": (opt.visibility_map.sum(axis=1) / opt.num_viewpoints).get().tolist() if opt.num_viewpoints > 0 else [],
        }

    np.savez_compressed(base + ".npz", **arrays)
    with open(base + ".json", "w") as f:
        _json.dump(json_meta, f, default=str)


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

    MESH_TARGET_LENGTH = args.mesh_target_length
    MESH_PATH = _vrp_cfg.MESH_PATH
    # Use the rotation-corrected pose (trimesh Y-up → Z-up adjustment).
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
    logger.info("  Frustum       : near=%.2f far=%.1f fov=%.0f° aspect=%.1f",
                args.frustum_near, args.frustum_far,
                args.frustum_fov_deg, args.frustum_aspect)
    logger.info("=" * 70)

    t0 = get_time()

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1 – Load & transform mesh
    # ══════════════════════════════════════════════════════════════════════
    logger.info("[1/9] Loading and transforming mesh …")
    import open3d as o3d
    from shared.mesh_loader import load_and_transform_mesh

    raw_tm = load_and_transform_mesh(MESH_PATH, MESH_TARGET_LENGTH, MESH_POSE)
    mesh_scale = raw_tm.metadata.get("scale_factor", 1.0)

    mesh_bounds_min = raw_tm.bounds[0]
    mesh_bounds_max = raw_tm.bounds[1]
    logger.info("  Mesh scale=%.6f  bounds_min=%s  bounds_max=%s",
                mesh_scale, mesh_bounds_min.round(2), mesh_bounds_max.round(2))

    # Convert to Open3D TriangleMesh (for visibility pipeline)
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(raw_tm.vertices))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(raw_tm.faces))
    o3d_mesh.compute_vertex_normals()

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2 – Sample surface points
    # ══════════════════════════════════════════════════════════════════════
    logger.info("[2/9] Sampling %d surface points …", args.num_surface_points)
    from shared.surface_sampler import SurfacePointSampler
    surface_sampler = SurfacePointSampler()
    target_points, normals = surface_sampler.sample(
        o3d_mesh, args.num_surface_points, seed=args.seed,
    )
    logger.info("  Sampled %d points.  Normal estimation done.", len(target_points))

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2b – Build surface-only occupancy grid for sampling
    # ══════════════════════════════════════════════════════════════════════
    logger.info("[2b/9] Building surface-only occupancy grid for sampling …")
    from visibility.sampling.utils.sampling_grid_builder import build_sampling_occupancy_grid
    sampling_min_clearance = 2 * _vrp_cfg.ROBOT_RADIUS
    sampling_og = build_sampling_occupancy_grid(
        mesh=o3d_mesh,
        frustum_far=args.frustum_far,
        min_clearance=sampling_min_clearance,
        resolution=_vrp_cfg.VOXEL_RESOLUTION,
    )
    logger.info("  Sampling OG shape: %s  res=%.2f m  free=%d",
                sampling_og.grid.shape, sampling_og.resolution, sampling_og.num_free)

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 3 – Generate candidate viewpoints + filter below mesh
    # ══════════════════════════════════════════════════════════════════════
    logger.info("[3/9] Generating %d candidate viewpoints …", args.num_candidates)
    if args.curvature_weighting:
        logger.info("  Curvature weighting: ENABLED")
    if args.resample_fraction > 0:
        logger.info("  Targeted resampling: ENABLED (fraction=%.2f, strategy=%s)",
                    args.resample_fraction, args.resampling_strategy)

    from visibility.sampling import WeightedViewpointSampler, TargetedViewpointSampler, OptimizingSampler, CMAESBackend

    sampler = TargetedViewpointSampler(
        mesh=o3d_mesh,
        target_points=target_points,
        normals=normals,
        frustum_far=args.frustum_far,
        collision_radius=_vrp_cfg.ROBOT_RADIUS,
        occupancy_grid=sampling_og,
    )

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 4 – Build visibility query
    # ══════════════════════════════════════════════════════════════════════
    logger.info("[4/9] Building raycast visibility query …")

    from visibility.core.types import FrustumParams, OptimizationResult
    from visibility.methods.raycast_cuda import RaycastingVisibilityQueryCuda

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
    # STAGE 5 – Greedy set cover at target coverage
    # ══════════════════════════════════════════════════════════════════════
    if args.resample_fraction > 0:
        n_targeted = int(args.num_candidates * args.resample_fraction)
        n_uniform = args.num_candidates - n_targeted

        # Phase 1: uniform sampling
        logger.info("[3–5/9] Sampling %d uniform + %d targeted (%s) …",
                    n_uniform, n_targeted, args.resampling_strategy)
        pos_gpu, rot_gpu = sampler.sample(
            np.arange(len(target_points)),
            n_uniform, side="outside",
            curvature_weighting=args.curvature_weighting)
        V, _ = raycast_query.compute_visibility_batch(pos_gpu, rot_gpu)

        # Phase 2: build per-point coverage counts and identify under-covered
        coverage_count_gpu = V.astype(cp.int32).sum(axis=0)
        under_k_mask = coverage_count_gpu < args.k_coverage
        uncovered = cp.asnumpy(cp.where(under_k_mask)[0])

        if len(uncovered) > 0 and n_targeted > 0:
            if args.resampling_strategy == "optimal":
                # Optimisation-based iterative resampling
                opt_sampler = OptimizingSampler(
                    mesh=o3d_mesh, target_points=target_points,
                    normals=normals, frustum_far=args.frustum_far,
                    collision_radius=_vrp_cfg.ROBOT_RADIUS,
                    occupancy_grid=sampling_og,
                    backend=CMAESBackend(),
                )
                opt_pos_gpu, opt_rot_gpu = opt_sampler.sample_optimized(
                    n_targeted, coverage_count_gpu, raycast_query,
                    existing_pos_gpu=pos_gpu,
                    existing_rot_gpu=rot_gpu,
                    k_coverage=args.k_coverage)
                if len(opt_pos_gpu) > 0:
                    V_new, _ = raycast_query.compute_visibility_batch(opt_pos_gpu, opt_rot_gpu)
                    V = cp.concatenate([V, V_new])
                    pos_gpu = cp.concatenate([pos_gpu, opt_pos_gpu])
                    rot_gpu = cp.concatenate([rot_gpu, opt_rot_gpu])
            else:
                # Random targeted sampling (iterative for k-coverage tracking)
                targeted_pos_gpu, targeted_rot_gpu = sampler.sample(
                    uncovered, n_targeted, side="outside",
                    curvature_weighting=args.curvature_weighting,
                    visibility_query=raycast_query,
                    k_coverage=args.k_coverage,
                    coverage_count_gpu=coverage_count_gpu)
                if len(targeted_pos_gpu) > 0:
                    V_new, _ = raycast_query.compute_visibility_batch(targeted_pos_gpu, targeted_rot_gpu)
                    V = cp.concatenate([V, V_new])
                    pos_gpu = cp.concatenate([pos_gpu, targeted_pos_gpu])
                    rot_gpu = cp.concatenate([rot_gpu, targeted_rot_gpu])

        logger.info("  Total candidates after resampling: %d", len(pos_gpu))
    else:
        pos_gpu, rot_gpu = sampler.sample(
            np.arange(len(target_points)),
            num_candidates=args.num_candidates,
            side="outside",
            curvature_weighting=args.curvature_weighting,
        )
        logger.info("  Generated %d candidates.", len(pos_gpu))
        V, _ = raycast_query.compute_visibility_batch(pos_gpu, rot_gpu)

    logger.info("[5/9] Running greedy set cover at %.0f%% (%d candidates) …",
                args.target_coverage * 100, len(pos_gpu))
    from visibility.set_cover import LazyGreedySetCoverCuda
    optimizer = LazyGreedySetCoverCuda(len(target_points), pos_gpu, rot_gpu, V)
    opt_result: OptimizationResult = optimizer.optimize(
        target_coverage=args.target_coverage,
        max_viewpoints=1000,
    )
    logger.info("  Selected %d viewpoints.  Coverage=%.2f%%  Time=%.1f s",
                opt_result.num_viewpoints, opt_result.total_coverage * 100,
                opt_result.optimization_time)

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 6 – Convert selected viewpoints to VRP waypoints
    # ══════════════════════════════════════════════════════════════════════
    logger.info("[6/9] Converting %d viewpoints to VRP waypoints …",
                opt_result.num_viewpoints)

    from scipy.spatial.transform import Rotation
    positions_np = opt_result.positions.get()
    orientations_np = opt_result.orientations.get()
    selected_wp_7dof = np.zeros((opt_result.num_viewpoints, 7), dtype=np.float64)
    for i in range(opt_result.num_viewpoints):
        selected_wp_7dof[i, :3] = positions_np[i]
        selected_wp_7dof[i, 3:] = Rotation.from_matrix(orientations_np[i]).as_quat(scalar_first=True)

    logger.info("  Waypoints array shape: %s", selected_wp_7dof.shape)

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 7 – Occupancy grid + depot positions + distance matrix
    # ══════════════════════════════════════════════════════════════════════
    logger.info("[7/9] Building occupancy grid & distance matrix …")

    from VRP.core.occupancy_grid import get_mesh_world_bounds, OccupancyGrid
    from VRP.scripts.vrp_planner import _compute_start_grid
    from VRP.core.distance_matrix import compute_distance_matrix, build_route_path_cache

    mesh_bmin, mesh_bmax = get_mesh_world_bounds(
        mesh_target_length=MESH_TARGET_LENGTH,
        mesh_pose=MESH_POSE,
    )
    robot_start_xyzs = _compute_start_grid(
        args.num_robots, mesh_bmin, mesh_bmax
    )
    for i, p in enumerate(robot_start_xyzs):
        logger.info("  Robot %d depot: %s", i, p)

    # Reuse the sampling OG (SamplingOccupancyGrid inherits from OccupancyGrid).
    og = sampling_og
    logger.info("  Grid shape: %s  res=%.2f m", og.grid.shape, og.resolution)

    # Assemble VRP node array: [depots | inspection waypoints]
    K = args.num_robots
    home_poses = np.array(
        [[*xyz, 1.0, 0.0, 0.0, 0.0] for xyz in robot_start_xyzs],
        dtype=np.float32,
    )
    waypoints_world = np.vstack([home_poses, selected_wp_7dof.astype(np.float32)])
    home_indices = list(range(K))
    N_insp = len(selected_wp_7dof)
    logger.info("  VRP nodes: %d (%d homes + %d inspection)", len(waypoints_world), K, N_insp)

    # cuGraph graph memory scales with free-voxel count (~12 B/edge × 26 adj).
    # Find the finest integer downsampling factor that keeps free voxels under
    # budget so the distance matrix is as accurate as VRAM allows.
    from shared.grid_utils import downsample_occupancy_grid
    _MAX_FREE_VOXELS = 5_000_000   # ~1.6 GB edge list → safe on 8-GB cards

    def _pick_distmatrix_og(fine_og, max_free: int):
        """Return the finest downsampled OG with ≤ max_free free voxels."""
        # Check if the original grid already fits under budget
        free_count = int((~fine_og.grid).sum())
        logger.info(
            "  Dist-matrix OG candidate: factor=1 res=%.2fm grid=%s free=%d",
            fine_og.resolution, fine_og.grid.shape, free_count,
        )
        if free_count <= max_free:
            return fine_og, 1, free_count
        # Otherwise search for the coarsest factor that fits
        for factor in range(2, 32):
            target_res = fine_og.resolution * factor
            cg, co, cr = downsample_occupancy_grid(
                fine_og.grid, fine_og.origin, fine_og.resolution, target_res
            )
            free_count = int((~cg).sum())
            logger.info(
                "  Dist-matrix OG candidate: factor=%d res=%.2fm grid=%s free=%d",
                factor, cr, cg.shape, free_count,
            )
            if free_count <= max_free:
                return OccupancyGrid(grid=cg, origin=co, resolution=cr), factor, free_count
        return OccupancyGrid(grid=cg, origin=co, resolution=cr), 31, int((~cg).sum())

    _dm_og, _dm_factor, _dm_free = _pick_distmatrix_og(og, _MAX_FREE_VOXELS)
    logger.info(
        "  Using factor=%d (%.2f m) for dist-matrix (%d free voxels).",
        _dm_factor, _dm_og.resolution, _dm_free,
    )
    dist_matrix = compute_distance_matrix(_dm_og, waypoints_world[:, :3])
    logger.info("  Distance matrix computed.  max=%.2f m",
                float(np.max(dist_matrix[np.isfinite(dist_matrix)])))

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 8 – Solve VRP + execute routes
    # ══════════════════════════════════════════════════════════════════════
    logger.info("[8/9] Solving VRP (%s) and executing routes …", args.solver)

    from VRP.vrp.vrp_solver import solve_vrp
    from VRP.core.types import VRPResult, ExecutionResult
    from VRP.mapf.route_executor import RouteExecutor
    from VRP.core.robot_config import load_local_robot_config

    vrp_result: VRPResult = solve_vrp(
        dist_matrix=dist_matrix,
        num_vehicles=args.num_robots,
        depot=home_indices,
        alpha=args.alpha,
        backend=args.solver,
    )
    logger.info("  VRP status=%s  cost=%.2f  solver=%s",
                vrp_result.status, vrp_result.total_cost, vrp_result.solver)

    # Wrap routes with home nodes
    routes = [
        [home_indices[i]] + list(r) + [home_indices[i]]
        for i, r in enumerate(vrp_result.routes)
    ]
    logger.info("  Routes (with homes): %s", routes)

    # Build A* sub-waypoint path cache
    path_cache = build_route_path_cache(og, waypoints_world[:, :3], routes)

    # Build start configs (8-DOF: [x, y, z, yaw, pitch, roll, cam_yaw, cam_pitch])
    robot_cfg = load_local_robot_config("brov.yml")
    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_cfg = robot_cfg["kinematics"]["cspace"]["retract_config"]

    start_configs = []
    for i in range(args.num_robots):
        s = list(default_cfg)
        xyz = robot_start_xyzs[i]
        s[0], s[1], s[2] = float(xyz[0]), float(xyz[1]), float(xyz[2])
        start_configs.append(np.array(s, dtype=np.float32))

    executor = RouteExecutor(
        start_configs=start_configs,
        joint_names=j_names,
        og=og,
    )
    exec_result: ExecutionResult = executor.execute(
        routes=routes,
        waypoints_world=waypoints_world,
        path_cache=path_cache,
    )
    logger.info("  Execution done.  Fail counts: %s", exec_result.fail_counts)

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 9 – Build per-robot waypoint mapping + save everything
    # ══════════════════════════════════════════════════════════════════════
    logger.info("[9/9] Saving pipeline data to %s …", args.output)

    # Map: for each robot, which *inspection* waypoint indices it visits (0-based
    # into selected_wp_7dof / opt_result.viewpoints).
    robot_inspection_wp_indices: list[list[int]] = []
    for i, route in enumerate(vrp_result.routes):
        # route entries are global indices; subtract K to get inspection index
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
        "target_points": target_points,
        "normals": normals,
        # Candidates
        "all_positions": cp.asnumpy(pos_gpu),
        "all_rotmats": cp.asnumpy(rot_gpu),
        # Visibility
        "visibility_map": visibility_map,
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
        # VRP waypoints
        "selected_waypoints_7dof": selected_wp_7dof,
        # VRP
        "vrp_routes": vrp_result.routes,          # raw routes (no home)
        "vrp_routes_with_homes": routes,           # routes with home book-ends
        "num_robots": args.num_robots,
        "home_indices": home_indices,
        "robot_start_xyzs": robot_start_xyzs,
        "robot_inspection_wp_indices": robot_inspection_wp_indices,
        # Trajectories
        "exec_result": exec_result,
    }

    _save_pipeline_data(pipeline_data, args.output)

    elapsed = get_time() - t0
    logger.info("=" * 70)
    logger.info("Pipeline complete in %.1f s.  Saved to: %s", elapsed, args.output)
    logger.info("  Pointcloud   : %d points", len(target_points))
    logger.info("  Candidates   : %d", len(pos_gpu))
    logger.info("  Selected VPs : %d", opt_result.num_viewpoints)
    logger.info("  Coverage     : %.2f%%", opt_result.total_coverage * 100)
    logger.info("  Robots       : %d", args.num_robots)
    logger.info("  Traj steps   : %d",
                max(len(t) for t in exec_result.all_traj_positions))
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
