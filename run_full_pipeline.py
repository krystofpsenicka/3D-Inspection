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
3. Generate 1 500 candidate viewpoints outside the mesh; filter any that
   fall below the hull.
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
import pickle
import sys
from dataclasses import asdict
from time import time as get_time

import numpy as np

# ── Ensure repo root is on sys.path ──────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ── Also make 3D-Inspection/methods_analysis importable ──────────────────────
METHODS_DIR = os.path.join(REPO_ROOT, "3D-Inspection", "methods_analysis")
if METHODS_DIR not in sys.path:
    sys.path.insert(0, METHODS_DIR)

# ── Override VRP config BEFORE importing VRP modules ─────────────────────────
# We need the mesh to be 50 m, not 40 m.
import VRP.config as _vrp_cfg

_ORIGINAL_MESH_TARGET_LENGTH = _vrp_cfg.MESH_TARGET_LENGTH

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
    p.add_argument("--frustum_fov_deg", type=float, default=30.0,
                   help="Frustum vertical FOV (degrees).")
    p.add_argument("--frustum_aspect", type=float, default=1.0,
                   help="Frustum aspect ratio (width/height).")
    p.add_argument("--solver", choices=["auto", "cuopt", "ortools"],
                   default="auto", help="VRP solver backend.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _direction_to_quaternion(direction: np.ndarray) -> np.ndarray:
    """Convert a unit view-direction vector to quaternion [qw, qx, qy, qz].

    The convention is: the camera looks along +X in its local frame, so we
    compute the rotation that maps +X → *direction*.
    """
    from scipy.spatial.transform import Rotation as R

    d = direction / (np.linalg.norm(direction) + 1e-12)
    forward = np.array([1.0, 0.0, 0.0])

    cross = np.cross(forward, d)
    cross_norm = np.linalg.norm(cross)

    if cross_norm < 1e-8:
        # Parallel or anti-parallel
        if np.dot(forward, d) > 0:
            return np.array([1.0, 0.0, 0.0, 0.0])  # identity
        else:
            # 180° rotation about any perpendicular axis
            return np.array([0.0, 0.0, 1.0, 0.0])  # 180° about Y

    axis = cross / cross_norm
    angle = np.arccos(np.clip(np.dot(forward, d), -1.0, 1.0))
    rot = R.from_rotvec(axis * angle)
    qx, qy, qz, qw = rot.as_quat()  # scipy returns [x,y,z,w]
    return np.array([qw, qx, qy, qz])


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

    MESH_TARGET_LENGTH = args.mesh_target_length
    MESH_PATH = _vrp_cfg.MESH_PATH
    # Use the rotation-corrected pose (trimesh Y-up → Z-up adjustment).
    MESH_POSE = _CORRECTED_MESH_POSE

    # Patch VRP config so every downstream module (occupancy_grid, etc.)
    # uses the correct scale and corrected pose.
    _vrp_cfg.MESH_TARGET_LENGTH = MESH_TARGET_LENGTH
    _vrp_cfg.MESH_POSE = MESH_POSE
    # Recompute inflation voxels for consistency.
    _vrp_cfg.INFLATION_VOXELS = int(_vrp_cfg.ROBOT_RADIUS / _vrp_cfg.VOXEL_RESOLUTION) + 1

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
    import trimesh
    import open3d as o3d
    from scipy.spatial.transform import Rotation as Rot

    raw_tm = trimesh.load(MESH_PATH, force="mesh")
    if isinstance(raw_tm, trimesh.Scene):
        raw_tm = trimesh.util.concatenate(list(raw_tm.geometry.values()))

    longest = float(raw_tm.extents.max())
    mesh_scale = MESH_TARGET_LENGTH / longest if longest > 0 else 1.0
    raw_tm.apply_scale(mesh_scale)

    T_pose = np.eye(4)
    T_pose[:3, 3] = MESH_POSE[:3]
    qw, qx, qy, qz = MESH_POSE[3:7]
    T_pose[:3, :3] = Rot.from_quat([qx, qy, qz, qw]).as_matrix()
    raw_tm.apply_transform(T_pose)

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
    pcd = o3d_mesh.sample_points_poisson_disk(
        number_of_points=args.num_surface_points
    )
    # Estimate outward normals.  Radius scaled for the 50 m mesh.
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30)
    )
    # Orient normals to point outward (away from mesh interior).
    pcd.orient_normals_consistent_tangent_plane(k=15)
    target_points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    logger.info("  Sampled %d points.  Normal estimation done.", len(target_points))

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 3 – Generate candidate viewpoints + filter below mesh
    # ══════════════════════════════════════════════════════════════════════
    logger.info("[3/9] Generating %d candidate viewpoints …", args.num_candidates)

    # Import from 3D-Inspection (now on sys.path)
    from sampling import ViewpointSampler  # type: ignore

    sampler = ViewpointSampler(
        mesh=o3d_mesh,
        target_points=target_points,
        normals=normals,
        frustum_far=args.frustum_far,
    )
    candidates = sampler.sample_outside_mesh(
        num_candidates=args.num_candidates,
        offset_scale=1.0,
        pos_noise_std=0.3,    # scaled up for 50 m mesh (was 0.05)
        dir_noise_std=0.01,
    )
    logger.info("  Generated %d candidates.", len(candidates))

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 4 – Compute raycast visibility for all candidates
    # ══════════════════════════════════════════════════════════════════════
    logger.info("[4/9] Computing raycast visibility for %d candidates …", len(candidates))

    # FrustumParams from the 3D-Inspection utils (uses fov_y in radians, aspect, near, far)
    from utils import FrustumParams, ViewpointResult, OptimizationResult  # type: ignore
    from raycast_visibility import RaycastingVisibilityQuery  # type: ignore

    frustum_params = FrustumParams(
        fov_y=np.deg2rad(args.frustum_fov_deg),
        aspect=args.frustum_aspect,
        near=args.frustum_near,
        far=args.frustum_far,
    )

    raycast_query = RaycastingVisibilityQuery(
        mesh=o3d_mesh,
        target_points=target_points,
        normals=normals,
        frustum_params=frustum_params,
    )

    visibility_map, vis_time = raycast_query.compute_visibility_for_all_candidates(
        candidates
    )
    logger.info("  Visibility computed in %.1f s.  %d candidates evaluated.",
                vis_time, len(visibility_map))

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 5 – Greedy set cover at target coverage
    # ══════════════════════════════════════════════════════════════════════
    logger.info("[5/9] Running greedy set cover at %.0f%% …", args.target_coverage * 100)

    from greedy_optimizer_with_kernel import GreedyOptimizer  # type: ignore

    optimizer = GreedyOptimizer(raycast_query)
    opt_result: OptimizationResult = optimizer.optimize(
        candidates=candidates,
        target_coverage=args.target_coverage,
        max_viewpoints=1000,
    )
    logger.info("  Selected %d viewpoints.  Coverage=%.2f%%  Time=%.1f s",
                opt_result.num_viewpoints, opt_result.total_coverage * 100,
                opt_result.total_time)

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 6 – Convert selected viewpoints to VRP waypoints
    # ══════════════════════════════════════════════════════════════════════
    logger.info("[6/9] Converting %d viewpoints to VRP waypoints …",
                opt_result.num_viewpoints)

    selected_wp_7dof = np.zeros((opt_result.num_viewpoints, 7), dtype=np.float64)
    for i, vp in enumerate(opt_result.viewpoints):
        selected_wp_7dof[i, :3] = vp.position
        selected_wp_7dof[i, 3:] = _direction_to_quaternion(vp.direction)

    logger.info("  Waypoints array shape: %s", selected_wp_7dof.shape)

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 7 – Occupancy grid + depot positions + distance matrix
    # ══════════════════════════════════════════════════════════════════════
    logger.info("[7/9] Building occupancy grid & distance matrix …")

    from VRP.occupancy_grid import build_occupancy_grid, get_mesh_world_bounds, OccupancyGrid
    from VRP.vrp_planner import _compute_start_grid
    from VRP.gpu_distance_matrix import compute_distance_matrix, build_route_path_cache

    mesh_bmin, mesh_bmax = get_mesh_world_bounds()
    robot_start_xyzs = _compute_start_grid(
        args.num_robots, mesh_bmin, mesh_bmax
    )
    for i, p in enumerate(robot_start_xyzs):
        logger.info("  Robot %d depot: %s", i, p)

    # Extend the OG bounds to cover ALL inspection viewpoint apices, not just
    # the robot depots.  Without this, viewpoints above/beside the ship (apex
    # at surface + frustum_far ≈ 6 m from the hull) fall outside the grid and
    # both the fine A* (path cache) and the coarse space-time A* clip the goal
    # to the grid boundary — causing AUVs to stop 4-5 m short of their targets.
    extra_free_pts = np.vstack([
        np.array(robot_start_xyzs, dtype=np.float32),
        selected_wp_7dof[:, :3].astype(np.float32),
    ])
    og = build_occupancy_grid(
        extra_free_points=extra_free_pts,
    )
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
    from VRP.space_time_astar import downsample_occupancy_grid
    _MAX_FREE_VOXELS = 5_000_000   # ~1.6 GB edge list → safe on 8-GB cards

    def _pick_distmatrix_og(fine_og, max_free: int):
        """Return the finest downsampled OG with ≤ max_free free voxels."""
        cg, co, cr = fine_og.grid, fine_og.origin, fine_og.resolution
        for factor in range(1, 32):
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

    from VRP.vrp_solver import solve_vrp, VRPResult
    from VRP.route_executor import RouteExecutor, ExecutionResult
    from VRP.utils import load_local_robot_config

    vrp_result: VRPResult = solve_vrp(
        dist_matrix=dist_matrix,
        num_vehicles=args.num_robots,
        depot=home_indices,
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
        "all_candidates": candidates,   # List[(pos, dir)]
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

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(pipeline_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = get_time() - t0
    logger.info("=" * 70)
    logger.info("Pipeline complete in %.1f s.  Saved to: %s", elapsed, args.output)
    logger.info("  Pointcloud   : %d points", len(target_points))
    logger.info("  Candidates   : %d", len(candidates))
    logger.info("  Selected VPs : %d", opt_result.num_viewpoints)
    logger.info("  Coverage     : %.2f%%", opt_result.total_coverage * 100)
    logger.info("  Robots       : %d", args.num_robots)
    logger.info("  Traj steps   : %d",
                max(len(t) for t in exec_result.all_traj_positions))
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
