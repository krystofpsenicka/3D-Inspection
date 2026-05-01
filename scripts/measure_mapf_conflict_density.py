#!/usr/bin/env python3
"""Measure MAPF conflict density on pure-VRP trajectories.

For each (fleet_size, seed) configuration we:

1. Run the VRP stage (no MAPF) to obtain per-robot routes.
2. Plan each robot's space-time trajectory with an *empty* reservation table
   so robots ignore each other (the pre-MAPF counterfactual).
3. Count pairwise collision events between the resulting trajectories using
   the project's standard ``find_trajectory_collisions`` helper.

Reports, per fleet size:

* mean number of collision events,
* mean collision events per robot-pair,
* fraction of pairs that have at least one collision event,
* fraction of fleets with at least one collision event.

Output is written to ``results/mapf_conflict_density.csv`` and a one-line
summary is logged.

Usage:
    conda run -n isaaclab python -m scripts.measure_mapf_conflict_density
    conda run -n isaaclab python -m scripts.measure_mapf_conflict_density \
        --fleet_sizes 2 3 4 5 6 7 8 9 10 --n_waypoints 100 --seeds 1 2 3
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import cupy as cp
import open3d as o3d

from experiments.common.config import RESULTS_DIR, ModelConfig
from experiments.common.runner import free_gpu_memory
from shared.grid_utils import downsample_occupancy_grid
from shared.mesh_loader import load_and_transform_mesh
from shared.surface_sampler import SurfacePointSampler
from shared.types import Side
from visibility.sampling import WeightedViewpointSampler
from visibility.sampling.utils.sampling_grid_builder import build_sampling_occupancy_grid
from VRP.core.constants import (
    MESH_PATH,
    MESH_POSE,
    MESH_TARGET_LENGTH,
    ROBOT_RADIUS,
    SPACE_TIME_DT,
    SPACE_TIME_DWELL_S,
    SPACE_TIME_MAX_HORIZON_S,
    SPACE_TIME_RESOLUTION,
    SPLINE_SAFETY_VOXELS,
)
from VRP.core.distance_matrix import compute_distance_matrix
from VRP.core.geometry import compute_start_grid
from VRP.core.types import VRPBackend, VRPResult
from VRP.mapf.reservation_table import ReservationTable
from VRP.mapf.route_planner import plan_robot_route_st
from VRP.utils.collision import find_trajectory_collisions
from VRP.vrp.vrp_solver import solve_vrp

logger = logging.getLogger(__name__)


@dataclass
class Row:
    fleet_size: int = 0
    n_waypoints: int = 0
    seed: int = 0
    n_pairs: int = 0
    n_pairs_in_collision: int = 0
    n_collision_events: int = 0
    has_collision: bool = False


def _build_setup(resolution: float = 0.20):
    """Mirror the setup of e07_vrp_fleet_scaling._build_setup."""
    mesh = load_and_transform_mesh(MESH_PATH, MESH_TARGET_LENGTH, MESH_POSE)
    mesh_bounds_min = np.asarray(mesh.bounds[0], dtype=float)
    mesh_bounds_max = np.asarray(mesh.bounds[1], dtype=float)

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    o3d_mesh.compute_vertex_normals()

    cfg = ModelConfig.duke_of_lancaster()
    og = build_sampling_occupancy_grid(
        mesh=o3d_mesh,
        frustum_far=cfg.frustum.far,
        min_clearance=2 * ROBOT_RADIUS,
        resolution=resolution,
    )
    pts_np, norms_np = SurfacePointSampler().sample(o3d_mesh, cfg.num_surface_points, seed=42)
    sampler = WeightedViewpointSampler(
        o3d_mesh,
        cp.asarray(pts_np, dtype=cp.float32),
        cp.asarray(norms_np, dtype=cp.float32),
        cfg.frustum.far,
        collision_radius=ROBOT_RADIUS,
        occupancy_grid=og,
    )
    return og, sampler, mesh_bounds_min, mesh_bounds_max


def _plan_independently(
    routes, waypoint_positions_gpu, coarse_og, max_time_steps, collision_radius_vox
):
    """Plan each robot through its route IGNORING inter-robot conflicts.

    A fresh empty reservation table is created per robot so commits made by
    earlier robots do not influence later robots' searches.
    """
    per_robot_paths: list = []
    per_robot_times: list = []
    for route in routes:
        fresh = ReservationTable(coarse_og.shape, max_time_steps, collision_radius_vox)
        world_xyz, coarse_t, _wp, _stats = plan_robot_route_st(
            coarse_og,
            fresh,
            route,
            waypoint_positions_gpu,
            dwell_s=SPACE_TIME_DWELL_S,
            dt=SPACE_TIME_DT,
            fine_occupancy_grid=None,
            robot_radius=ROBOT_RADIUS,
        )
        per_robot_paths.append(world_xyz)
        per_robot_times.append(coarse_t)
    return per_robot_paths, per_robot_times


def _align_in_time(per_robot_paths, per_robot_times) -> list[cp.ndarray]:
    """Build dense (T, 3) per-robot trajectories on a common coarse-time axis.

    Each robot's coarse path has its own time-step samples; for collision
    detection we resample each robot onto the union time axis (held at the
    last position once the robot's mission ends).
    """
    if not per_robot_paths:
        return []

    max_t = int(max(int(t[-1]) for t in per_robot_times if len(t) > 0))
    common_t = cp.arange(max_t + 1, dtype=cp.float32)

    out: list[cp.ndarray] = []
    for world_xyz, coarse_t in zip(per_robot_paths, per_robot_times, strict=False):
        if len(coarse_t) == 0:
            out.append(cp.zeros((max_t + 1, 3), dtype=cp.float32))
            continue
        t_f = coarse_t.astype(cp.float32)
        # Drop duplicate time samples (e.g. when a robot dwells)
        unique_t, idx = cp.unique(t_f, return_index=True)
        idx = cp.sort(idx)
        unique_t = t_f[idx]
        unique_xyz = world_xyz[idx]
        traj = cp.column_stack(
            [cp.interp(common_t, unique_t, unique_xyz[:, d]) for d in range(3)]
        ).astype(cp.float32)
        out.append(traj)
    return out


def measure_one(fleet_size, n_waypoints, seed, og, sampler, bmin, bmax) -> Row:
    cp.random.seed(seed)
    np.random.seed(seed)

    pos_gpu, _ = sampler.sample(n_waypoints, side=Side.OUTSIDE)
    insp_positions = cp.asnumpy(pos_gpu).astype(np.float32)
    K = fleet_size

    robot_start_xyzs = compute_start_grid(K, bmin, bmax)
    home_positions = np.array(
        [[float(xyz[0]), float(xyz[1]), float(xyz[2])] for xyz in robot_start_xyzs],
        dtype=np.float32,
    )
    all_positions = np.vstack([home_positions, insp_positions])
    home_indices = list(range(K))
    dist_matrix = compute_distance_matrix(og, cp.asarray(all_positions))

    vrp_result: VRPResult = solve_vrp(
        dist_matrix=dist_matrix,
        num_vehicles=K,
        depots=home_indices,
        alpha=0.5,
        backend=VRPBackend.CUOPT,
        time_limit=30,
    )
    if not any(vrp_result.routes):
        return Row(fleet_size=K, n_waypoints=n_waypoints, seed=seed, n_pairs=K * (K - 1) // 2)

    routes = [
        [home_indices[i]] + list(r) + [home_indices[i]] for i, r in enumerate(vrp_result.routes)
    ]

    coarse_og = downsample_occupancy_grid(og, coarse_res=SPACE_TIME_RESOLUTION)
    max_time_steps = max(1, int(math.ceil(SPACE_TIME_MAX_HORIZON_S / SPACE_TIME_DT)))
    collision_radius_vox = ROBOT_RADIUS / coarse_og.resolution + SPLINE_SAFETY_VOXELS

    wp_pos_gpu = cp.asarray(all_positions, dtype=cp.float32)
    paths, times = _plan_independently(
        routes,
        wp_pos_gpu,
        coarse_og,
        max_time_steps,
        collision_radius_vox,
    )
    aligned = _align_in_time(paths, times)
    if not aligned or all(len(t) == 0 for t in aligned):
        return Row(fleet_size=K, n_waypoints=n_waypoints, seed=seed, n_pairs=K * (K - 1) // 2)

    events = find_trajectory_collisions(aligned, radius=ROBOT_RADIUS)
    pairs_in_collision = {(a, b) for (_step, a, b, _pen) in events}
    return Row(
        fleet_size=K,
        n_waypoints=n_waypoints,
        seed=seed,
        n_pairs=K * (K - 1) // 2,
        n_pairs_in_collision=len(pairs_in_collision),
        n_collision_events=len(events),
        has_collision=len(events) > 0,
    )


def main():
    p = argparse.ArgumentParser(description="MAPF conflict density measurement")
    p.add_argument("--fleet_sizes", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8, 9, 10])
    p.add_argument("--n_waypoints", type=int, default=100)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--output_dir", default=os.path.join(RESULTS_DIR, "mapf_conflict_density"))
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "results.csv")

    og, sampler, bmin, bmax = _build_setup()

    rows: list[Row] = []
    fields = list(Row.__dataclass_fields__.keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for k in args.fleet_sizes:
            for seed in args.seeds:
                t0 = time.perf_counter()
                try:
                    row = measure_one(k, args.n_waypoints, seed, og, sampler, bmin, bmax)
                except Exception as e:
                    logger.error("fleet=%d seed=%d failed: %s", k, seed, e, exc_info=True)
                    row = Row(
                        fleet_size=k,
                        n_waypoints=args.n_waypoints,
                        seed=seed,
                        n_pairs=k * (k - 1) // 2,
                    )
                logger.info(
                    "fleet=%d seed=%d  events=%d  pairs_in_collision=%d/%d  (%.1fs)",
                    k,
                    seed,
                    row.n_collision_events,
                    row.n_pairs_in_collision,
                    row.n_pairs,
                    time.perf_counter() - t0,
                )
                rows.append(row)
                writer.writerow(asdict(row))
                f.flush()
                free_gpu_memory()

    # ── Summary ───────────────────────────────────────────────────────
    by_fleet: dict[int, list[Row]] = {}
    for r in rows:
        by_fleet.setdefault(r.fleet_size, []).append(r)

    logger.info("=" * 64)
    logger.info("Pure-VRP conflict density (n_waypoints=%d)", args.n_waypoints)
    logger.info(
        "%-6s %-9s %-10s %-12s %-12s",
        "fleet",
        "events",
        "ev/pair",
        "pair-coll-frac",
        "fleet-coll-frac",
    )
    for k in sorted(by_fleet):
        rs = by_fleet[k]
        events_mean = float(np.mean([r.n_collision_events for r in rs]))
        ev_per_pair = events_mean / max(1, k * (k - 1) // 2)
        pair_coll_frac = float(np.mean([r.n_pairs_in_collision / max(1, r.n_pairs) for r in rs]))
        fleet_coll_frac = float(np.mean([1.0 if r.has_collision else 0.0 for r in rs]))
        logger.info(
            "%-6d %-9.1f %-10.2f %-12.2f %-12.2f",
            k,
            events_mean,
            ev_per_pair,
            pair_coll_frac,
            fleet_coll_frac,
        )


if __name__ == "__main__":
    main()
