#!/usr/bin/env python3
"""E11: VRP cut ablation - isolate the two original VRP additions.

Reviewer request (R2.5 / R4.4): the paper lists a β-aware per-vehicle tour
upper bound (used as a Desrochers--Laporte reachability arc filter) and
forbidden-pair cuts among its contributions, but never shows their benefit.

This experiment builds one VRP instance per (seed, waypoint count) and solves it
four times under a fixed time budget with the two cuts toggled independently:

    none  - plain lifted MTZ (both cuts off)
    reach - β-aware reachability arc filter only
    pair  - forbidden-pair cuts only
    both  - full model (default)

For each configuration it records solve time, the blended objective, makespan,
total cost, the solver dual bound, and the achieved optimality gap - so the
table shows whether the cuts tighten the bound / reduce the gap / speed the
solve at equal wall-clock budget.

    python -m experiments.e11_vrp_cuts_ablation
    python -m experiments.e11_vrp_cuts_ablation --waypoints 30 50 --seeds 42 123 7 2024 314
    python -m experiments.e11_vrp_cuts_ablation --plots_only
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.common.config import RESULTS_DIR, SEEDS_5, ModelConfig
from experiments.common.persistence import load_run_result, save_run_result

_RUNTIME_IMPORT_ERROR: ImportError | None = None
try:
    import cupy as cp
    import open3d as o3d

    from experiments.common.runner import free_gpu_memory, handle_row_exception
    from shared.mesh_loader import load_and_transform_mesh
    from shared.surface_sampler import SurfacePointSampler
    from shared.types import Side
    from visibility.sampling import WeightedViewpointSampler
    from visibility.sampling.utils.sampling_grid_builder import build_sampling_occupancy_grid
    from VRP.core.constants import MESH_PATH, MESH_POSE, MESH_TARGET_LENGTH, ROBOT_RADIUS
    from VRP.core.distance_matrix import compute_distance_matrix
    from VRP.core.geometry import compute_start_grid
    from VRP.core.types import VRPBackend
    from VRP.vrp._helpers import per_vehicle_costs
    from VRP.vrp.vrp_solver import solve_vrp

    _RUNTIME_AVAILABLE = True
except ImportError as _e:
    cp = None
    _RUNTIME_AVAILABLE = False
    _RUNTIME_IMPORT_ERROR = _e

logger = logging.getLogger(__name__)

N_ROBOTS = 5
VRP_ALPHA = 0.5
TIME_LIMIT = 120
MIP_GAP = 0.05

# (label, beta_aware_filter, forbidden_pair_cuts)
CUT_CONFIGS = [
    ("none", False, False),
    ("reach", True, False),
    ("pair", False, True),
    ("both", True, True),
]


def _build_instance(n_waypoints: int, seed: int, og, sampler, bmin, bmax):
    np.random.seed(seed)
    cp.random.seed(seed)
    pos_gpu, rot_gpu = sampler.sample(n_waypoints, side=Side.OUTSIDE)
    insp_pos = cp.asnumpy(pos_gpu).astype(np.float32)
    K = N_ROBOTS
    robot_xyzs = compute_start_grid(K, bmin, bmax)
    home_pos = np.array(
        [[float(x[0]), float(x[1]), float(x[2])] for x in robot_xyzs], dtype=np.float32
    )
    all_pos = np.vstack([home_pos, insp_pos])
    home_indices = list(range(K))
    dist_matrix = compute_distance_matrix(og, cp.asarray(all_pos))
    return K, home_indices, dist_matrix


def _solve_one(dist_matrix, K, home_indices, label, beta_flag, pair_flag, backend):
    t0 = time.perf_counter()
    try:
        vr = solve_vrp(
            dist_matrix=dist_matrix,
            num_vehicles=K,
            depots=home_indices,
            alpha=VRP_ALPHA,
            backend=backend,
            time_limit=TIME_LIMIT,
            mip_gap=MIP_GAP,
            beta_aware_filter=beta_flag,
            forbidden_pair_cuts=pair_flag,
        )
        solve_time = time.perf_counter() - t0
        per_v = per_vehicle_costs(vr.routes, dist_matrix, home_indices)
        makespan = max(per_v) if per_v else float("nan")
        obj = float(vr.objective_value)
        bound = float(vr.best_bound)
        # cuOpt/HiGHS bound on the blended objective; gap = (obj - bound) / |obj|.
        gap = (obj - bound) / abs(obj) if bound > 0 and obj != 0 else float("nan")
        return {
            "config": label,
            "solve_time": solve_time,
            "objective": obj,
            "makespan": makespan,
            "total_cost": float(vr.total_cost),
            "best_bound": bound,
            "gap": gap,
            "gap_reached": bool(not np.isnan(gap) and gap <= MIP_GAP),
            "status": vr.status,
        }
    except Exception as exc:  # noqa: BLE001 - record failure, keep other configs
        return {
            "config": label,
            "solve_time": time.perf_counter() - t0,
            "objective": float("nan"),
            "makespan": float("nan"),
            "total_cost": float("nan"),
            "best_bound": float("nan"),
            "gap": float("nan"),
            "gap_reached": False,
            "status": f"failed: {exc}",
        }


def _setup_mesh_and_sampler():
    mesh = load_and_transform_mesh(MESH_PATH, MESH_TARGET_LENGTH, MESH_POSE)
    bmin = np.asarray(mesh.bounds[0], dtype=float)
    bmax = np.asarray(mesh.bounds[1], dtype=float)
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    o3d_mesh.compute_vertex_normals()
    cfg = ModelConfig.duke_of_lancaster()
    og, _, _ = build_sampling_occupancy_grid(
        mesh=o3d_mesh, frustum_far=cfg.frustum.far,
        min_clearance=2 * ROBOT_RADIUS, resolution=0.20,
    )
    pts, norms = SurfacePointSampler().sample(o3d_mesh, cfg.num_surface_points, seed=42)
    sampler = WeightedViewpointSampler(
        o3d_mesh, cp.asarray(pts, dtype=cp.float32), cp.asarray(norms, dtype=cp.float32),
        cfg.frustum.far, collision_radius=ROBOT_RADIUS, occupancy_grid=og,
    )
    return og, sampler, bmin, bmax


def _print_summary(results: list[dict]):
    labels = [c[0] for c in CUT_CONFIGS]
    logger.info("\n%s\nE11 SUMMARY - VRP cut ablation (mean over seeds/waypoints)\n%s",
                "=" * 78, "=" * 78)
    logger.info("%-8s %10s %10s %10s %9s %9s", "config", "obj", "gap%",
                "solve_s", "makespan", "gap_hit%")
    for lab in labels:
        rows = [r for r in results if r["config"] == lab and not np.isnan(r["objective"])]
        if not rows:
            logger.info("%-8s   (no successful solves)", lab)
            continue
        obj = np.mean([r["objective"] for r in rows])
        gap = np.nanmean([r["gap"] for r in rows]) * 100
        st = np.mean([r["solve_time"] for r in rows])
        mk = np.nanmean([r["makespan"] for r in rows])
        hit = np.mean([1.0 if r["gap_reached"] else 0.0 for r in rows]) * 100
        logger.info("%-8s %10.4f %9.2f %10.1f %9.1f %8.0f", lab, obj, gap, st, mk, hit)


def main():
    p = argparse.ArgumentParser(description="E11: VRP cut ablation")
    p.add_argument("--waypoints", type=int, nargs="+", default=[30, 50])
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_5)
    p.add_argument("--backend", choices=["cuopt", "highs"], default="cuopt")
    p.add_argument("--output_dir", default=os.path.join(RESULTS_DIR, "e11_vrp_cuts_ablation"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--plots_only", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)-8s %(name)s: %(message)s")

    raw_dir = os.path.join(args.output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    all_results: list[dict] = []

    if args.plots_only:
        for f in sorted(os.listdir(raw_dir)):
            if f.endswith(".json"):
                all_results.append(load_run_result(os.path.join(raw_dir, f[:-5])))
        _print_summary(all_results)
        return

    if not _RUNTIME_AVAILABLE:
        raise SystemExit(f"Runtime imports unavailable ({_RUNTIME_IMPORT_ERROR}).")

    backend = VRPBackend.CUOPT if args.backend == "cuopt" else VRPBackend.HIGHS
    og, sampler, bmin, bmax = _setup_mesh_and_sampler()

    for nwp in args.waypoints:
        for seed in args.seeds:
            # All four cut configs solve the identical (seeded) distance matrix,
            # so build the instance once and reuse it across configs.
            instance = None
            for label, beta_flag, pair_flag in CUT_CONFIGS:
                stem = f"wp={nwp}_seed={seed}_cfg={label}"
                rpath = os.path.join(raw_dir, stem)
                if args.resume and os.path.exists(rpath + ".json"):
                    all_results.append(load_run_result(rpath))
                    logger.info("[resume] skip %s", stem)
                    continue
                logger.info("Running wp=%d seed=%d cfg=%s", nwp, seed, label)
                try:
                    if instance is None:
                        instance = _build_instance(nwp, seed, og, sampler, bmin, bmax)
                    K, home_indices, dist_matrix = instance
                    row = _solve_one(dist_matrix, K, home_indices, label,
                                     beta_flag, pair_flag, backend)
                except Exception as exc:  # noqa: BLE001
                    handle_row_exception(exc, stem, resume=args.resume)
                    row = {"config": label, "status": f"crashed: {exc}",
                           "objective": float("nan"), "gap": float("nan"),
                           "makespan": float("nan"), "solve_time": 0.0,
                           "best_bound": float("nan"), "total_cost": float("nan"),
                           "gap_reached": False}
                row.update({"n_waypoints": nwp, "seed": seed})
                all_results.append(row)
                save_run_result(row, rpath)
                logger.info("  obj=%.4f gap=%.3f solve=%.1fs status=%s",
                            row["objective"], row["gap"], row["solve_time"], row["status"])
            free_gpu_memory()

    _print_summary(all_results)


if __name__ == "__main__":
    main()
