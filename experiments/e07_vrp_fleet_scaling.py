#!/usr/bin/env python3
"""E07: VRP Fleet Scaling - sweep fleet sizes (<=10) x waypoint counts x seeds with LBs.

    python -m experiments.e07_vrp_fleet_scaling
    python -m experiments.e07_vrp_fleet_scaling --fleet_sizes 1 2 3 --waypoint_counts 10 20
"""

from __future__ import annotations

import argparse
import csv
import itertools
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.common.config import (
    E07_FLEET_SIZES,
    E07_WAYPOINT_COUNTS,
    RESULTS_DIR,
    SEEDS_5,
    ModelConfig,
)
from experiments.common.lb_sidecar import (
    append_lb_csv_row,
    compute_all_lbs,
    lb_csv_fieldnames,
    load_lb_csv,
    recompute_lbs,
    write_lb_csv,
)
from experiments.common.plotting import (
    CATEGORICAL_COLORS,
    THESIS_COL,
    save_figure,
    setup_thesis_style,
    stacked_bar,
)

_RUNTIME_IMPORT_ERROR: ImportError | None = None
try:
    import cupy as cp
    import open3d as o3d

    from experiments.common.runner import free_gpu_memory
    from shared.mesh_loader import load_and_transform_mesh
    from shared.surface_sampler import SurfacePointSampler
    from shared.types import Side
    from visibility.sampling import WeightedViewpointSampler
    from VRP.core.constants import (
        AUV_CRUISE_SPEED,
        MESH_PATH,
        MESH_POSE,
        MESH_TARGET_LENGTH,
        ROBOT_RADIUS,
        SPACE_TIME_DWELL_S,
    )
    from VRP.core.distance_matrix import compute_distance_matrix
    from VRP.core.geometry import compute_start_grid
    from VRP.core.types import ExecutionResult, VRPBackend, VRPResult
    from VRP.mapf.mapf_planner import MultiAgentPathPlanner
    from VRP.utils.collision import find_trajectory_collisions
    from VRP.vrp.vrp_solver import solve_vrp

    _RUNTIME_AVAILABLE = True
except ImportError as _e:
    cp = None
    o3d = None
    free_gpu_memory = None
    load_and_transform_mesh = None
    SurfacePointSampler = None
    Side = None
    WeightedViewpointSampler = None
    compute_distance_matrix = None
    solve_vrp = None
    VRPBackend = VRPResult = ExecutionResult = None
    MultiAgentPathPlanner = None
    compute_start_grid = None
    find_trajectory_collisions = None
    MESH_PATH = MESH_POSE = MESH_TARGET_LENGTH = None
    ROBOT_RADIUS = AUV_CRUISE_SPEED = SPACE_TIME_DWELL_S = None
    _RUNTIME_AVAILABLE = False
    _RUNTIME_IMPORT_ERROR = _e

# Module-scope so LB-only mode reuses the same value.
_VRP_ALPHA = 0.5
_VRP_TIME_LIMIT = 120

logger = logging.getLogger(__name__)


@dataclass
class RunMetrics:
    fleet_size: int = 0
    n_waypoints: int = 0
    seed: int = 0
    status: str = ""
    total_cost: float = 0.0
    makespan: float = 0.0
    shortest_route: float = 0.0
    route_balance_ratio: float = 0.0
    route_cost_std: float = 0.0
    mean_route_cost: float = 0.0
    total_traj_steps: int = 0
    residual_collisions: int = 0
    fail_count_total: int = 0
    t_dist_matrix: float = 0.0
    t_vrp_solve: float = 0.0
    t_trajectory: float = 0.0
    t_total: float = 0.0


_LB_KEY_COLS = ("fleet_size", "n_waypoints", "seed")


def run_single(fleet_size, n_waypoints, seed, og, mesh_bounds_min, mesh_bounds_max, sampler):
    """Returns ``(m, lb)``; ``lb`` is None when the run failed before the VRP solve."""
    m = RunMetrics(fleet_size=fleet_size, n_waypoints=n_waypoints, seed=seed)
    t_total_start = time.perf_counter()
    lb: dict | None = None

    try:
        cp.random.seed(seed)
        pos_gpu, rot_gpu = sampler.sample(n_waypoints, side=Side.OUTSIDE)
        insp_positions = cp.asnumpy(pos_gpu).astype(np.float32)
        insp_rotmats = cp.asnumpy(rot_gpu).astype(np.float32)
        K = fleet_size
        robot_start_xyzs = compute_start_grid(K, mesh_bounds_min, mesh_bounds_max)
        home_positions = np.array(
            [[float(xyz[0]), float(xyz[1]), float(xyz[2])] for xyz in robot_start_xyzs],
            dtype=np.float32,
        )
        home_rotmats = np.tile(np.eye(3, dtype=np.float32), (K, 1, 1))
        all_positions = np.vstack([home_positions, insp_positions])
        all_rotmats = np.concatenate([home_rotmats, insp_rotmats])
        home_indices = list(range(K))

        t0 = time.perf_counter()
        dist_matrix = compute_distance_matrix(og, cp.asarray(all_positions))
        m.t_dist_matrix = time.perf_counter() - t0

        t0 = time.perf_counter()
        vrp_result: VRPResult = solve_vrp(
            dist_matrix=dist_matrix,
            num_vehicles=K,
            depots=home_indices,
            alpha=_VRP_ALPHA,
            backend=VRPBackend.CUOPT,
            time_limit=_VRP_TIME_LIMIT,
        )
        m.t_vrp_solve = time.perf_counter() - t0
        m.status = vrp_result.status
        m.total_cost = vrp_result.total_cost

        # Run regardless of VRP success so analytical LBs are logged even alongside failed solves.
        try:
            lb = compute_all_lbs(
                dist_matrix,
                home_indices,
                K,
                n_waypoints,
                _VRP_ALPHA,
                include_mapf=True,
                vrp_best_bound_m=vrp_result.best_bound,
                vrp_objective_value_m=vrp_result.objective_value,
            )
        except Exception as e:
            logger.warning("LB computation failed: %s", e)

        if not any(vrp_result.routes):
            m.status = "empty_routes"
            m.t_total = time.perf_counter() - t_total_start
            return m, lb

        routes = [
            [home_indices[i]] + list(r) + [home_indices[i]] for i, r in enumerate(vrp_result.routes)
        ]

        from VRP.vrp._helpers import per_vehicle_costs

        rc = np.array(per_vehicle_costs(vrp_result.routes, dist_matrix, home_indices))
        m.makespan = float(rc.max())
        m.shortest_route = float(rc[rc > 0].min()) if np.any(rc > 0) else 0.0
        m.mean_route_cost = float(rc.mean())
        m.route_cost_std = float(rc.std())
        m.route_balance_ratio = (
            m.makespan / m.shortest_route if m.shortest_route > 0 else float("inf")
        )

        t0 = time.perf_counter()
        start_positions = [np.array(xyz, dtype=np.float32) for xyz in robot_start_xyzs]
        wp_pos_gpu = cp.asarray(all_positions, dtype=cp.float32)
        wp_rot_gpu = cp.asarray(all_rotmats, dtype=cp.float32)

        executor = MultiAgentPathPlanner(start_positions=start_positions, og=og)
        exec_result: ExecutionResult = executor.execute(
            routes=routes,
            waypoint_positions=wp_pos_gpu,
            waypoint_rotmats=wp_rot_gpu,
            home_indices=set(home_indices),
            dist_matrix=dist_matrix,
            alpha=0.5,
        )
        m.t_trajectory = time.perf_counter() - t0

        m.total_traj_steps = (
            max(len(t) for t in exec_result.all_traj_positions)
            if exec_result.all_traj_positions
            else 0
        )
        m.fail_count_total = sum(exec_result.fail_counts)
        m.residual_collisions = len(find_trajectory_collisions(exec_result.all_traj_positions))

    except Exception as e:
        logger.error("Run failed: %s", e, exc_info=True)
        m.status = f"error: {e}"

    m.t_total = time.perf_counter() - t_total_start
    return m, lb


def generate_plots(
    all_metrics, fleet_sizes, waypoint_counts, output_dir, lb_by_key: dict | None = None
):
    setup_thesis_style()
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    lb_by_key = lb_by_key or {}

    successful = [m for m in all_metrics if m.status == "success"]
    if not successful:
        logger.warning("No successful runs for plotting")
        return

    from collections import defaultdict

    groups = defaultdict(list)
    for r in successful:
        groups[(r.fleet_size, r.n_waypoints)].append(r)

    # Distinct categorical colors per waypoint-count line so bands and lines are easy to distinguish.
    wp_colors = [
        CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)] for i in range(len(waypoint_counts))
    ]

    def _lb_band(nw, lb_field):
        """Per-seed min/max band across fleet sizes for the given LB field. Returns (xs, lb_min,
        lb_max), only fleet sizes with at least one positive LB."""
        xs, lb_min, lb_max = [], [], []
        for k in fleet_sizes:
            vals = []
            for r in groups.get((k, nw), []):
                lb = lb_by_key.get((r.fleet_size, r.n_waypoints, r.seed), {})
                v = float(lb.get(lb_field, 0.0))
                if v > 0.0:
                    vals.append(v)
            if vals:
                xs.append(k)
                lb_min.append(float(np.min(vals)))
                lb_max.append(float(np.max(vals)))
        return xs, lb_min, lb_max

    def _objective_for(r) -> float:
        return _VRP_ALPHA * r.makespan + (1.0 - _VRP_ALPHA) * r.total_cost

    fig, ax = plt.subplots(figsize=(THESIS_COL, 3))
    for wi, nw in enumerate(waypoint_counts):
        xs, means, stds = [], [], []
        for k in fleet_sizes:
            vals = [r.makespan for r in groups.get((k, nw), [])]
            if vals:
                xs.append(k)
                means.append(np.mean(vals))
                stds.append(np.std(vals))
        if xs:
            ax.errorbar(
                xs, means, yerr=stds, marker="o", color=wp_colors[wi], label=f"{nw} wps", capsize=2
            )
        lb_xs, lb_min, lb_max = _lb_band(nw, "vrp_makespan_lb_m")
        if lb_xs:
            ax.fill_between(lb_xs, lb_min, lb_max, alpha=0.12, color=wp_colors[wi], linewidth=0)
    ax.set_xlabel("Fleet size")
    ax.set_ylabel("Makespan (m)")
    ax.set_title("Makespan vs. Fleet Size (shaded: analytical LB band, per-seed)")
    ax.legend(fontsize=6, ncol=2)
    save_figure(fig, os.path.join(fig_dir, "e07_makespan_vs_fleet"))

    fig, ax = plt.subplots(figsize=(THESIS_COL, 3))
    for wi, nw in enumerate(waypoint_counts):
        xs, means, stds = [], [], []
        for k in fleet_sizes:
            vals = [r.total_cost for r in groups.get((k, nw), [])]
            if vals:
                xs.append(k)
                means.append(np.mean(vals))
                stds.append(np.std(vals))
        if xs:
            ax.errorbar(
                xs, means, yerr=stds, marker="o", color=wp_colors[wi], label=f"{nw} wps", capsize=2
            )
        lb_xs, lb_min, lb_max = _lb_band(nw, "vrp_total_cost_lb_m")
        if lb_xs:
            ax.fill_between(lb_xs, lb_min, lb_max, alpha=0.12, color=wp_colors[wi], linewidth=0)
    ax.set_xlabel("Fleet size")
    ax.set_ylabel("Total route cost (m)")
    ax.set_title("Total Cost vs. Fleet Size (shaded: analytical LB band, per-seed)")
    ax.legend(fontsize=6, ncol=2)
    save_figure(fig, os.path.join(fig_dir, "e07_total_cost_vs_fleet"))

    fig, ax = plt.subplots(figsize=(THESIS_COL, 3))
    any_obj = False
    for wi, nw in enumerate(waypoint_counts):
        xs, means, stds = [], [], []
        for k in fleet_sizes:
            vals = [_objective_for(r) for r in groups.get((k, nw), [])]
            if vals:
                xs.append(k)
                means.append(np.mean(vals))
                stds.append(np.std(vals))
        if xs:
            any_obj = True
            ax.errorbar(
                xs, means, yerr=stds, marker="o", color=wp_colors[wi], label=f"{nw} wps", capsize=2
            )
        # Prefer cuOpt bound; fall back to analytical objective LB.
        lb_xs, lb_min, lb_max = _lb_band(nw, "vrp_objective_best_bound_m")
        if not lb_xs:
            lb_xs, lb_min, lb_max = _lb_band(nw, "vrp_objective_lb_m")
        if lb_xs:
            ax.fill_between(lb_xs, lb_min, lb_max, alpha=0.15, color=wp_colors[wi], linewidth=0)
    if any_obj:
        ax.set_xlabel("Fleet size")
        ax.set_ylabel(f"VRP objective (β={_VRP_ALPHA}) (m)")
        ax.set_title("VRP Objective vs. Fleet Size (shaded: cuOpt/analytical LB band)")
        ax.legend(fontsize=6, ncol=2)
        save_figure(fig, os.path.join(fig_dir, "e07_objective_vs_fleet"))

    mid_wps = waypoint_counts[len(waypoint_counts) // 2]
    fig, ax = plt.subplots(figsize=(THESIS_COL, 3))
    stage_data = {"Dist matrix": [], "VRP solve": [], "Trajectory": []}
    used_fleets = []
    for k in fleet_sizes:
        ok = groups.get((k, mid_wps), [])
        if ok:
            used_fleets.append(str(k))
            stage_data["Dist matrix"].append(np.mean([r.t_dist_matrix for r in ok]))
            stage_data["VRP solve"].append(np.mean([r.t_vrp_solve for r in ok]))
            stage_data["Trajectory"].append(np.mean([r.t_trajectory for r in ok]))
    if used_fleets:
        stacked_bar(
            ax, used_fleets, stage_data, ylabel="Time (s)", title=f"Timing ({mid_wps} waypoints)"
        )
        save_figure(fig, os.path.join(fig_dir, "e07_timing_breakdown"))

    logger.info("E07 figures saved to %s", fig_dir)


def _build_setup(resolution: float = 0.20):
    """Load Duke mesh + build OG + sampler once. Shared between full-run and ``--compute_lbs_only``."""
    logger.info("Loading mesh and building occupancy grid (res=%.2f) ...", resolution)
    mesh = load_and_transform_mesh(MESH_PATH, MESH_TARGET_LENGTH, MESH_POSE)
    mesh_bounds_min = np.asarray(mesh.bounds[0], dtype=float)
    mesh_bounds_max = np.asarray(mesh.bounds[1], dtype=float)

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    o3d_mesh.compute_vertex_normals()

    _model_cfg = ModelConfig.duke_of_lancaster()
    from visibility.sampling.utils.sampling_grid_builder import build_sampling_occupancy_grid

    og, _, _ = build_sampling_occupancy_grid(
        mesh=o3d_mesh,
        frustum_far=_model_cfg.frustum.far,
        min_clearance=2 * ROBOT_RADIUS,
        resolution=resolution,
    )
    logger.info("  Grid: %s res=%.2f", og.grid.shape, og.resolution)

    _pts_np, _norms_np = SurfacePointSampler().sample(
        o3d_mesh, _model_cfg.num_surface_points, seed=42
    )
    sampler = WeightedViewpointSampler(
        o3d_mesh,
        cp.asarray(_pts_np, dtype=cp.float32),
        cp.asarray(_norms_np, dtype=cp.float32),
        _model_cfg.frustum.far,
        collision_radius=ROBOT_RADIUS,
        occupancy_grid=og,
    )
    return og, sampler, mesh_bounds_min, mesh_bounds_max


def _recompute_lbs_for_row(
    m: RunMetrics, og, sampler, bmin, bmax, *, include_cuopt: bool = False
) -> dict:
    """Re-generate the instance deterministically (sampler is deterministic given a CuPy RNG seed
    - same as ``run_single``). ``include_cuopt`` runs a short cuOpt solve to extract a dual bound;
    expensive, intended only for augmenting a few rows of interest."""
    cp.random.seed(m.seed)
    pos_gpu, _ = sampler.sample(m.n_waypoints, side=Side.OUTSIDE)
    insp_positions = cp.asnumpy(pos_gpu).astype(np.float32)
    K = m.fleet_size
    robot_start_xyzs = compute_start_grid(K, bmin, bmax)
    home_positions = np.array(
        [[float(xyz[0]), float(xyz[1]), float(xyz[2])] for xyz in robot_start_xyzs],
        dtype=np.float32,
    )
    all_positions = np.vstack([home_positions, insp_positions])
    home_indices = list(range(K))
    dist_matrix = compute_distance_matrix(og, cp.asarray(all_positions))

    return recompute_lbs(
        dist_matrix,
        home_indices,
        K,
        m.n_waypoints,
        _VRP_ALPHA,
        include_mapf=True,
        include_cuopt=include_cuopt,
    )


def main():
    p = argparse.ArgumentParser(description="E07: VRP Fleet Scaling")
    p.add_argument("--fleet_sizes", type=int, nargs="+", default=E07_FLEET_SIZES)
    p.add_argument("--waypoint_counts", type=int, nargs="+", default=E07_WAYPOINT_COUNTS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_5)
    p.add_argument("--output_dir", default=os.path.join(RESULTS_DIR, "e07_vrp_fleet_scaling"))
    p.add_argument("--plots_only", action="store_true")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing results.csv, skipping completed runs",
    )
    p.add_argument(
        "--compute_lbs_only",
        action="store_true",
        help="Skip full pipeline; recompute LBs for each unique (fleet_size, n_waypoints, seed) "
        "from results.csv into a sidecar lower_bounds.csv.",
    )
    p.add_argument(
        "--include_cuopt_bound",
        action="store_true",
        help="In --compute_lbs_only mode, also run a short cuOpt solve per instance to extract "
        "the MIP dual bound. Expensive; off by default.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    if not args.plots_only and not _RUNTIME_AVAILABLE:
        raise SystemExit(
            f"Runtime imports unavailable ({_RUNTIME_IMPORT_ERROR}). "
            "Activate the isaaclab env or pass --plots_only."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    all_metrics = []

    csv_path = os.path.join(args.output_dir, "results.csv")
    lb_csv_path = os.path.join(args.output_dir, "lower_bounds.csv")
    fieldnames = ["run_id"] + list(RunMetrics.__dataclass_fields__.keys())

    def _load_existing_csv() -> list[RunMetrics]:
        out: list[RunMetrics] = []
        if not os.path.exists(csv_path):
            return out
        import csv as csv_mod

        with open(csv_path) as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                m = RunMetrics()
                for fk, v in row.items():
                    if fk == "run_id":
                        continue
                    if hasattr(m, fk):
                        field_type = type(getattr(m, fk))
                        try:
                            setattr(m, fk, field_type(v))
                        except (ValueError, TypeError):
                            setattr(m, fk, v)
                out.append(m)
        return out

    if args.compute_lbs_only:
        all_metrics = _load_existing_csv()
        if not all_metrics:
            logger.error("No existing results.csv at %s to augment with LBs.", csv_path)
            return
        og, sampler, bmin, bmax = _build_setup()
        # Dedupe per-instance: main CSV may have many rows per tuple if re-run in overwrite mode.
        unique_keys = {}
        for m in all_metrics:
            unique_keys.setdefault((m.fleet_size, m.n_waypoints, m.seed), m)
        logger.info(
            "Recomputing LBs for %d unique instances (from %d rows)%s ...",
            len(unique_keys),
            len(all_metrics),
            " including cuOpt bound" if args.include_cuopt_bound else "",
        )

        lb_rows = []
        for idx, ((fs, nw, seed), m) in enumerate(unique_keys.items(), 1):
            try:
                lb = _recompute_lbs_for_row(
                    m,
                    og,
                    sampler,
                    bmin,
                    bmax,
                    include_cuopt=args.include_cuopt_bound,
                )
                lb_rows.append(
                    {
                        "fleet_size": fs,
                        "n_waypoints": nw,
                        "seed": seed,
                        **lb,
                    }
                )
            except Exception as e:
                logger.error("LB recompute failed (fleet=%d wps=%d seed=%d): %s", fs, nw, seed, e)
            finally:
                free_gpu_memory()
            if idx % 20 == 0:
                logger.info("  ... %d/%d instances processed", idx, len(unique_keys))

        write_lb_csv(lb_csv_path, _LB_KEY_COLS, lb_rows, include_mapf=True)
        logger.info("Wrote %d LB rows to %s", len(lb_rows), lb_csv_path)

        lb_by_key = load_lb_csv(lb_csv_path, _LB_KEY_COLS)
        generate_plots(
            all_metrics,
            args.fleet_sizes,
            args.waypoint_counts,
            args.output_dir,
            lb_by_key=lb_by_key,
        )
        return

    if not args.plots_only:
        # 0.20m resolution (2× coarser than default) avoids CUDA OOM during distance-matrix
        # construction on the Duke model.
        og, sampler, mesh_bounds_min, mesh_bounds_max = _build_setup()

        configs = list(itertools.product(args.fleet_sizes, args.waypoint_counts, args.seeds))
        total = len(configs)

        completed = set()
        next_run_id = 1
        if args.resume and os.path.exists(csv_path):
            existing = _load_existing_csv()
            all_metrics.extend(existing)
            import csv as csv_mod

            with open(csv_path) as f:
                for row in csv_mod.DictReader(f):
                    completed.add(
                        (int(row["fleet_size"]), int(row["n_waypoints"]), int(row["seed"]))
                    )
                    next_run_id = max(next_run_id, int(row["run_id"]) + 1)
            logger.info("Resuming: skipping %d completed runs", len(completed))

        file_mode = "a" if args.resume and completed else "w"
        with open(csv_path, file_mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if file_mode == "w":
                writer.writeheader()

            run_id = next_run_id
            for k, nw, seed in configs:
                if (k, nw, seed) in completed:
                    continue
                logger.info(
                    "=== Run %d/%d: fleet=%d wps=%d seed=%d ===", run_id, total, k, nw, seed
                )
                m, lb = run_single(k, nw, seed, og, mesh_bounds_min, mesh_bounds_max, sampler)
                all_metrics.append(m)
                row = asdict(m)
                row["run_id"] = run_id
                writer.writerow(row)
                f.flush()
                if lb is not None:
                    append_lb_csv_row(
                        lb_csv_path,
                        _LB_KEY_COLS,
                        {"fleet_size": k, "n_waypoints": nw, "seed": seed},
                        lb,
                        include_mapf=True,
                    )
                logger.info(
                    "  status=%s makespan=%.1f t_total=%.1fs", m.status, m.makespan, m.t_total
                )
                run_id += 1
                free_gpu_memory()
    else:
        all_metrics.extend(_load_existing_csv())

    lb_by_key = load_lb_csv(lb_csv_path, _LB_KEY_COLS)
    if all_metrics:
        generate_plots(
            all_metrics,
            args.fleet_sizes,
            args.waypoint_counts,
            args.output_dir,
            lb_by_key=lb_by_key,
        )


if __name__ == "__main__":
    main()
