#!/usr/bin/env python3
"""Shared "pipeline to routes" helper for the camera-ready ablation experiments.

Runs the full inspection pipeline (mesh -> surface -> OG -> sampling -> visibility
-> set cover -> distance matrix -> VRP) once for a given model/seed and returns
all intermediate artifacts needed by the MAPF-focused ablations (E12, E13) and by
the scaling study (E14). This mirrors the body of ``e10_cross_model.run_single``
up to (but not including) the MAPF stage, so the two stay behaviourally
consistent; E10 itself is left untouched.

The helper is import-guarded exactly like the experiment scripts: importing it on
a CPU-only box (no cupy / VRP stack) raises a clear error only when actually
called, so ``--plots_only`` paths in the callers keep working.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PipelineArtifacts:
    """Everything the MAPF / scaling experiments need after routing."""

    og: object
    target_points: object          # (P, 3) cupy/numpy
    normals: object                # (P, 3)
    vis_query: object              # visibility query (raycast) for coverage rechecks
    num_points: int
    num_candidates: int
    # Set-cover selection
    num_viewpoints: int = 0
    coverage: float = 0.0
    selected_indices: np.ndarray | None = None
    insp_positions: np.ndarray | None = None
    insp_rotmats: np.ndarray | None = None
    # VRP layout
    home_positions: np.ndarray | None = None
    home_rotmats: np.ndarray | None = None
    all_positions: np.ndarray | None = None
    all_rotmats: np.ndarray | None = None
    home_indices: list[int] = field(default_factory=list)
    robot_starts: object = None
    dist_matrix: object = None     # cupy (N, N)
    routes: list[list[int]] | None = None   # customer routes (depots stripped)
    vrp_status: str = "skipped"
    vrp_makespan: float = float("nan")
    vrp_total_cost: float = float("nan")
    # Per-stage timings (seconds)
    timings: dict = field(default_factory=dict)


def run_pipeline_to_routes(
    ctx,
    model_cfg,
    seed: int,
    *,
    fleet_size: int = 5,
    target_coverage: float = 0.95,
    vrp_alpha: float = 0.5,
    vrp_time_limit: int = 60,
    num_candidates: int | None = None,
    num_points: int | None = None,
    sampler: str = "weighted_curvature",
    do_routing: bool = True,
) -> PipelineArtifacts:
    """Run the pipeline up to and including VRP routing.

    ``ctx`` must be a :class:`PipelineContext` whose ``load_mesh`` /
    ``sample_surface`` / ``build_sampling_og`` have already been called by the
    caller (so the expensive mesh setup is shared across seeds), matching E10.

    Returns a :class:`PipelineArtifacts` with the routing solution and all
    artifacts required to run MAPF and coverage rechecks.
    """
    import cupy as cp

    from experiments.common.runner import set_seed, timed
    from experiments.common.sampling_dispatch import sample_strategy
    from visibility.set_cover import LazyGreedySetCover
    from VRP.core.distance_matrix import compute_distance_matrix
    from VRP.core.geometry import compute_start_grid
    from VRP.core.types import VRPBackend
    from VRP.vrp._helpers import per_vehicle_costs
    from VRP.vrp.vrp_solver import solve_vrp

    n_candidates = num_candidates if num_candidates is not None else model_cfg.num_candidates
    timings: dict = {}

    target_points, normals = ctx.sample_surface(seed=42, num_points=num_points)
    og = ctx.build_sampling_og()

    set_seed(seed)
    vis_query = ctx.build_visibility_query("raycast")
    with timed() as t_sample:
        pos_gpu, rot_gpu, _, _, _, _ = sample_strategy(
            ctx, sampler, n_candidates, target_points, normals, vis_query, model_cfg
        )
    timings["t_sample"] = t_sample.elapsed

    with timed() as t_vis:
        V, _ = vis_query.compute_visibility_batch(pos_gpu, rot_gpu)
    timings["t_vis"] = t_vis.elapsed

    V_np = cp.asnumpy(V)
    pos_np = cp.asnumpy(pos_gpu)
    rot_np = cp.asnumpy(rot_gpu)
    num_points = int(len(target_points))

    with timed() as t_opt:
        optimizer = LazyGreedySetCover(num_points, pos_np, rot_np, V_np)
        opt_result = optimizer.optimize(target_coverage=target_coverage, max_viewpoints=1000)
    timings["t_opt"] = t_opt.elapsed

    art = PipelineArtifacts(
        og=og,
        target_points=target_points,
        normals=normals,
        vis_query=vis_query,
        num_points=num_points,
        num_candidates=n_candidates,
        num_viewpoints=int(opt_result.num_viewpoints),
        coverage=float(opt_result.total_coverage),
        timings=timings,
    )

    sel_idx = opt_result.selected_indices
    if hasattr(sel_idx, "get"):
        sel_idx = sel_idx.get()
    sel_idx = np.asarray(sel_idx)
    art.selected_indices = sel_idx
    art.insp_positions = pos_np[sel_idx]
    art.insp_rotmats = rot_np[sel_idx]

    if opt_result.num_viewpoints == 0 or not do_routing:
        return art

    bounds_min, bounds_max = ctx.mesh_bounds
    robot_starts = compute_start_grid(fleet_size, bounds_min, bounds_max)
    home_positions = np.array(
        [[float(x[0]), float(x[1]), float(x[2])] for x in robot_starts], dtype=np.float32
    )
    home_rotmats = np.tile(np.eye(3, dtype=np.float32), (fleet_size, 1, 1))
    all_positions = np.vstack([home_positions, art.insp_positions])
    all_rotmats = np.concatenate([home_rotmats, art.insp_rotmats])
    home_indices = list(range(fleet_size))

    with timed() as t_vrp:
        dist_matrix = compute_distance_matrix(og, cp.asarray(all_positions))
        vrp_result = solve_vrp(
            dist_matrix=dist_matrix,
            num_vehicles=fleet_size,
            depots=home_indices,
            backend=VRPBackend.CUOPT,
            time_limit=vrp_time_limit,
            alpha=vrp_alpha,
        )
    timings["t_vrp"] = t_vrp.elapsed

    art.robot_starts = robot_starts
    art.home_positions = home_positions
    art.home_rotmats = home_rotmats
    art.all_positions = all_positions
    art.all_rotmats = all_rotmats
    art.home_indices = home_indices
    art.dist_matrix = dist_matrix
    art.routes = vrp_result.routes
    art.vrp_status = vrp_result.status
    art.vrp_total_cost = float(vrp_result.total_cost)
    if any(vrp_result.routes):
        rc = np.array(per_vehicle_costs(vrp_result.routes, dist_matrix, home_indices))
        art.vrp_makespan = float(rc.max())
    return art
