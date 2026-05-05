"""Strategy dispatch for sampling experiments.

Strategies:
  weighted             -- SDF^2 uniform, all N from base sampler
  weighted_curvature   -- SDF^2 + curvature bias
  targeted             -- targeted toward uncovered (iterative refinement)
  cmaes                -- CMA-ES optimised
"""

from __future__ import annotations

import cupy as cp

from shared.types import Side


def sample_strategy(
    ctx,
    strategy: str,
    num_candidates: int,
    target_points: cp.ndarray,
    normals: cp.ndarray,
    vis_query,
    model,
    k_coverage: int = 4,
    samples_per_iteration: int = 25,
    travel_weight: float | None = None,
    popsize: int | None = None,
    maxiter: int | None = None,
) -> tuple:
    """Sample candidates for the given strategy.

    Returns (pos_gpu, rot_gpu, n_base, n_iterative, base_sampler_name,
             n_warmstart_fallbacks).
    """
    og = ctx.build_sampling_og()
    sampler = ctx.build_sampler("targeted")  # weighted & targeted share the same object

    # One-shot strategies
    if strategy in ("weighted", "weighted_curvature"):
        curvature = strategy == "weighted_curvature"
        pos_gpu, rot_gpu = sampler.sample(
            cp.arange(len(target_points)),
            num_candidates,
            side=Side.OUTSIDE,
            curvature_weighting=curvature,
        )
        return pos_gpu, rot_gpu, num_candidates, 0, strategy, 0

    if strategy not in ("targeted", "cmaes"):
        raise ValueError(f"Unknown strategy: {strategy!r}")

    n_iterative = num_candidates
    n_base = 0
    n_warmstart_fallbacks = 0

    pos_gpu = cp.empty((0, 3), dtype=cp.float32)
    rot_gpu = cp.empty((0, 3, 3), dtype=cp.float32)
    uncovered = cp.arange(len(target_points))
    coverage_count = cp.zeros(len(target_points), dtype=cp.int32)

    # Iterative / optimising phase
    if len(uncovered) > 0:
        if strategy == "targeted":
            t_pos, t_rot = sampler.sample(
                uncovered,
                n_iterative,
                side=Side.OUTSIDE,
                curvature_weighting=False,
                k_coverage=k_coverage,
                coverage_count_gpu=coverage_count,
                visibility_query=vis_query,
                samples_per_iteration=samples_per_iteration,
            )
            if len(t_pos) > 0:
                pos_gpu = cp.concatenate([pos_gpu, t_pos]) if len(pos_gpu) > 0 else t_pos
                rot_gpu = cp.concatenate([rot_gpu, t_rot]) if len(rot_gpu) > 0 else t_rot

        elif strategy == "cmaes":
            from visibility.sampling import CMAESBackend, OptimizingSampler

            opt_sampler = OptimizingSampler(
                mesh=ctx._o3d_mesh,
                target_points=target_points,
                normals=normals,
                frustum_far=model.frustum.far,
                collision_radius=model.collision_radius,
                occupancy_grid=og,
                backend=CMAESBackend(),
                random_sampler=sampler,
            )
            tw_kw = {} if travel_weight is None else {"travel_weight": travel_weight}
            cmaes_kw = {}
            if popsize is not None:
                cmaes_kw["popsize"] = popsize
            if maxiter is not None:
                cmaes_kw["maxiter"] = maxiter
            opt_pos, opt_rot, n_warmstart_fallbacks = opt_sampler.sample_optimized(
                n_iterative,
                coverage_count,
                vis_query,
                existing_pos_gpu=pos_gpu if len(pos_gpu) > 0 else None,
                existing_rot_gpu=rot_gpu if len(rot_gpu) > 0 else None,
                k_coverage=k_coverage,
                **tw_kw,
                **cmaes_kw,
            )
            if len(opt_pos) > 0:
                pos_gpu = cp.concatenate([pos_gpu, opt_pos]) if len(pos_gpu) > 0 else opt_pos
                rot_gpu = cp.concatenate([rot_gpu, opt_rot]) if len(rot_gpu) > 0 else opt_rot

    return pos_gpu, rot_gpu, n_base, n_iterative, "weighted", n_warmstart_fallbacks
