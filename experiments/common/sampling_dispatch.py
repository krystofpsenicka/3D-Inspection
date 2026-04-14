"""Strategy dispatch for sampling experiments.

Maps a strategy name to sampled candidate viewpoints.  Used by e01, e02, e13.

Strategy names:
  weighted            — SDF² uniform, all N from base sampler
  weighted_curvature  — SDF² + curvature bias, all N from base sampler
  targeted_X          — X% targeted toward uncovered, (100-X)% weighted base
  cmaes_X             — X% CMA-ES optimised, (100-X)% weighted base
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
    k_coverage: int = 1,
    samples_per_iteration: int = 1,
    travel_weight: float | None = None,
) -> tuple:
    """Sample candidates for the given strategy.

    Args:
        ctx:                   PipelineContext (provides sampler and OG).
        strategy:              e.g. "weighted", "targeted_50", "cmaes_25".
        num_candidates:        total candidates requested.
        target_points:         CuPy (M, 3) surface points.
        normals:               CuPy (M, 3) surface normals.
        vis_query:             visibility query object.
        model:                 ModelConfig (provides frustum / collision_radius).
        k_coverage:            minimum per-point coverage count for targeted/cmaes phase.
        samples_per_iteration: batch size for targeted iterative mode (default 1).
        travel_weight:         CMA-ES objective travel penalty weight; None uses sampler default.

    Returns:
        (pos_gpu, rot_gpu, n_base, n_iterative, base_sampler_name)
          pos_gpu / rot_gpu  — CuPy arrays of actual candidates generated.
          n_base             — candidates from the base (weighted) sampler.
          n_iterative        — candidates from the targeted/CMA-ES phase.
          base_sampler_name  — "weighted" for hybrid strategies, else strategy.
    """
    og = ctx.build_sampling_og()
    sampler = ctx.build_sampler("targeted")  # weighted & targeted share the same object

    # ── One-shot strategies ───────────────────────────────────────────────
    if strategy in ("weighted", "weighted_curvature"):
        curvature = (strategy == "weighted_curvature")
        pos_gpu, rot_gpu = sampler.sample(
            cp.arange(len(target_points)), num_candidates,
            side=Side.OUTSIDE, curvature_weighting=curvature,
        )
        return pos_gpu, rot_gpu, num_candidates, 0, strategy

    # ── Hybrid strategies: parse percentage ──────────────────────────────
    if strategy.startswith("targeted_"):
        pct = int(strategy.split("_")[1])
    elif strategy.startswith("cmaes_"):
        pct = int(strategy.split("_")[1])
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    n_iterative = int(num_candidates * pct / 100)
    n_base = num_candidates - n_iterative

    # ── Base (weighted) phase ─────────────────────────────────────────────
    if n_base > 0:
        pos_gpu, rot_gpu = sampler.sample(
            cp.arange(len(target_points)), n_base,
            side=Side.OUTSIDE, curvature_weighting=False,
        )
        V_init, _ = vis_query.compute_visibility_batch(pos_gpu, rot_gpu)
        coverage_count = V_init.astype(cp.int32).sum(axis=0)
        uncovered = cp.where(coverage_count < k_coverage)[0]
    else:
        pos_gpu = cp.empty((0, 3), dtype=cp.float32)
        rot_gpu = cp.empty((0, 3, 3), dtype=cp.float32)
        uncovered = cp.arange(len(target_points))
        coverage_count = cp.zeros(len(target_points), dtype=cp.int32)

    # ── Iterative / optimising phase ──────────────────────────────────────
    if n_iterative > 0 and len(uncovered) > 0:
        if strategy.startswith("targeted_"):
            t_pos, t_rot = sampler.sample(
                uncovered, n_iterative,
                side=Side.OUTSIDE, curvature_weighting=False,
                k_coverage=k_coverage,
                coverage_count_gpu=coverage_count,
                visibility_query=vis_query,
                samples_per_iteration=samples_per_iteration,
            )
            if len(t_pos) > 0:
                pos_gpu = cp.concatenate([pos_gpu, t_pos]) if len(pos_gpu) > 0 else t_pos
                rot_gpu = cp.concatenate([rot_gpu, t_rot]) if len(rot_gpu) > 0 else t_rot

        elif strategy.startswith("cmaes_"):
            from visibility.sampling import OptimizingSampler, CMAESBackend
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
            opt_pos, opt_rot = opt_sampler.sample_optimized(
                n_iterative, coverage_count, vis_query,
                existing_pos_gpu=pos_gpu if len(pos_gpu) > 0 else None,
                existing_rot_gpu=rot_gpu if len(rot_gpu) > 0 else None,
                k_coverage=k_coverage,
                **tw_kw,
            )
            if len(opt_pos) > 0:
                pos_gpu = cp.concatenate([pos_gpu, opt_pos]) if len(pos_gpu) > 0 else opt_pos
                rot_gpu = cp.concatenate([rot_gpu, opt_rot]) if len(rot_gpu) > 0 else opt_rot

    return pos_gpu, rot_gpu, n_base, n_iterative, "weighted"
