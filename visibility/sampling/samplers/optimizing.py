"""Optimization-based viewpoint resampling with injectable optimization backends."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import cupy as cp
import numpy as np

from shared.geometry import directions_rolls_to_rotmats, rotmats_to_directions_rolls
from shared.types import Side

from ...core.constants import (
    DEFAULT_K_COVERAGE,
    DEFAULT_MAX_DISTANCE_OFFSET,
    OPT_SAMPLER_MAXITER,
    OPT_SAMPLER_POPSIZE,
    OPT_SAMPLER_TRAVEL_ROT_FRACTION,
    OPT_SAMPLER_TRAVEL_WEIGHT,
)
from ...visibility.base_cuda import VisibilityQueryCuda
from .base import ProbabilisticSampler, ViewpointSamplerBase

logger = logging.getLogger(__name__)


class OptimizationBackend(ABC):
    """Abstract backend for black-box optimisation in [0,1]^D."""

    @abstractmethod
    def optimize(
        self,
        objective_fn: callable,
        n_dims: int,
        popsize: int,
        maxiter: int,
        verbose: bool = False,
        center_init: cp.ndarray | None = None,
    ) -> cp.ndarray:
        """Search [0,1]^n_dims.

        Args:
            objective_fn: ``(pop, D) CuPy -> (pop,) CuPy`` scores to minimise.
            n_dims:       search space dimensions.
            popsize:      population size per generation.
            maxiter:      maximum generations.
            verbose:      log per-generation progress.
            center_init:  optional ``(D,)`` CuPy array  --  initial centre of the
                          search distribution in [0,1]^D.

        Returns:
            ``(D,)`` CuPy array  --  best solution found in [0,1]^D.
        """


class OptimizingSampler(ViewpointSamplerBase):
    """Iterative viewpoint optimisation with injected optimization
    backend.

    Each round uses the backend to search 6-D pose space
    ``(x, y, z, theta, phi, roll)`` for the single best viewpoint
    covering under-covered surface points, then updates coverage.
    """

    def __init__(
        self,
        *args,
        backend: OptimizationBackend,
        random_sampler: ProbabilisticSampler | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.backend = backend
        self.random_sampler = random_sampler

    # ── Collision helper ─────────────────────────────────────────────────

    def _is_free(self, positions_gpu):
        """Check OG collision for (N, 3) world positions on GPU."""
        og = self._occupancy_grid
        ijk = cp.floor((positions_gpu - og.origin) / og.resolution).astype(cp.int32)

        shape = cp.array(og.grid.shape, dtype=cp.int32)
        in_bounds = (
            (ijk[:, 0] >= 0)
            & (ijk[:, 0] < shape[0])
            & (ijk[:, 1] >= 0)
            & (ijk[:, 1] < shape[1])
            & (ijk[:, 2] >= 0)
            & (ijk[:, 2] < shape[2])
        )

        result = cp.zeros(len(positions_gpu), dtype=cp.bool_)
        if cp.any(in_bounds):
            valid_ijk = ijk[in_bounds]
            result[in_bounds] = ~og.grid[valid_ijk[:, 0], valid_ijk[:, 1], valid_ijk[:, 2]]
        return result

    def sample_optimized(
        self,
        n_rounds: int,
        coverage_count_gpu: cp.ndarray,
        visibility_query: VisibilityQueryCuda,
        existing_pos_gpu: cp.ndarray | None = None,
        existing_rot_gpu: cp.ndarray | None = None,
        side: Side = Side.OUTSIDE,
        target_coverage: float = 1.0,
        k_coverage: int = DEFAULT_K_COVERAGE,
        travel_weight: float = OPT_SAMPLER_TRAVEL_WEIGHT,
        popsize: int = OPT_SAMPLER_POPSIZE,
        maxiter: int = OPT_SAMPLER_MAXITER,
        verbose: bool = False,
    ) -> tuple[cp.ndarray, cp.ndarray, int]:
        """Find optimal viewpoints via iterative optimisation.

        Each round runs the backend in 6-D pose space to find the viewpoint
        maximising coverage of under-covered points while penalising distance
        from existing viewpoints.

        Returns:
            ``(positions_gpu, rotmats_gpu, n_warmstart_fallbacks)``  --  CuPy
            ``(K, 3)`` and ``(K, 3, 3)``, and a count of rounds in which the
            backend returned an infeasible position and we fell back to the
            warm-start centre.
        """
        n_warmstart_fallbacks = 0
        # Feasible bounds
        centers_gpu, _, coarse_res = self.get_feasible_sampling_data(
            side, None, DEFAULT_MAX_DISTANCE_OFFSET, False
        )
        half_res = coarse_res / 2.0
        bounds_lo_gpu = cp.min(centers_gpu, axis=0) - half_res
        bounds_hi_gpu = cp.max(centers_gpu, axis=0) + half_res

        # Apply sphere restriction if active (intersect bounding box)
        if self._sphere_center is not None:
            bounds_lo_gpu = cp.maximum(bounds_lo_gpu, self._sphere_center - self._sphere_radius)
            bounds_hi_gpu = cp.minimum(bounds_hi_gpu, self._sphere_center + self._sphere_radius)

        spatial_diag = float(cp.sqrt(cp.sum((bounds_hi_gpu - bounds_lo_gpu) ** 2)))

        # 6-D bounds: [x, y, z, theta, phi, roll]
        lo_gpu = cp.concatenate(
            [bounds_lo_gpu, cp.array([0.0, 0.0, -cp.pi], dtype=cp.float32)]
        ).astype(cp.float32)
        hi_gpu = cp.concatenate(
            [bounds_hi_gpu, cp.array([cp.pi, 2 * cp.pi, cp.pi], dtype=cp.float32)]
        ).astype(cp.float32)
        ranges_gpu = (hi_gpu - lo_gpu).astype(cp.float32)

        logger.info(
            "[OptimizingSampler] Feasible bounds lo=%s hi=%s (coarse_res=%.3f)",
            cp.asnumpy(lo_gpu[:3]).round(2),
            cp.asnumpy(hi_gpu[:3]).round(2),
            coarse_res,
        )

        # Pre-allocate GPU arrays
        max_total = (0 if existing_pos_gpu is None else len(existing_pos_gpu)) + n_rounds
        all_pos = cp.empty((max_total, 3), dtype=cp.float32)
        all_rot = cp.empty((max_total, 3, 3), dtype=cp.float32)
        n_all = 0

        if existing_pos_gpu is not None and len(existing_pos_gpu) > 0:
            n_ex = len(existing_pos_gpu)
            all_pos[:n_ex] = cp.asarray(existing_pos_gpu, dtype=cp.float32)
            all_rot[:n_ex] = cp.asarray(existing_rot_gpu, dtype=cp.float32)
            n_all = n_ex

        new_pos = cp.empty((n_rounds, 3), dtype=cp.float32)
        new_rot = cp.empty((n_rounds, 3, 3), dtype=cp.float32)
        n_new = 0

        for round_i in range(n_rounds):
            under_k_mask = coverage_count_gpu < k_coverage
            n_under_k = int(under_k_mask.sum())
            frac_k_covered = 1.0 - n_under_k / visibility_query.num_points

            if frac_k_covered >= target_coverage:
                logger.info(
                    "[OptimizingSampler] k=%d coverage %.1f%% >= target -- done after %d rounds.",
                    k_coverage,
                    frac_k_covered * 100,
                    round_i,
                )
                break

            logger.info(
                "[OptimizingSampler %d/%d] k=%d coverage=%.1f%%, "
                "%d under-covered -- running backend...",
                round_i + 1,
                n_rounds,
                k_coverage,
                frac_k_covered * 100,
                n_under_k,
            )

            # Per-round state
            deficit_gpu = cp.maximum(k_coverage - coverage_count_gpu, 0).astype(cp.float32)
            deficit_total = max(float(deficit_gpu.sum()), 1.0)

            avg_pos_gpu = None
            avg_rotmat_gpu = None
            if n_all > 0:
                avg_pos_gpu = all_pos[:n_all].mean(axis=0)
                mean_rot = all_rot[:n_all].mean(axis=0)
                U, _, Vt = cp.linalg.svd(mean_rot)
                avg_rotmat_gpu = U @ Vt

            # Build objective for this round (shared by warm-start and optimizer)
            objective_fn = self._make_objective(
                lo_gpu,
                ranges_gpu,
                deficit_gpu,
                deficit_total,
                visibility_query,
                avg_pos_gpu,
                avg_rotmat_gpu,
                spatial_diag,
                travel_weight,
            )

            # Generate warm-start center_init from random sampler
            center_init_norm = self._generate_warm_start(
                coverage_count_gpu, k_coverage, lo_gpu, ranges_gpu, side, popsize, objective_fn
            )

            best_pos, best_rot, score, used_fallback = self._optimize_one(
                lo_gpu,
                ranges_gpu,
                deficit_gpu,
                visibility_query,
                objective_fn,
                popsize,
                maxiter,
                verbose,
                center_init_norm=center_init_norm,
            )
            if used_fallback:
                n_warmstart_fallbacks += 1

            if score == 0:
                logger.info("[OptimizingSampler] Backend found no useful viewpoint -- stopping.")
                break

            # Update coverage from the newly found viewpoint
            V_single, _ = visibility_query.compute_visibility_batch(
                best_pos[cp.newaxis], best_rot[cp.newaxis]
            )
            visible_mask = V_single[0].astype(cp.bool_)
            coverage_count_gpu[visible_mask] += 1

            new_pos[n_new] = best_pos
            new_rot[n_new] = best_rot
            n_new += 1
            all_pos[n_all] = best_pos
            all_rot[n_all] = best_rot
            n_all += 1

            logger.info(
                "[OptimizingSampler %d/%d] Found VP covering %d under-k pts.",
                round_i + 1,
                n_rounds,
                score,
            )

        if n_new == 0:
            return (
                cp.empty((0, 3), dtype=cp.float32),
                cp.empty((0, 3, 3), dtype=cp.float32),
                n_warmstart_fallbacks,
            )

        return new_pos[:n_new], new_rot[:n_new], n_warmstart_fallbacks

    # ── Warm-start generation ──────────────────────────────────────────────

    def _viewpoints_to_norm6d(self, pos_gpu, rot_gpu, lo_gpu, ranges_gpu):
        """Convert (N,3) positions + (N,3,3) rotmats -> (N,6) in [0,1]^6.

        The 6-D parameterisation is ``[x, y, z, theta, phi, roll]`` where
        theta/phi are spherical coordinates of the viewing direction and
        roll is the camera roll angle.
        """
        dirs, rolls = rotmats_to_directions_rolls(rot_gpu)
        theta = cp.arccos(cp.clip(dirs[:, 2], -1.0, 1.0))
        phi = cp.arctan2(dirs[:, 1], dirs[:, 0]) % (2.0 * cp.pi)

        real = cp.empty((len(pos_gpu), 6), dtype=cp.float32)
        real[:, :3] = pos_gpu
        real[:, 3] = theta
        real[:, 4] = phi
        real[:, 5] = rolls
        return cp.clip((real - lo_gpu) / ranges_gpu, 0.0, 1.0)

    def _generate_warm_start(
        self, coverage_count_gpu, k_coverage, lo_gpu, ranges_gpu, side, popsize, objective_fn
    ):
        """Generate a warm-start ``center_init`` using the random sampler.

        Generates candidates, converts to [0,1]^6, evaluates with the actual
        objective function, and returns the best as ``center_init``.

        Returns a ``(6,)`` CuPy array in [0,1]^6, or None.
        """
        if self.random_sampler is None:
            return None

        n_ws = popsize * 3
        under_k_indices = cp.where(coverage_count_gpu < k_coverage)[0]

        # Generate candidates via the random sampler
        from .targeted import TargetedViewpointSampler

        if isinstance(self.random_sampler, TargetedViewpointSampler):
            ws_pos, ws_rot = self.random_sampler.sample(
                under_k_indices,
                n_ws,
                side=side,
                k_coverage=k_coverage,
                coverage_count_gpu=coverage_count_gpu,
            )
        else:
            ws_pos, ws_rot = self.random_sampler.sample(n_ws, side=side)

        if len(ws_pos) == 0:
            return None

        # Convert to [0,1]^6 and evaluate with the actual objective
        ws_norm = self._viewpoints_to_norm6d(ws_pos, ws_rot, lo_gpu, ranges_gpu)
        scores = objective_fn(ws_norm)

        # Pick the best valid candidate (score < penalty threshold)
        valid_mask = scores < 1e5
        if not cp.any(valid_mask):
            logger.warning("  Warm-start: all %d candidates in occupied space", len(ws_pos))
            return None

        valid_idx = cp.where(valid_mask)[0]
        best_idx = valid_idx[int(cp.argmin(scores[valid_idx]))]
        center = ws_norm[best_idx]
        logger.info(
            "  Warm-start: best score=%.4f from %d/%d valid candidates",
            float(scores[best_idx]),
            int(valid_mask.sum()),
            len(ws_pos),
        )
        return center

    # ── Objective function factory ───────────────────────────────────────

    def _make_objective(
        self,
        lo_gpu,
        ranges_gpu,
        deficit_gpu,
        deficit_total,
        visibility_query,
        avg_pos_gpu,
        avg_rotmat_gpu,
        spatial_diag,
        travel_weight,
    ):
        """Create the objective closure for one optimisation round.

        Returns a callable ``(pop, 6) CuPy -> (pop,) CuPy`` (scores to minimise).
        """
        is_free_fn = self._is_free

        def objective_fn(vals_gpu):
            vals_gpu = cp.clip(vals_gpu, 0.0, 1.0)
            real = vals_gpu * ranges_gpu + lo_gpu

            positions = real[:, :3]
            theta = real[:, 3]
            phi = real[:, 4]
            roll = real[:, 5]

            # Spherical -> direction
            directions = cp.stack(
                [
                    cp.sin(theta) * cp.cos(phi),
                    cp.sin(theta) * cp.sin(phi),
                    cp.cos(theta),
                ],
                axis=1,
            ).astype(cp.float32)

            # Free-space check
            free_mask = is_free_fn(positions)
            free_idx = cp.where(free_mask)[0]

            pop_size = len(vals_gpu)
            scores_gpu = cp.full(pop_size, 1e6, dtype=cp.float32)

            if len(free_idx) == 0:
                return scores_gpu

            free_pos = positions[free_idx]
            free_dirs = directions[free_idx]
            free_rolls = roll[free_idx]

            # Build rotation matrices
            free_rotmats = directions_rolls_to_rotmats(free_dirs, free_rolls)

            # Batch visibility  --  returns (n_free, M) visibility matrix
            V_gpu, _ = visibility_query.compute_visibility_batch(free_pos, free_rotmats)

            # Deficit-weighted coverage: points needing more covers contribute more
            weighted_coverage = V_gpu.astype(cp.float32) @ deficit_gpu
            f_obs = weighted_coverage / deficit_total

            # Travel cost
            travel_cost = cp.zeros(len(free_idx), dtype=cp.float32)
            if avg_pos_gpu is not None and travel_weight > 0:
                pos_dist = cp.linalg.norm(free_pos - avg_pos_gpu, axis=1) / max(spatial_diag, 1e-6)

                angle_dist = cp.zeros(len(free_idx), dtype=cp.float32)
                if avg_rotmat_gpu is not None:
                    R_rel = cp.matmul(free_rotmats, avg_rotmat_gpu.T)
                    traces = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
                    angle_dist = cp.arccos(cp.clip((traces - 1) / 2, -1, 1)) / cp.float32(np.pi)

                rot_frac = OPT_SAMPLER_TRAVEL_ROT_FRACTION
                travel_cost = (1.0 - rot_frac) * pos_dist + rot_frac * angle_dist

            scores_gpu[free_idx] = -f_obs + travel_weight * travel_cost
            return scores_gpu

        return objective_fn

    # ── Single-round optimization ────────────────────────────────────────

    def _optimize_one(
        self,
        lo_gpu,
        ranges_gpu,
        deficit_gpu,
        visibility_query,
        objective_fn,
        popsize,
        maxiter,
        verbose,
        center_init_norm=None,
    ):
        """Run one round of backend optimization.

        Returns ``(best_pos_gpu, best_rotmat_gpu, score, used_fallback)``.
        """
        # Run backend
        best_norm = self.backend.optimize(
            objective_fn,
            n_dims=6,
            popsize=popsize,
            maxiter=maxiter,
            verbose=verbose,
            center_init=center_init_norm,
        )
        best_norm = cp.clip(best_norm, 0.0, 1.0)

        # If the backend converged into occupied space, fall back to the
        # warm-start centre (known-feasible by _generate_warm_start). This
        # prevents a single bad CMA-ES run from aborting sample_optimized.
        best_pos_check = (best_norm * ranges_gpu + lo_gpu)[:3]
        used_fallback = False
        if not bool(self._is_free(best_pos_check[cp.newaxis])[0]) and center_init_norm is not None:
            logger.warning("  Backend returned infeasible pos; falling back to warm-start")
            best_norm = cp.clip(center_init_norm, 0.0, 1.0)
            used_fallback = True

        # Denormalise best solution on GPU
        best_real = best_norm * ranges_gpu + lo_gpu

        best_pos_gpu = best_real[:3]
        theta_val = best_real[3]
        phi_val = best_real[4]
        roll_val = best_real[5]

        best_dir = cp.stack(
            [
                cp.sin(theta_val) * cp.cos(phi_val),
                cp.sin(theta_val) * cp.sin(phi_val),
                cp.cos(theta_val),
            ]
        ).astype(cp.float32)

        best_rotmat_gpu = directions_rolls_to_rotmats(best_dir[cp.newaxis], roll_val[cp.newaxis])[0]

        # Recompute deficit-weighted score for best viewpoint
        V_best, _ = visibility_query.compute_visibility_batch(
            best_pos_gpu[cp.newaxis], best_rotmat_gpu[cp.newaxis]
        )
        visible_mask = V_best[0].astype(cp.bool_)
        best_score = int((deficit_gpu[visible_mask]).sum())

        return best_pos_gpu, best_rotmat_gpu, best_score, used_fallback
