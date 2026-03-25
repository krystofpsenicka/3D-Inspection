"""Differential Evolution viewpoint sampling for optimal coverage."""

import logging

import cupy as cp
import numpy as np
from typing import List, Tuple

from ...core.constants import DE_POPSIZE, DE_MAXITER, DE_TRAVEL_WEIGHT
from .base import ViewpointSamplerBase

logger = logging.getLogger(__name__)


class DEViewpointSampler(ViewpointSamplerBase):
    """Iterative viewpoint optimization via Differential Evolution.

    Each round runs scipy's DE to find the single best viewpoint for
    remaining uncovered surface points, then updates the uncovered mask.
    Does not use distance/curvature weighting — DE directly optimizes
    a coverage + travel cost objective.
    """

    def sample_de(self, n_rounds: int, uncovered_mask_gpu,
                  visibility_query,
                  existing_candidates: list | None = None,
                  side: str = "outside",
                  target_coverage: float = 1.0,
                  travel_weight: float = DE_TRAVEL_WEIGHT,
                  popsize: int = DE_POPSIZE,
                  maxiter: int = DE_MAXITER,
                  verbose: bool = False,
                  ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Find optimal viewpoints via iterative Differential Evolution.

        Each round runs DE in 6D pose space (x, y, z, theta, phi, roll)
        to find the viewpoint maximizing coverage of uncovered points
        while penalizing distance from existing viewpoints.

        Args:
            n_rounds:           max number of DE rounds (viewpoints to find).
            uncovered_mask_gpu: (num_points,) CuPy bool — True for uncovered.
            visibility_query:   must implement ``compute_visibility_batch()``.
            existing_candidates: list of (pos, quat) already selected.
            side:               "outside" or "inside" — used for feasible bounds.
            target_coverage:    stop early if this fraction is reached.
            travel_weight:      weight of travel cost vs coverage in objective.
            popsize:            DE population size multiplier.
            maxiter:            max DE generations per round.
            verbose:            log per-generation progress.

        Returns:
            List of (position, quaternion) tuples found by DE.
        """
        if existing_candidates is None:
            existing_candidates = []

        # Get feasible bounds from cached free-space data
        centers_gpu, _, _ = self._get_cached_free_space(
            side, None, 0.95, False)
        centers = cp.asnumpy(centers_gpu)
        feasible_bounds = list(zip(centers.min(axis=0).tolist(),
                                   centers.max(axis=0).tolist()))

        all_candidates = list(existing_candidates)
        new_candidates = []

        # Build a mutable visibility map for tracking coverage
        vis_map = {}

        for round_i in range(n_rounds):
            # Compute current coverage
            covered_mask = cp.zeros(visibility_query.num_points, dtype=cp.bool_)
            covered_mask |= ~uncovered_mask_gpu
            for vis_indices in vis_map.values():
                if len(vis_indices) > 0:
                    covered_mask[cp.asarray(vis_indices)] = True
            current_uncovered = ~covered_mask
            union_coverage = float(covered_mask.sum()) / visibility_query.num_points

            if union_coverage >= target_coverage:
                logger.info("[DESampler] Coverage %.1f%% >= target -- done after %d rounds.",
                            union_coverage * 100, round_i)
                break

            logger.info("[DESampler %d/%d] Coverage=%.1f%%, %d uncovered -- running DE...",
                        round_i + 1, n_rounds, union_coverage * 100,
                        int(current_uncovered.sum()))

            best_cand, score = _optimize_viewpoint_de(
                feasible_bounds, current_uncovered, visibility_query,
                self._is_free, existing_candidates=all_candidates,
                travel_weight=travel_weight, popsize=popsize,
                maxiter=maxiter, verbose=verbose)

            if score == 0:
                logger.info("[DESampler] DE found no useful viewpoint -- stopping.")
                break

            # Compute visibility for best candidate and track it
            new_vis, _ = visibility_query.compute_visibility_batch([best_cand])
            for k, v in new_vis.items():
                vis_map[len(new_candidates) + k] = v

            new_candidates.append(best_cand)
            all_candidates.append(best_cand)

            logger.info("[DESampler %d/%d] DE found VP covering %d uncovered pts.",
                        round_i + 1, n_rounds, score)

        return new_candidates


def _optimize_viewpoint_de(feasible_bounds, uncovered_mask_gpu,
                           visibility_query, is_free_fn,
                           existing_candidates,
                           travel_weight=DE_TRAVEL_WEIGHT,
                           popsize=DE_POPSIZE, maxiter=DE_MAXITER,
                           verbose=False):
    """Find the viewpoint maximizing coverage of uncovered points using Differential Evolution.

    Objective (following Glorieux et al. 2020):
      minimize  -f_obs + travel_weight * f_trav
    where:
      f_obs  = |visible & uncovered|  (coverage of uncovered points)
      f_trav = ||pose - avg_pose||    (distance to mean pose of existing VPs)

    Args:
        feasible_bounds: [(xmin,xmax), (ymin,ymax), (zmin,zmax)]
        uncovered_mask_gpu: (num_points,) CuPy bool — True for uncovered
        visibility_query: for batch visibility computation
        is_free_fn: callable(positions_gpu) -> CuPy bool array
        existing_candidates: list of (pos, quat) already selected
        travel_weight: weight of travel cost in objective

    Returns:
        (best_candidate, best_score) where candidate = (pos, quat)
        and score = number of uncovered points newly covered.
    """
    from scipy.optimize import differential_evolution
    from scipy.spatial.transform import Rotation as R
    from shared.geometry import direction_roll_to_quaternion

    # 6D bounds: x, y, z, theta, phi, roll
    bounds = list(feasible_bounds) + [
        (0, np.pi),        # theta (polar)
        (0, 2 * np.pi),    # phi (azimuth)
        (-np.pi, np.pi),   # roll
    ]

    # Precompute average pose of existing viewpoints for travel cost
    n_uncovered = int(uncovered_mask_gpu.sum())
    avg_pos = None
    avg_rotvec = None
    if existing_candidates:
        all_pos = np.array([c[0] for c in existing_candidates], dtype=np.float64)
        avg_pos = all_pos.mean(axis=0)
        all_rotvec = np.array([
            R.from_quat([c[1][1], c[1][2], c[1][3], c[1][0]]).as_rotvec()
            for c in existing_candidates
        ], dtype=np.float64)
        avg_rotvec = all_rotvec.mean(axis=0)

    spatial_diag = np.sqrt(sum((hi - lo)**2 for lo, hi in feasible_bounds))

    def objective(X):
        """Vectorized objective."""
        X = X.T
        N = X.shape[0]
        positions = X[:, :3].astype(np.float32)
        theta, phi, roll = X[:, 3], X[:, 4], X[:, 5]

        # Spherical -> direction
        directions = np.column_stack([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ]).astype(np.float32)

        # Free-space check on GPU
        pos_gpu = cp.asarray(positions)
        free_mask = cp.asnumpy(is_free_fn(pos_gpu))
        free_idx = np.where(free_mask)[0]

        scores = np.full(N, 1e6)  # penalty for infeasibility
        if len(free_idx) == 0:
            return scores

        # Build candidates as (pos, quaternion)
        free_cands = []
        for i in free_idx:
            q = direction_roll_to_quaternion(directions[i], roll[i])
            free_cands.append((positions[i], q))

        vis_map, _ = visibility_query.compute_visibility_batch(free_cands)

        # Score: -f_obs + travel_weight * f_trav
        for j, orig_i in enumerate(free_idx):
            visible = vis_map.get(j, np.array([], dtype=np.int64))
            if len(visible) > 0:
                vis_gpu = cp.asarray(visible)
                newly_covered = int(uncovered_mask_gpu[vis_gpu].sum())
            else:
                newly_covered = 0

            # Travel cost: distance from this candidate to avg pose
            travel_cost = 0.0
            if avg_pos is not None and travel_weight > 0:
                pos_dist = float(np.linalg.norm(positions[orig_i] - avg_pos)) / spatial_diag
                cand_rotvec = R.from_quat([
                    free_cands[j][1][1], free_cands[j][1][2],
                    free_cands[j][1][3], free_cands[j][1][0]
                ]).as_rotvec()
                angle_dist = float(np.linalg.norm(cand_rotvec - avg_rotvec)) / np.pi
                travel_cost = 0.9 * pos_dist + 0.1 * angle_dist

            # Normalize coverage by n_uncovered so both terms are ~O(1)
            f_obs = newly_covered / max(n_uncovered, 1)
            scores[orig_i] = -f_obs + travel_weight * travel_cost

        return scores

    gen_counter = [0]

    def de_callback(xk, convergence):
        gen_counter[0] += 1
        if not verbose:
            return
        pos = xk[:3].astype(np.float32)
        theta_cb, phi_cb, roll_cb = xk[3], xk[4], xk[5]
        direction = np.array([
            np.sin(theta_cb) * np.cos(phi_cb),
            np.sin(theta_cb) * np.sin(phi_cb),
            np.cos(theta_cb),
        ], dtype=np.float32)
        q = direction_roll_to_quaternion(direction, roll_cb)
        cand = (pos, q)
        test_vis, _ = visibility_query.compute_visibility_batch([cand])
        visible = test_vis.get(0, np.array([], dtype=np.int64))
        if len(visible) > 0:
            new_cov = int(uncovered_mask_gpu[cp.asarray(visible)].sum())
        else:
            new_cov = 0
        logger.info("  DE gen %d: best covers %d/%d uncovered pts (convergence=%.4f)",
                    gen_counter[0], new_cov, n_uncovered, convergence)

    result = differential_evolution(
        objective, bounds, vectorized=True,
        maxiter=maxiter, popsize=popsize,
        seed=np.random.randint(0, 2**31),
        tol=0,
        callback=de_callback,
    )
    logger.info("  DE finished: %d generations, fun=%.4f", result.nit, result.fun)

    best_pos = result.x[:3].astype(np.float32)
    theta, phi, roll = result.x[3], result.x[4], result.x[5]
    best_dir = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ], dtype=np.float32)
    best_quat = direction_roll_to_quaternion(best_dir, roll)

    # Recompute f_obs for the best to report raw coverage score
    best_cand = (best_pos, best_quat)
    vis_map, _ = visibility_query.compute_visibility_batch([best_cand])
    visible = vis_map.get(0, np.array([], dtype=np.int64))
    if len(visible) > 0:
        best_score = int(uncovered_mask_gpu[cp.asarray(visible)].sum())
    else:
        best_score = 0

    return best_cand, best_score
