"""Helpers for targeted resampling toward uncovered surface regions."""

import logging

import numpy as np
import cupy as cp

from ..core.constants import DE_POPSIZE, DE_MAXITER, DE_TRAVEL_WEIGHT

logger = logging.getLogger(__name__)


def compute_proximity_weights(feasible_gpu, uncovered_gpu, sigma, nn_func):
    """Exponential proximity weighting toward nearest uncovered point.

    Args:
        feasible_gpu:  (N, 3) CuPy array — feasible viewpoint positions.
        uncovered_gpu: (U, 3) CuPy array — uncovered surface points.
        sigma:         length scale for exponential decay.
        nn_func:       callable(query, targets) → (distances, indices).

    Returns:
        (N,) CuPy float32 — proximity weights (higher = closer to uncovered).
    """
    dists, _ = nn_func(feasible_gpu, uncovered_gpu)
    return cp.exp(-dists / sigma)


def optimize_viewpoint_de(feasible_bounds, uncovered_mask_gpu,
                          visibility_query, is_free_fn,
                          existing_candidates,
                          travel_weight=DE_TRAVEL_WEIGHT,
                          popsize=DE_POPSIZE, maxiter=DE_MAXITER,
                          verbose=False):
    """Find the viewpoint maximizing coverage of uncovered points via DE.

    Objective (following Glorieux et al. 2020):
      minimize  -f_obs + travel_weight * f_trav
    where:
      f_obs  = |visible intersect uncovered|  (coverage of uncovered points)
      f_trav = ||pose - avg_pose||    (distance to mean pose of existing VPs)

    Args:
        feasible_bounds: [(xmin,xmax), (ymin,ymax), (zmin,zmax)]
        uncovered_mask_gpu: (num_points,) CuPy bool — True for uncovered
        visibility_query: for batch visibility computation
        is_free_fn: callable(positions_gpu) -> CuPy bool array
        existing_candidates: list of (pos, quat) already selected
        travel_weight: weight of travel cost relative to coverage

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

    def objective(X):
        """Vectorized objective: X is (N, 6) population matrix."""
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

        scores = np.full(N, 1e6)  # penalty for infeasible
        if len(free_idx) == 0:
            return scores

        # Build candidates as (pos, quaternion)
        free_cands = []
        for i in free_idx:
            q = direction_roll_to_quaternion(directions[i], roll[i])
            free_cands.append((positions[i], q))

        vis_map, _ = visibility_query.compute_visibility_for_all_candidates(free_cands)

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
                pos_dist = float(np.linalg.norm(positions[orig_i] - avg_pos))
                cand_rotvec = R.from_quat([
                    free_cands[j][1][1], free_cands[j][1][2],
                    free_cands[j][1][3], free_cands[j][1][0]
                ]).as_rotvec()
                angle_dist = float(np.linalg.norm(cand_rotvec - avg_rotvec))
                travel_cost = pos_dist + angle_dist

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
        test_vis, _ = visibility_query.compute_visibility_for_all_candidates([cand])
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
    vis_map, _ = visibility_query.compute_visibility_for_all_candidates([best_cand])
    visible = vis_map.get(0, np.array([], dtype=np.int64))
    if len(visible) > 0:
        best_score = int(uncovered_mask_gpu[cp.asarray(visible)].sum())
    else:
        best_score = 0

    return best_cand, best_score
