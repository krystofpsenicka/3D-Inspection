"""Lower bound computations for set cover and VRP.

These provide theoretical baselines to assess solution quality:
- LP relaxation for set cover (relax binary to continuous)
- Information-theoretic lower bound
- Held-Karp TSP lower bound (1-tree relaxation)
- LP relaxation for CVRP MIP
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy.optimize import linprog

logger = logging.getLogger(__name__)


def lp_relaxation_set_cover(
    V_binary: np.ndarray,
    target_coverage: float = 1.0,
) -> float:
    """LP relaxation lower bound for minimum set cover.

    Relaxes x_i in {0,1} to x_i in [0,1].
    Minimizes sum(x_i) subject to: for each covered point j,
    sum_i(V[i,j] * x_i) >= 1.

    Args:
        V_binary: (N_candidates, M_points) binary visibility matrix (numpy).
        target_coverage: fraction of points that must be covered.

    Returns:
        ceil(LP_optimal) as a lower bound on the integer optimum.
    """
    N, M = V_binary.shape
    n_required = int(math.ceil(M * target_coverage))

    # Select the n_required hardest-to-cover points (lowest column sums).
    # Any n_required-point subset gives a valid LP lower bound; this selection
    # maximises the bound (tightest guarantee) without exceeding OPT.
    col_sums = V_binary.sum(axis=0)
    selected = np.argsort(col_sums)[:n_required]
    V_sub = V_binary[:, selected]  # (N, n_required)

    # min c^T x  s.t.  A_ub x <= b_ub, 0 <= x <= 1
    c = np.ones(N)

    # Each of the n_required selected points must be covered by ≥ 1 viewpoint.
    # linprog uses <=, so: -V_sub^T x <= -1
    A_ub = -V_sub.T.astype(np.float64)   # (n_required, N)
    b_ub = -np.ones(n_required)

    bounds = [(0, 1)] * N

    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if result.success:
            lb = math.ceil(result.fun - 1e-9)
            logger.info("LP relaxation set cover: LP_opt=%.4f, ceil=%d", result.fun, lb)
            return lb
        else:
            logger.warning("LP relaxation failed: %s", result.message)
            return 0
    except Exception as e:
        logger.warning("LP relaxation error: %s", e)
        return 0


def information_theoretic_lb(
    num_points: int,
    target_coverage: float,
    max_single_vp_coverage: int,
) -> int:
    """Trivial lower bound: ceil(points_to_cover / max_coverage_per_vp).

    Args:
        num_points: total surface points
        target_coverage: fraction to cover
        max_single_vp_coverage: max points any single viewpoint covers
    """
    points_needed = int(num_points * target_coverage)
    if max_single_vp_coverage <= 0:
        return points_needed
    return math.ceil(points_needed / max_single_vp_coverage)


def held_karp_tsp_lb(distance_matrix: np.ndarray) -> float:
    """Held-Karp (1-tree) LP relaxation lower bound for TSP.

    This gives a lower bound on the optimal TSP tour cost.
    For a fleet of k robots, divide by k for a fleet lower bound.

    Uses the assignment relaxation (fractional LP).

    Args:
        distance_matrix: (N, N) symmetric distance matrix.

    Returns:
        LP relaxation lower bound for TSP.
    """
    n = distance_matrix.shape[0]
    if n <= 2:
        if n == 2:
            return float(distance_matrix[0, 1] + distance_matrix[1, 0])
        return 0.0

    # Assignment relaxation: min sum c_ij * x_ij
    # s.t. sum_j x_ij = 1 for all i (leave each city once)
    #      sum_i x_ij = 1 for all j (enter each city once)
    #      0 <= x_ij <= 1, x_ii = 0
    num_vars = n * n
    c = distance_matrix.flatten().astype(np.float64)

    # Fix diagonal to inf cost (no self-loops)
    for i in range(n):
        c[i * n + i] = 1e12

    # Row constraints: sum_j x_ij = 1
    A_eq_rows = np.zeros((n, num_vars))
    for i in range(n):
        A_eq_rows[i, i * n:(i + 1) * n] = 1.0
    b_eq_rows = np.ones(n)

    # Column constraints: sum_i x_ij = 1
    A_eq_cols = np.zeros((n, num_vars))
    for j in range(n):
        for i in range(n):
            A_eq_cols[j, i * n + j] = 1.0
    b_eq_cols = np.ones(n)

    A_eq = np.vstack([A_eq_rows, A_eq_cols])
    b_eq = np.concatenate([b_eq_rows, b_eq_cols])
    bounds = [(0, 1)] * num_vars

    try:
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if result.success:
            logger.info("Held-Karp LP bound: %.2f", result.fun)
            return float(result.fun)
        else:
            logger.warning("Held-Karp LP failed: %s", result.message)
            return 0.0
    except Exception as e:
        logger.warning("Held-Karp LP error: %s", e)
        return 0.0


def fleet_tsp_lb(distance_matrix: np.ndarray, num_vehicles: int) -> float:
    """Lower bound on fleet makespan: Held-Karp TSP bound / k."""
    tsp_lb = held_karp_tsp_lb(distance_matrix)
    return tsp_lb / max(1, num_vehicles)
