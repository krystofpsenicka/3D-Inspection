"""VRP solver entry point — combined-objective MIP.

``solve_vrp()`` builds a nearest-neighbour warm-start
and solves the VRP with a blended makespan/total-distance objective.

Two MIP backends are available:
- ``MIPSolverGPU`` — cuOpt MILP solver (GPU).
- ``MIPSolverCPU`` — PuLP + HiGHS solver (CPU).
"""

from __future__ import annotations

import logging

import cupy as cp

from ..core.constants import (
    MIP_GAP,
    MIP_TIME_LIMIT,
    RAPIDS_PYTHON,
)
from ..core.types import VRPBackend, VRPResult
from ._helpers import (
    nearest_neighbor_warmstart as _nearest_neighbor_warmstart,
)

logger = logging.getLogger(__name__)


# ─── Unified solver entry-point ──────────────────────────────────────────────

def solve_vrp(
    dist_matrix: cp.ndarray,
    num_vehicles: int,
    depots: list[int],
    alpha: float = 1.0,
    backend: VRPBackend = VRPBackend.HIGHS,
    rapids_python: str = RAPIDS_PYTHON,
    time_limit: int = MIP_TIME_LIMIT,
    gpu_timeout: int = 300,
    mip_gap: float = MIP_GAP,
) -> VRPResult:
    """Solve the VRP using the specified MIP backend.

    Args:
        dist_matrix: Square (N, N) CuPy cost matrix.
        num_vehicles: Number of AUVs / robots.
        depots: Per-vehicle depot index list.
        alpha: Objective blending in [0, 1]. 1.0 = makespan, 0.0 = total dist.
        backend: ``VRPBackend.CUOPT`` (GPU) or ``VRPBackend.HIGHS`` (CPU).
        rapids_python: Python binary path for the rapids_solver env.
        time_limit: MIP solver time budget (seconds).
        gpu_timeout: Wall-clock timeout for cuOpt subprocess (seconds).
        mip_gap: Relative optimality gap.

    Raises:
        RuntimeError: if the selected backend fails to find a solution.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if not isinstance(backend, VRPBackend):
        raise TypeError(f"backend must be a VRPBackend, got {backend!r}")

    from .mip_solver_cpu import MIPSolverCPU
    from .mip_solver_gpu import MIPSolverGPU

    warm_start_routes = _nearest_neighbor_warmstart(
        dist_matrix, num_vehicles, depots,
    )
    logger.info("[solve_vrp] Nearest-neighbour warm-start ready.")

    if backend == VRPBackend.CUOPT:
        solver = MIPSolverGPU(
            rapids_python=rapids_python,
            time_limit=time_limit,
            mip_gap=mip_gap,
            timeout=gpu_timeout,
        )
    else:
        solver = MIPSolverCPU(
            time_limit=time_limit,
            mip_gap=mip_gap,
        )

    result = solver.solve(
        dist_matrix, num_vehicles, depots,
        alpha=alpha,
        warm_start_routes=warm_start_routes,
    )

    if result.status != "success":
        raise RuntimeError(
            f"[solve_vrp] {backend} solver failed: {result.status}"
        )

    result.alpha = alpha
    result.objective_value = (
        alpha * result.makespan + (1 - alpha) * result.total_cost
    )
    return result
