"""CPU backend for VRP MIP."""

from __future__ import annotations

import logging

import cupy as cp

from ._helpers import per_vehicle_costs as _per_vehicle_costs
from ._solver_base import VRPSolverBase
from .mip_model import build_vrp_mip, extract_routes

logger = logging.getLogger(__name__)


class MIPSolverCPU(VRPSolverBase):
    """Combined-objective VRP via MIP  --  HiGHS (CPU).

    Uses the shared MIP formulation from :func:`build_vrp_mip` and
    solves with the HiGHS solver via PuLP.
    """

    def __init__(self, time_limit: int = 120, mip_gap: float = 0.05):
        self.time_limit = time_limit
        self.mip_gap = mip_gap

    def solve(
        self,
        dist_matrix: cp.ndarray,
        num_vehicles: int,
        depots: list[int],
        alpha: float = 1.0,
        warm_start_routes: list[list[int]] | None = None,
        beta_aware_filter: bool = True,
        forbidden_pair_cuts: bool = True,
    ):
        from ..core.types import VRPResult

        depot_set = set(depots)
        n = dist_matrix.shape[0]

        try:
            import pulp
        except ImportError:
            logger.error("[MIPSolverCPU] PuLP not installed.")
            return VRPResult(
                routes=[], total_cost=float("inf"), solver="highs_mip", status="import_error: pulp"
            )

        # PuLP needs numpy; convert once
        dist_matrix_np = cp.asnumpy(dist_matrix)

        prob, warm_start = build_vrp_mip(
            dist_matrix_np,
            num_vehicles,
            depots,
            alpha=alpha,
            warm_start_routes=warm_start_routes,
            mip_gap=self.mip_gap,
            beta_aware_filter=beta_aware_filter,
            forbidden_pair_cuts=forbidden_pair_cuts,
        )

        solver = pulp.HiGHS(
            timeLimit=self.time_limit,
            gapRel=self.mip_gap,
            msg=1,
        )

        if warm_start:
            for var in prob.variables():
                if var.name in warm_start:
                    var.varValue = warm_start[var.name]

        logger.info(
            "[MIPSolverCPU] Solving with HiGHS (limit=%ds, gap=%.1f%%) ...",
            self.time_limit,
            self.mip_gap * 100,
        )

        prob.solve(solver)

        status_str = pulp.LpStatus[prob.status]
        logger.info("[MIPSolverCPU] Solver status: %s", status_str)

        if prob.status not in (pulp.constants.LpStatusOptimal,):
            has_solution = any(
                v.varValue is not None and v.varValue > 0.5
                for v in prob.variables()
                if v.name.startswith("x_")
            )
            if not has_solution:
                logger.warning("[MIPSolverCPU] No feasible solution found.")
                return VRPResult(
                    routes=[],
                    total_cost=float("inf"),
                    solver="highs_mip",
                    status=f"no_solution ({status_str})",
                )

        routes = extract_routes(prob, num_vehicles, depots, n)
        per_v = _per_vehicle_costs(routes, dist_matrix, depots)
        makespan = max(per_v) if per_v else 0.0
        total_cost = sum(per_v)

        # Best-effort dual-bound extraction. PuLP's HiGHS wrapper exposes
        # the HiGHS Highs object as prob.solverModel on recent versions;
        # the MIP dual bound is getObjectiveBound(). Older versions don't
        # expose this  --  leave best_bound=0.0 in that case.
        best_bound = 0.0
        try:
            model = getattr(prob, "solverModel", None)
            if model is not None and hasattr(model, "getObjectiveBound"):
                best_bound = float(model.getObjectiveBound())
        except Exception as exc:
            logger.debug("[MIPSolverCPU] dual-bound read failed: %s", exc)

        logger.info(
            "[MIPSolverCPU] makespan=%.2f  total_cost=%.2f  best_bound=%.2f  per_vehicle=%s",
            makespan,
            total_cost,
            best_bound,
            [f"{c:.1f}" for c in per_v],
        )

        return VRPResult(
            routes=routes,
            total_cost=total_cost,
            makespan=makespan,
            per_vehicle_costs=per_v,
            best_bound=best_bound,
            solver="highs_mip",
            status="success",
        )
