"""GPU backend for VRP MIP — cuOpt MILP solver (in-process)."""

from __future__ import annotations

import logging
import math
import os
import tempfile
from typing import List, Optional

import cupy as cp
import numpy as np

from ._helpers import per_vehicle_costs as _per_vehicle_costs
from ._solver_base import VRPSolverBase
from .mip_model import build_vrp_mip

logger = logging.getLogger(__name__)


def _extract_routes_gpu(var_values: dict, num_vehicles: int, depots: list[int], n: int):
    """GPU-vectorized route extraction from solved variable values."""
    depot_set = set(depots)

    active_i, active_j, active_v = [], [], []
    for var_name, val in var_values.items():
        if var_name.startswith("x_") and val > 0.5:
            parts = var_name.split("_")
            active_i.append(int(parts[1]))
            active_j.append(int(parts[2]))
            active_v.append(int(parts[3]))

    if not active_i:
        return [[] for _ in range(num_vehicles)]

    K = num_vehicles
    next_node = cp.full((K, n), -1, dtype=cp.int32)
    next_node[
        cp.array(active_v, dtype=cp.int32),
        cp.array(active_i, dtype=cp.int32),
    ] = cp.array(active_j, dtype=cp.int32)

    is_depot = cp.zeros(n, dtype=cp.bool_)
    for d in depot_set:
        is_depot[d] = True

    depots_gpu = cp.array(depots, dtype=cp.int32)
    vehicle_range = cp.arange(K, dtype=cp.int32)
    current = next_node[vehicle_range, depots_gpu]
    active = (current >= 0) & ~is_depot[current]

    route_matrix = cp.full((K, n), -1, dtype=cp.int32)
    for step in range(n):
        if not active.any():
            break
        route_matrix[active, step] = current[active]
        next_step = cp.full(K, -1, dtype=cp.int32)
        next_step[active] = next_node[vehicle_range[active], current[active]]
        current = next_step
        active = (current >= 0) & ~is_depot[current]

    route_matrix_cpu = cp.asnumpy(route_matrix)
    return [
        [int(node) for node in route_matrix_cpu[v] if node >= 0]
        for v in range(K)
    ]


class MIPSolverGPU(VRPSolverBase):
    """Combined-objective VRP via MIP — cuOpt MILP solver (GPU).

    Builds the MIP with PuLP, exports to MPS, and solves with the
    cuOpt MILP solver directly in-process.
    """

    def __init__(
        self,
        time_limit: int = 120,
        mip_gap: float = 0.05,
    ):
        self.time_limit = time_limit
        self.mip_gap = mip_gap

    def solve(
        self,
        dist_matrix: cp.ndarray,
        num_vehicles: int,
        depots: list[int],
        alpha: float = 1.0,
        warm_start_routes: Optional[List[List[int]]] = None,
    ):
        from ..core.types import VRPResult

        depot_set = set(depots)
        n = dist_matrix.shape[0]
        n_inspection = n - len(depot_set)
        base_cap = math.ceil(n_inspection / num_vehicles)
        capacity = base_cap + max(1, math.ceil(0.15 * base_cap))

        # PuLP needs numpy for model building
        dist_matrix_np = cp.asnumpy(dist_matrix)

        prob, warm_start = build_vrp_mip(
            dist_matrix_np, num_vehicles, depots, capacity,
            alpha=alpha,
            warm_start_routes=warm_start_routes,
            mip_gap=self.mip_gap,
        )

        mps_path = tempfile.mktemp(suffix=".mps")
        prob.writeMPS(mps_path)

        try:
            from cuopt.linear_programming import Solve, SolverSettings
            from cuopt.linear_programming.cuopt_mps_parser import ParseMps
            from cuopt.linear_programming.solver.solver_parameters import (
                CUOPT_TIME_LIMIT, CUOPT_MIP_RELATIVE_GAP,
            )

            data_model = ParseMps(mps_path)

            settings = SolverSettings()
            settings.set_parameter(CUOPT_TIME_LIMIT, str(self.time_limit))
            settings.set_parameter(CUOPT_MIP_RELATIVE_GAP, str(self.mip_gap))

            if warm_start:
                var_names = data_model.get_variable_names()
                initial = np.zeros(len(var_names), dtype=np.float64)
                name_to_idx = {name: i for i, name in enumerate(var_names)}
                for var_name, val in warm_start.items():
                    if var_name in name_to_idx:
                        initial[name_to_idx[var_name]] = val
                data_model.set_initial_primal_solution(initial)

            logger.info("[MIPSolverGPU] Solving with cuOpt MILP (limit=%ds, gap=%.1f%%) ...",
                        self.time_limit, self.mip_gap * 100)
            solution = Solve(data_model, settings)

            if solution.get_error_status() != 0:
                raise RuntimeError(f"cuOpt MILP error: {solution.get_error_message()}")

            var_values = solution.get_vars()
            routes = _extract_routes_gpu(var_values, num_vehicles, depots, n)

            per_v = _per_vehicle_costs(routes, dist_matrix, depots)
            makespan = max(per_v) if per_v else 0.0
            total_cost = sum(per_v)

            logger.info("[MIPSolverGPU] makespan=%.2f  total_cost=%.2f  "
                        "per_vehicle=%s",
                        makespan, total_cost,
                        [f"{c:.1f}" for c in per_v])

            return VRPResult(
                routes=routes,
                total_cost=total_cost,
                makespan=makespan,
                per_vehicle_costs=per_v,
                solver="cuopt_mip",
                status="success",
            )

        except Exception as exc:
            logger.error("[MIPSolverGPU] Solver error: %s", exc)
            return VRPResult(routes=[], total_cost=float("inf"),
                             solver="cuopt_mip", status=f"error: {exc}")
        finally:
            try:
                os.unlink(mps_path)
            except FileNotFoundError:
                pass
