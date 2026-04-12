"""
DEPRECATED: cuOpt is now called directly in-process in mip_solver_gpu.py.
This file is no longer used and can be safely deleted.

---

Original docstring:

VRP MIP – cuOpt MILP Subprocess Script
=======================================

Runs **inside** the ``rapids_solver`` conda environment.  Invoked by
:class:`MIPSolverGPU` with a JSON config path containing the MPS file
path, solver parameters, and optional warm-start values.

Input JSON keys:
    mps_path      : str   — path to the MPS model file
    time_limit    : int   — solver time limit (seconds)
    mip_gap       : float — relative optimality gap
    num_vehicles  : int   — number of vehicles
    depots        : list  — per-vehicle depot indices
    n             : int   — number of nodes
    warm_start    : dict  — optional variable name -> value mapping

Output JSON keys (written to ``out_path``):
    routes  : list[list[int]]
    status  : "success" or error string
    solver  : "cuopt_mip"
"""

from __future__ import annotations

import json
import sys


def _extract_routes_gpu(var_values: dict, num_vehicles: int, depots: list[int], n: int):
    """GPU-vectorized route extraction from solved variable values."""
    import cupy as _cp

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
    next_node = _cp.full((K, n), -1, dtype=_cp.int32)
    next_node[
        _cp.array(active_v, dtype=_cp.int32),
        _cp.array(active_i, dtype=_cp.int32),
    ] = _cp.array(active_j, dtype=_cp.int32)

    is_depot = _cp.zeros(n, dtype=_cp.bool_)
    for d in depot_set:
        is_depot[d] = True

    depots_gpu = _cp.array(depots, dtype=_cp.int32)
    vehicle_range = _cp.arange(K, dtype=_cp.int32)
    current = next_node[vehicle_range, depots_gpu]
    active = (current >= 0) & ~is_depot[current]

    route_matrix = _cp.full((K, n), -1, dtype=_cp.int32)
    for step in range(n):
        if not active.any():
            break
        route_matrix[active, step] = current[active]
        next_step = _cp.full(K, -1, dtype=_cp.int32)
        next_step[active] = next_node[vehicle_range[active], current[active]]
        current = next_step
        active = (current >= 0) & ~is_depot[current]

    route_matrix_cpu = _cp.asnumpy(route_matrix)
    return [
        [int(node) for node in route_matrix_cpu[v] if node >= 0]
        for v in range(K)
    ]


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: mip_vrp_subprocess.py <config.json> <output.json>",
              file=sys.stderr)
        sys.exit(1)

    config_path = sys.argv[1]
    out_path = sys.argv[2]

    with open(config_path) as f:
        cfg = json.load(f)

    mps_path = cfg["mps_path"]
    time_limit = int(cfg.get("time_limit", 120))
    mip_gap = float(cfg.get("mip_gap", 0.05))
    num_vehicles = int(cfg["num_vehicles"])
    depots = [int(d) for d in cfg["depots"]]
    n = int(cfg["n"])
    warm_start = cfg.get("warm_start")

    try:
        from cuopt.linear_programming import Solve, SolverSettings  # type: ignore
        from cuopt.linear_programming.cuopt_mps_parser import ParseMps  # type: ignore
        from cuopt.linear_programming.solver.solver_parameters import (  # type: ignore
            CUOPT_TIME_LIMIT, CUOPT_MIP_RELATIVE_GAP,
        )
        import numpy as _np

        data_model = ParseMps(mps_path)

        settings = SolverSettings()
        settings.set_parameter(CUOPT_TIME_LIMIT, str(time_limit))
        settings.set_parameter(CUOPT_MIP_RELATIVE_GAP, str(mip_gap))

        if warm_start:
            var_names = data_model.get_variable_names()
            initial = _np.zeros(len(var_names), dtype=_np.float64)
            name_to_idx = {name: i for i, name in enumerate(var_names)}
            for var_name_ws, val_ws in warm_start.items():
                if var_name_ws in name_to_idx:
                    initial[name_to_idx[var_name_ws]] = val_ws
            data_model.set_initial_primal_solution(initial)

        solution = Solve(data_model, settings)

        if solution.get_error_status() != 0:
            raise RuntimeError(f"cuOpt MILP error: {solution.get_error_message()}")

        var_values = solution.get_vars()
        routes = _extract_routes_gpu(var_values, num_vehicles, depots, n)

        print(f"[cuOpt MILP] Solved. routes={routes}", flush=True)
        out = {"routes": routes, "status": "success", "solver": "cuopt_mip"}

    except Exception as exc:
        print(f"[cuOpt MILP] Solver error: {exc}", flush=True, file=sys.stderr)
        out = {"routes": [], "status": f"cuopt_mip_error: {exc}", "solver": "cuopt_mip"}
        with open(out_path, "w") as f:
            json.dump(out, f)
        sys.exit(1)

    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"[cuOpt MILP] Solution written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
