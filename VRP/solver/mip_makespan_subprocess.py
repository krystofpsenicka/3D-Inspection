"""
MIP Makespan – cuOpt MILP Subprocess Script
============================================

Runs **inside** the ``rapids_solver`` conda environment.  Invoked by
:class:`MIPMakespanGPU` with a JSON config path containing the MPS file
path, solver parameters, and optional warm-start values.

Input JSON keys:
    mps_path      : str   — path to the MPS model file
    time_limit    : int   — solver time limit (seconds)
    mip_gap       : float — relative optimality gap
    num_vehicles  : int   — number of vehicles
    depot         : list  — per-vehicle depot indices
    n             : int   — number of nodes
    warm_start    : dict  — optional variable name → value mapping

Output JSON keys (written to ``out_path``):
    routes  : list[list[int]]
    status  : "success" or error string
    solver  : "cuopt_mip"
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: mip_makespan_subprocess.py <config.json> <output.json>",
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
    depots = [int(d) for d in cfg["depot"]]
    n = int(cfg["n"])
    warm_start = cfg.get("warm_start")

    try:
        from cuopt.linear_programming import Solve, SolverSettings  # type: ignore
        from cuopt.linear_programming.cuopt_mps_parser import ParseMps  # type: ignore
        from cuopt.linear_programming.solver.solver_parameters import (  # type: ignore
            CUOPT_TIME_LIMIT, CUOPT_MIP_RELATIVE_GAP,
        )
        import numpy as _np

        # 1. Parse MPS → DataModel
        data_model = ParseMps(mps_path)

        # 2. Configure solver
        settings = SolverSettings()
        settings.set_parameter(CUOPT_TIME_LIMIT, str(time_limit))
        settings.set_parameter(CUOPT_MIP_RELATIVE_GAP, str(mip_gap))

        # 3. Warm start (optional) — set initial primal solution
        if warm_start:
            var_names = data_model.get_variable_names()
            initial = _np.zeros(len(var_names), dtype=_np.float64)
            name_to_idx = {name: i for i, name in enumerate(var_names)}
            for var_name_ws, val_ws in warm_start.items():
                if var_name_ws in name_to_idx:
                    initial[name_to_idx[var_name_ws]] = val_ws
            data_model.set_initial_primal_solution(initial)

        # 4. Solve
        solution = Solve(data_model, settings)

        # 5. Check result
        if solution.get_error_status() != 0:
            raise RuntimeError(f"cuOpt MILP error: {solution.get_error_message()}")

        # 6. Extract variable values
        var_values = solution.get_vars()  # dict: var_name → value
        depot_set = set(depots)

        # Parse x variables to build adjacency
        adj = {v: {} for v in range(num_vehicles)}
        for var_name, val in var_values.items():
            if var_name.startswith("x_") and val > 0.5:
                parts = var_name.split("_")
                i, j, v = int(parts[1]), int(parts[2]), int(parts[3])
                adj[v][i] = j

        # Trace routes
        routes = []
        for v in range(num_vehicles):
            route = []
            dv = depots[v]
            current = adj[v].get(dv)
            visited = set()
            while current is not None and current not in depot_set and current not in visited:
                route.append(current)
                visited.add(current)
                current = adj[v].get(current)
            routes.append(route)

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
