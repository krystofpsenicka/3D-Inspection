"""GPU backend for VRP MIP — cuOpt MILP solver via subprocess."""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import tempfile
from typing import List, Optional

import cupy as cp

from ._helpers import per_vehicle_costs as _per_vehicle_costs
from ._solver_base import VRPSolverBase
from .mip_model import build_vrp_mip

logger = logging.getLogger(__name__)


class MIPSolverGPU(VRPSolverBase):
    """Combined-objective VRP via MIP — cuOpt MILP solver (GPU).

    Builds the MIP with PuLP, exports to MPS, and runs the cuOpt MILP
    solver in a subprocess inside the ``rapids_solver`` conda environment.
    """

    def __init__(
        self,
        rapids_python: str = "",
        time_limit: int = 120,
        mip_gap: float = 0.05,
        timeout: int = 300,
    ):
        from ..core.constants import RAPIDS_PYTHON, VRP_ROOT
        self.rapids_python = os.path.expanduser(rapids_python or RAPIDS_PYTHON)
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.timeout = timeout
        self._script = os.path.join(VRP_ROOT, "vrp", "mip_vrp_subprocess.py")

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

        cfg = {
            "mps_path": mps_path,
            "time_limit": self.time_limit,
            "mip_gap": self.mip_gap,
            "num_vehicles": num_vehicles,
            "depots": depots,
            "n": n,
        }
        if warm_start:
            cfg["warm_start"] = warm_start

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as cfg_f:
            json.dump(cfg, cfg_f)
            cfg_path = cfg_f.name

        out_path = cfg_path.replace(".json", "_out.json")

        try:
            cmd = [self.rapids_python, self._script, cfg_path, out_path]
            logger.info("[MIPSolverGPU] Running cuOpt MILP subprocess: %s",
                        " ".join(cmd))

            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout,
            )

            if proc.returncode != 0:
                logger.warning(
                    "[MIPSolverGPU] Subprocess failed (rc=%d):\n%s",
                    proc.returncode, proc.stderr,
                )
                return VRPResult(
                    routes=[], total_cost=float("inf"),
                    solver="cuopt_mip",
                    status=f"subprocess_error rc={proc.returncode}",
                )

            with open(out_path) as f:
                result = json.load(f)

            if result.get("status") != "success":
                return VRPResult(
                    routes=[], total_cost=float("inf"),
                    solver="cuopt_mip",
                    status=result.get("status", "unknown"),
                )

            routes = result["routes"]
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

        except subprocess.TimeoutExpired:
            logger.error("[MIPSolverGPU] Subprocess timed out after %ds",
                         self.timeout)
            return VRPResult(routes=[], total_cost=float("inf"),
                             solver="cuopt_mip", status="timeout")
        except Exception as exc:
            logger.error("[MIPSolverGPU] Unexpected error: %s", exc)
            return VRPResult(routes=[], total_cost=float("inf"),
                             solver="cuopt_mip", status=f"error: {exc}")
        finally:
            for p in (mps_path, cfg_path, out_path):
                try:
                    os.unlink(p)
                except FileNotFoundError:
                    pass
