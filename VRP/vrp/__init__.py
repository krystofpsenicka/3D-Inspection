"""VRP solver — MIP backends for combined-objective VRP."""

from .vrp_solver import solve_vrp
from ._solver_base import VRPSolverBase
from .mip_solver_cpu import MIPSolverCPU
from .mip_solver_gpu import MIPSolverGPU
from ..core.types import VRPResult

__all__ = ["solve_vrp", "VRPSolverBase", "MIPSolverCPU", "MIPSolverGPU", "VRPResult"]
