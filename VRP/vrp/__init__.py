"""VRP solver — MIP backends for combined-objective VRP."""

from .vrp_solver import solve_vrp
from .mip_makespan_solver import MIPMakespanCPU, MIPMakespanGPU
from ..core.types import VRPResult

__all__ = ["solve_vrp", "MIPMakespanCPU", "MIPMakespanGPU", "VRPResult"]
