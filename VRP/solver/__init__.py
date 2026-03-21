"""VRP solver – VRP solving backends (cuOpt / OR-Tools / MIP makespan)."""

from .vrp_solver import VRPResult, solve_vrp, ORToolsSolver, GPUSolver
from .mip_makespan_solver import MIPMakespanCPU, MIPMakespanGPU
