"""VRP solver – VRP solving backends (cuOpt / OR-Tools)."""

from .vrp_solver import VRPResult, solve_vrp, ORToolsSolver, GPUSolver
