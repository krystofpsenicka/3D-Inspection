"""VRP Planner package."""

# Types
# Core
from shared.occupancy_grid import OccupancyGrid

# Configuration
from VRP.core.constants import (
    INFLATION_VOXELS,
    ROBOT_RADIUS,
    VOXEL_RESOLUTION,
)
from VRP.core.distance_matrix import compute_distance_matrix
from VRP.core.types import ExecutionResult, PlanningStats, VRPBackend, VRPResult

# Routing
from VRP.mapf.mapf_planner import MultiAgentPathPlanner

# Helpers
from VRP.utils.serialization import load_solution, save_solution

# Solver
from VRP.vrp.vrp_solver import solve_vrp

__all__ = [
    # Types
    "VRPBackend",
    "VRPResult",
    "ExecutionResult",
    "PlanningStats",
    # Configuration
    "VOXEL_RESOLUTION",
    "ROBOT_RADIUS",
    "INFLATION_VOXELS",
    # Core
    "OccupancyGrid",
    "compute_distance_matrix",
    # Solver
    "solve_vrp",
    # Routing
    "MultiAgentPathPlanner",
    # Helpers
    "save_solution",
    "load_solution",
]
