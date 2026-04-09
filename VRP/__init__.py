"""VRP Planner package."""

# Types
from VRP.core.types import VRPBackend, VRPResult, ExecutionResult, PlanningStats, PipelineConfig

# Configuration
from VRP.core.constants import (
    VOXEL_RESOLUTION,
    ROBOT_RADIUS,
    INFLATION_VOXELS,
)

# Core
from shared.occupancy_grid import OccupancyGrid
from VRP.core.distance_matrix import compute_distance_matrix
from VRP.core.waypoint_loader import load_waypoints, load_viewpoints_gpu

# Solver
from VRP.vrp.vrp_solver import solve_vrp

# Routing
from VRP.mapf.route_executor import RouteExecutor

# Helpers
from VRP.core.serialization import save_solution, load_solution

__all__ = [
    # Types
    "VRPBackend", "VRPResult", "ExecutionResult", "PlanningStats", "PipelineConfig",
    # Configuration
    "VOXEL_RESOLUTION", "ROBOT_RADIUS", "INFLATION_VOXELS",
    # Core
    "OccupancyGrid",
    "compute_distance_matrix",
    "load_waypoints", "load_viewpoints_gpu",
    # Solver
    "solve_vrp",
    # Routing
    "RouteExecutor",
    # Helpers
    "save_solution", "load_solution",
]
