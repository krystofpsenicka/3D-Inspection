"""VRP Planner package."""

# Configuration
from VRP.config import (
    VOXEL_RESOLUTION,
    ROBOT_RADIUS,
    INFLATION_VOXELS,
    BROV_CUBOID_DIMS,
    TRAJOPT_HORIZON,
    STATIC_OBSTACLES,
    RAPIDS_PYTHON,
)

# Core
from VRP.core.occupancy_grid import OccupancyGrid, build_occupancy_grid, get_mesh_world_bounds
from VRP.core.gpu_distance_matrix import compute_distance_matrix, build_route_path_cache
from VRP.core.waypoint_loader import load_waypoints

# Solver
from VRP.solver.vrp_solver import VRPResult, solve_vrp

# Routing
from VRP.routing.route_executor import ExecutionResult, RouteExecutor

# Helpers
from VRP.utils import save_solution, load_solution

__all__ = [
    # Configuration
    "config",
    # Subpackages
    "core",
    "solver",
    "routing",
    "viz",
    "scripts",
    # Re-exports
    "OccupancyGrid",
    "build_occupancy_grid",
    "get_mesh_world_bounds",
    "compute_distance_matrix",
    "build_route_path_cache",
    "load_waypoints",
    "VRPResult",
    "solve_vrp",
    "ExecutionResult",
    "RouteExecutor",
    "save_solution",
    "load_solution",
    # Helpers
    "utils",
]
