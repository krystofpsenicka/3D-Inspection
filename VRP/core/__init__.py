"""VRP core  --  data types, environment, distance computation, utilities."""

from shared.occupancy_grid import OccupancyGrid
from VRP.utils.collision import find_trajectory_collisions
from VRP.utils.serialization import load_solution, save_solution

from .distance_matrix import compute_distance_matrix
from .geometry import viewpoints_to_robot_waypoints
from .types import ExecutionResult, PlanningStats, VRPBackend, VRPResult

__all__ = [
    "VRPBackend",
    "VRPResult",
    "ExecutionResult",
    "PlanningStats",
    "OccupancyGrid",
    "compute_distance_matrix",
    "find_trajectory_collisions",
    "viewpoints_to_robot_waypoints",
    "save_solution",
    "load_solution",
]
