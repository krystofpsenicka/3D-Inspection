"""VRP core — data types, environment, distance computation, utilities."""

from .types import VRPBackend, VRPResult, ExecutionResult, PlanningStats, PipelineConfig
from shared.occupancy_grid import OccupancyGrid
from .distance_matrix import compute_distance_matrix
from .waypoint_loader import load_waypoints, load_viewpoints_gpu
from .robot_config import load_local_robot_config
from .collision import find_trajectory_collisions
from .serialization import save_solution, load_solution

__all__ = [
    "VRPBackend", "VRPResult", "ExecutionResult", "PlanningStats", "PipelineConfig",
    "OccupancyGrid",
    "compute_distance_matrix",
    "load_waypoints", "load_viewpoints_gpu",
    "load_local_robot_config",
    "find_trajectory_collisions",
    "save_solution", "load_solution",
]
