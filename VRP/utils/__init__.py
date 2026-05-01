"""VRP utility helpers: serialization, occupancy grid wrappers, collision checks."""

from shared.occupancy_grid import OccupancyGrid

from .collision import find_trajectory_collisions
from .serialization import load_solution, save_solution

__all__ = [
    "OccupancyGrid",
    "save_solution",
    "load_solution",
    "find_trajectory_collisions",
]
