"""VRP core – Environment representation & distance computation."""

from .occupancy_grid import OccupancyGrid, build_occupancy_grid, get_mesh_world_bounds
from .gpu_distance_matrix import compute_distance_matrix, build_route_path_cache
from .waypoint_loader import load_waypoints
