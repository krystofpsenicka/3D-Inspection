"""Smoke tests for package imports."""


def test_vrp_import():
    import VRP
    assert hasattr(VRP, "config")


def test_vrp_subpackage_imports():
    from VRP.core import occupancy_grid
    from VRP.solver import vrp_solver
    from VRP.routing import route_executor
    assert occupancy_grid is not None
    assert vrp_solver is not None
    assert route_executor is not None


def test_visibility_import():
    from visibility import GreedyOptimizer, LazyGreedyOptimizer
    assert GreedyOptimizer is not None
    assert LazyGreedyOptimizer is not None


def test_shared_import():
    from shared.occupancy_grid import OccupancyGrid
    from shared.grid_utils import snap_to_free, OFFSETS_26
    from shared.geometry import direction_to_quaternion
    assert OccupancyGrid is not None
    assert snap_to_free is not None
    assert OFFSETS_26 is not None
    assert direction_to_quaternion is not None


def test_static_obstacles_empty():
    from VRP.config import STATIC_OBSTACLES
    assert STATIC_OBSTACLES == {}


def test_obstacle_cuboid_dims_removed():
    """OBSTACLE_CUBOID_DIMS should no longer exist in config."""
    import VRP.config as cfg
    assert not hasattr(cfg, "OBSTACLE_CUBOID_DIMS")
