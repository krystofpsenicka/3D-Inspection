"""Shared fixtures for the 3D-Inspection test suite."""
import numpy as np
import pytest
from shared.occupancy_grid import OccupancyGrid


@pytest.fixture
def small_og():
    """20x20x20 grid with a 4x4x4 obstacle block at centre (voxels 8:12)."""
    grid = np.zeros((20, 20, 20), dtype=bool)
    grid[8:12, 8:12, 8:12] = True
    return OccupancyGrid(
        grid=grid,
        origin=np.array([0.0, 0.0, 0.0]),
        resolution=0.5,
    )


@pytest.fixture
def corridor_og():
    """10x10x10 grid with a wall at x=5 and a 1-voxel gap at (5,5,5).
    Used for A* pathfinding tests — forces a path through the gap."""
    grid = np.zeros((10, 10, 10), dtype=bool)
    grid[5, :, :] = True       # solid wall at x=5
    grid[5, 5, 5] = False      # single gap
    return OccupancyGrid(
        grid=grid,
        origin=np.zeros(3),
        resolution=1.0,
    )


@pytest.fixture
def rng():
    return np.random.RandomState(42)
