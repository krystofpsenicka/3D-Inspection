"""Fixtures for the 3D-Inspection tests."""
import cupy as cp
import numpy as np
import pytest
from shared.occupancy_grid import OccupancyGrid


@pytest.fixture
def small_og():
    """20x20x20 grid with a 4x4x4 obstacle block at centre (voxels 8:12)."""
    grid = cp.zeros((20, 20, 20), dtype=cp.bool_)
    grid[8:12, 8:12, 8:12] = True
    return OccupancyGrid(
        grid=grid,
        origin=cp.array([0.0, 0.0, 0.0]),
        resolution=0.5,
    )


@pytest.fixture
def corridor_og():
    """10x10x10 grid with a wall at x=5 and a 1-voxel gap at (5,5,5).
    Used for pathfinding tests — forces a path through the gap."""
    grid = cp.zeros((10, 10, 10), dtype=cp.bool_)
    grid[5, :, :] = True       # wall at x=5
    grid[5, 5, 5] = False      # single gap
    return OccupancyGrid(
        grid=grid,
        origin=cp.zeros(3),
        resolution=1.0,
    )


@pytest.fixture
def og(small_og):
    """Alias for small_og — used by test_occupancy_grid."""
    return small_og


@pytest.fixture
def rng():
    return np.random.RandomState(42)
