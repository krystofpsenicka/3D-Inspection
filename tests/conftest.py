"""Fixtures for the 3D-Inspection tests."""

import cupy as cp
import numpy as np
import open3d as o3d
import pytest

from shared.occupancy_grid import OccupancyGrid

# Shared tolerances.
ATOL_F32 = 1e-6  # float32 has ~7-digit precision; 1e-7 is below eps after a few ops
ATOL_F64 = 1e-9
RTOL_PATH = 0.05


@pytest.fixture(autouse=True)
def _seed_cupy_rng():
    """Seed cupy's global RNG before every test so ``cp.random.*`` calls
    are reproducible across runs."""
    cp.random.seed(0)


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
    Used for pathfinding tests  --  forces a path through the gap."""
    grid = cp.zeros((10, 10, 10), dtype=cp.bool_)
    grid[5, :, :] = True  # wall at x=5
    grid[5, 5, 5] = False  # single gap
    return OccupancyGrid(
        grid=grid,
        origin=cp.zeros(3),
        resolution=1.0,
    )


@pytest.fixture
def og(small_og):
    """Alias for small_og  --  used by test_occupancy_grid."""
    return small_og


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def cube_mesh():
    """Unit cube TriangleMesh centered at origin."""
    mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
    mesh.translate((-0.5, -0.5, -0.5))
    return mesh


@pytest.fixture
def flat_target_cloud():
    """100 points on the plane z=0 with normals = (0, 0, 1).

    A 10x10 grid covering [-1, 1] x [-1, 1] -- useful as visibility targets
    for tests that place a camera above looking down.
    """
    xs, ys = np.meshgrid(np.linspace(-1.0, 1.0, 10), np.linspace(-1.0, 1.0, 10))
    pts = np.stack([xs.ravel(), ys.ravel(), np.zeros(100)], axis=1).astype(np.float64)
    normals = np.tile(np.array([0.0, 0.0, 1.0]), (100, 1))
    return pts, normals


@pytest.fixture
def random_visibility_matrix():
    """Factory: build a deterministic random (N, M) bool visibility matrix.

    Each call uses its own RandomState so different calls within a test
    produce different matrices.
    """

    def _make(
        n_candidates: int,
        n_points: int,
        density: float = 0.15,
        seed: int = 42,
    ) -> np.ndarray:
        local_rng = np.random.RandomState(seed)
        return local_rng.random((n_candidates, n_points)) < density

    return _make
