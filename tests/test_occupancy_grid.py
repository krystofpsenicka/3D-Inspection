"""Tests for OccupancyGrid, inflate_grid, and downsample_occupancy_grid."""
import numpy as np
import pytest
from shared.occupancy_grid import (
    OccupancyGrid,
    inflate_grid,
    downsample_occupancy_grid,
)


# ── Coordinate transforms ──────────────────────────────────────────────────


class TestCoordinateTransforms:
    def test_world_to_voxel_round_trip(self, og):
        world_pt = np.array([2.25, 3.75, 1.25])
        voxel = og.world_to_voxel(world_pt)
        world_back = og.voxel_to_world(voxel)
        assert np.allclose(world_pt, world_back, atol=og.resolution)


# ── Free / occupied queries ────────────────────────────────────────────────


class TestFreeOccupied:
    def test_occupied_centre(self, og):
        centre = og.voxel_to_world(np.array([10, 10, 10]))
        assert not og.is_free_world(centre)

    def test_free_corner(self, og):
        corner = og.voxel_to_world(np.array([0, 0, 0]))
        assert og.is_free_world(corner)

    def test_num_free_plus_occupied_equals_total(self, og):
        total = np.prod(og.grid.shape)
        assert og.num_free + og.num_occupied == total


# ── Flat index ─────────────────────────────────────────────────────────────


class TestFlatIndex:
    def test_round_trip(self, og):
        world_pt = og.voxel_to_world(np.array([3, 7, 2]))
        flat = og.world_to_flat_index(world_pt)
        world_back = og.flat_index_to_world(flat)
        assert np.allclose(world_pt, world_back)


# ── Persistence (NPZ round-trip) ──────────────────────────────────────────


class TestSaveLoad:
    def test_npz_round_trip(self, og, tmp_path):
        path = str(tmp_path / "test_og.pkl")
        og.save(path)
        loaded = OccupancyGrid.load(path)
        assert np.array_equal(loaded.grid, og.grid)
        assert np.allclose(loaded.origin, og.origin)
        assert loaded.resolution == og.resolution


# ── inflate_grid ──────────────────────────────────────────────────────────


class TestInflateGrid:
    def test_single_point_sphere(self):
        grid = np.zeros((11, 11, 11), dtype=bool)
        grid[5, 5, 5] = True
        r = 2
        inflated = inflate_grid(grid, r)
        # Voxels within radius should be occupied
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    dist2 = dx**2 + dy**2 + dz**2
                    if dist2 <= r**2:
                        assert inflated[5 + dx, 5 + dy, 5 + dz], \
                            f"({dx},{dy},{dz}) dist2={dist2} should be occupied"

    def test_zero_inflation_identity(self):
        grid = np.zeros((5, 5, 5), dtype=bool)
        grid[2, 2, 2] = True
        result = inflate_grid(grid, 0)
        assert np.array_equal(result, grid)

    def test_monotonic(self):
        grid = np.zeros((10, 10, 10), dtype=bool)
        grid[5, 5, 5] = True
        small = inflate_grid(grid, 1)
        big = inflate_grid(grid, 2)
        # big must be a superset of small
        assert np.all(big | ~small)


# ── downsample_occupancy_grid ─────────────────────────────────────────────


class TestDownsample:
    def test_output_shape(self):
        grid = np.zeros((20, 20, 20), dtype=bool)
        cg, co, cr = downsample_occupancy_grid(grid, np.zeros(3), 0.1, 0.2)
        assert cg.shape == (10, 10, 10)

    def test_conservative(self):
        """If any fine voxel is occupied, coarse voxel must be occupied."""
        grid = np.zeros((10, 10, 10), dtype=bool)
        grid[0, 0, 0] = True
        cg, _, _ = downsample_occupancy_grid(grid, np.zeros(3), 1.0, 2.0)
        assert cg[0, 0, 0]

    def test_origin_preserved(self):
        origin = np.array([1.0, 2.0, 3.0])
        grid = np.zeros((10, 10, 10), dtype=bool)
        _, co, _ = downsample_occupancy_grid(grid, origin, 0.1, 0.2)
        assert np.allclose(co, origin)
