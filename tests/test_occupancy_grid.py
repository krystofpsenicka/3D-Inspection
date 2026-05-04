"""Tests for OccupancyGrid, inflate_grid, and downsample_occupancy_grid."""

import cupy as cp
import numpy as np
import pytest

from shared.grid_utils import downsample_occupancy_grid, inflate_grid
from shared.occupancy_grid import OccupancyGrid

# ── Coordinate transforms ──────────────────────────────────────────────────


class TestCoordinateTransforms:
    def test_world_to_voxel_round_trip(self, og):
        world_pt = cp.array([2.25, 3.75, 1.25])
        voxel = og.world_to_voxel(world_pt)
        world_back = og.voxel_to_world(voxel)
        assert cp.allclose(world_pt, world_back, atol=og.resolution)


# ── Free / occupied queries ────────────────────────────────────────────────


class TestFreeOccupied:
    def test_occupied_centre(self, og):
        centre = og.voxel_to_world(cp.array([10, 10, 10]))
        assert not og.is_free_world(centre)

    def test_free_corner(self, og):
        corner = og.voxel_to_world(cp.array([0, 0, 0]))
        assert og.is_free_world(corner)

    def test_num_free_plus_occupied_equals_total(self, og):
        total = int(np.prod(og.grid.shape))
        assert og.num_free + og.num_occupied == total

    def test_is_valid_voxel_in_bounds(self, og):
        assert og.is_valid_voxel(cp.array([0, 0, 0]))
        assert og.is_valid_voxel(cp.array([19, 19, 19]))

    def test_is_valid_voxel_out_of_bounds(self, og):
        # Negative and beyond-shape indices must be rejected
        assert not og.is_valid_voxel(cp.array([-1, 0, 0]))
        assert not og.is_valid_voxel(cp.array([20, 0, 0]))
        assert not og.is_valid_voxel(cp.array([0, 0, 25]))

    def test_is_free_world_returns_false_for_oob(self, og):
        # World point well outside the grid (origin=0, resolution=0.5, shape=20 -> max=10m)
        far_outside = cp.array([100.0, 0.0, 0.0])
        assert not og.is_free_world(far_outside)

    def test_is_free_world_batch_mixed_inside_outside_occupied(self, og):
        # voxel (0,0,0) free, voxel (10,10,10) occupied, far point out of bounds
        free_corner = og.voxel_to_world(cp.array([0, 0, 0]))
        occupied = og.voxel_to_world(cp.array([10, 10, 10]))
        oob = cp.array([100.0, 100.0, 100.0])
        pts = cp.stack([free_corner, occupied, oob])
        result = cp.asnumpy(og.is_free_world_batch(pts))
        assert result.tolist() == [True, False, False]


# ── Flat index ─────────────────────────────────────────────────────────────


class TestFlatIndex:
    def test_round_trip(self, og):
        world_pt = og.voxel_to_world(cp.array([3, 7, 2]))
        flat = og.world_to_flat_index(world_pt)
        world_back = og.flat_index_to_world(flat)
        assert cp.allclose(world_pt, world_back)


# ── Persistence (NPZ round-trip) ──────────────────────────────────────────


class TestSaveLoad:
    def test_npz_round_trip(self, og, tmp_path):
        path = str(tmp_path / "test_og.npz")
        og.save(path)
        # save() strips the extension and writes <stem>.npz + <stem>.json
        assert (tmp_path / "test_og.npz").exists()
        assert (tmp_path / "test_og.json").exists()
        loaded = OccupancyGrid.load(path)
        assert cp.array_equal(loaded.grid, og.grid)
        assert cp.allclose(loaded.origin, og.origin)
        assert loaded.resolution == og.resolution


# ── inflate_grid ──────────────────────────────────────────────────────────


class TestInflateGrid:
    def test_single_point_sphere(self):
        grid = cp.zeros((11, 11, 11), dtype=cp.bool_)
        grid[5, 5, 5] = True
        r = 2
        inflated = inflate_grid(grid, r)
        inflated_np = cp.asnumpy(inflated)
        # Voxels within radius should be occupied
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    dist2 = dx**2 + dy**2 + dz**2
                    if dist2 <= r**2:
                        assert inflated_np[5 + dx, 5 + dy, 5 + dz], (
                            f"({dx},{dy},{dz}) dist2={dist2} should be occupied"
                        )

    def test_zero_inflation_identity(self):
        grid = cp.zeros((5, 5, 5), dtype=cp.bool_)
        grid[2, 2, 2] = True
        result = inflate_grid(grid, 0)
        assert cp.array_equal(result, grid)

    def test_monotonic(self):
        grid = cp.zeros((10, 10, 10), dtype=cp.bool_)
        grid[5, 5, 5] = True
        small = inflate_grid(grid, 1)
        big = inflate_grid(grid, 2)
        # big must be a superset of small
        assert bool(cp.all(big | ~small))


# ── downsample_occupancy_grid ─────────────────────────────────────────────


class TestDownsample:
    def test_conservative(self):
        """If any fine voxel is occupied, coarse voxel must be occupied."""
        grid = cp.zeros((10, 10, 10), dtype=cp.bool_)
        grid[0, 0, 0] = True
        og = OccupancyGrid(grid=grid, origin=cp.zeros(3, dtype=cp.float64), resolution=1.0)
        coarse_og = downsample_occupancy_grid(og, 2.0)
        assert bool(coarse_og.grid[0, 0, 0])

    def test_origin_preserved(self):
        origin = cp.array([1.0, 2.0, 3.0])
        og = OccupancyGrid(
            grid=cp.zeros((10, 10, 10), dtype=cp.bool_), origin=origin, resolution=0.1
        )
        coarse_og = downsample_occupancy_grid(og, 0.2)
        assert cp.allclose(coarse_og.origin, origin)
