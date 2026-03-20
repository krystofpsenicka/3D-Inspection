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
    def test_world_to_voxel_round_trip(self, small_og):
        world_pt = np.array([2.25, 3.75, 1.25])
        voxel = small_og.world_to_voxel(world_pt)
        world_back = small_og.voxel_to_world(voxel)
        assert np.allclose(world_pt, world_back, atol=small_og.resolution)

    def test_voxel_to_world_centre(self, small_og):
        ijk = np.array([0, 0, 0])
        w = small_og.voxel_to_world(ijk)
        assert np.allclose(w, small_og.origin + small_og.resolution * 0.5)


# ── Free / occupied queries ────────────────────────────────────────────────


class TestFreeOccupied:
    def test_occupied_centre(self, small_og):
        centre = small_og.voxel_to_world(np.array([10, 10, 10]))
        assert not small_og.is_free_world(centre)

    def test_free_corner(self, small_og):
        corner = small_og.voxel_to_world(np.array([0, 0, 0]))
        assert small_og.is_free_world(corner)

    def test_out_of_bounds_not_free(self, small_og):
        oob = np.array([-10.0, -10.0, -10.0])
        assert not small_og.is_free_world(oob)

    def test_batch_matches_scalar(self, small_og):
        pts = np.array([
            small_og.voxel_to_world(np.array([0, 0, 0])),
            small_og.voxel_to_world(np.array([10, 10, 10])),
            np.array([-10.0, -10.0, -10.0]),
        ])
        batch = small_og.is_free_world_batch(pts)
        for i, pt in enumerate(pts):
            assert batch[i] == small_og.is_free_world(pt), f"Mismatch at {i}"

    def test_num_free_plus_occupied_equals_total(self, small_og):
        total = np.prod(small_og.grid.shape)
        assert small_og.num_free + small_og.num_occupied == total


# ── Flat index ─────────────────────────────────────────────────────────────


class TestFlatIndex:
    def test_round_trip(self, small_og):
        world_pt = small_og.voxel_to_world(np.array([3, 7, 2]))
        flat = small_og.world_to_flat_index(world_pt)
        world_back = small_og.flat_index_to_world(flat)
        assert np.allclose(world_pt, world_back)


# ── Sampling ──────────────────────────────────────────────────────────────


class TestSampling:
    def test_shape(self, small_og, rng):
        pts = small_og.sample_random_free_points(10, rng=rng)
        assert pts.shape == (10, 3)

    def test_all_free(self, small_og, rng):
        pts = small_og.sample_random_free_points(50, rng=rng)
        free = small_og.is_free_world_batch(pts)
        assert np.all(free)

    def test_deterministic(self, small_og):
        a = small_og.sample_random_free_points(5, rng=np.random.RandomState(99))
        b = small_og.sample_random_free_points(5, rng=np.random.RandomState(99))
        assert np.array_equal(a, b)


# ── Persistence (NPZ round-trip) ──────────────────────────────────────────


class TestSaveLoad:
    def test_npz_round_trip(self, small_og, tmp_path):
        path = str(tmp_path / "test_og.pkl")
        small_og.save(path)
        loaded = OccupancyGrid.load(path)
        assert np.array_equal(loaded.grid, small_og.grid)
        assert np.allclose(loaded.origin, small_og.origin)
        assert loaded.resolution == small_og.resolution

    def test_npz_with_raw_grid(self, tmp_path):
        grid = np.zeros((5, 5, 5), dtype=bool)
        raw = np.ones((5, 5, 5), dtype=bool)
        og = OccupancyGrid(
            grid=grid,
            origin=np.array([1.0, 2.0, 3.0]),
            resolution=0.25,
            raw_grid=raw,
            mesh_scale=0.5,
        )
        path = str(tmp_path / "og_raw.pkl")
        og.save(path)
        loaded = OccupancyGrid.load(path)
        assert np.array_equal(loaded.raw_grid, raw)
        assert loaded.mesh_scale == 0.5

    def test_npz_none_raw_grid(self, tmp_path):
        og = OccupancyGrid(
            grid=np.zeros((3, 3, 3), dtype=bool),
            origin=np.zeros(3),
            resolution=1.0,
        )
        path = str(tmp_path / "og_none.pkl")
        og.save(path)
        loaded = OccupancyGrid.load(path)
        assert loaded.raw_grid is None


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

    def test_gpu_vs_cpu(self):
        """If CuPy is available, verify GPU and CPU produce identical results."""
        try:
            import cupy  # noqa: F401
        except ImportError:
            pytest.skip("CuPy not available")
        grid = np.zeros((15, 15, 15), dtype=bool)
        grid[7, 7, 7] = True
        grid[3, 3, 3] = True
        # GPU path (default)
        gpu_result = inflate_grid(grid, 2)
        # CPU path (force scipy)
        from scipy.ndimage import binary_dilation
        r = 2
        coords = np.mgrid[-r:r+1, -r:r+1, -r:r+1]
        se = (coords[0]**2 + coords[1]**2 + coords[2]**2) <= r**2
        cpu_result = binary_dilation(grid, structure=se)
        assert np.array_equal(gpu_result, cpu_result)


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

    def test_factor_one_identity(self):
        grid = np.random.RandomState(42).random((8, 8, 8)) > 0.5
        cg, _, cr = downsample_occupancy_grid(grid, np.zeros(3), 1.0, 1.0)
        assert np.array_equal(cg, grid)
        assert cr == 1.0
