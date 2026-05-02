"""Tests for visibility/sampling/utils/sampling_space_builder.py.

Verifies that build_sampling_space filters positions by side and distance
correctly and returns only voxels that satisfy the SDF bounds.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from shared.occupancy_grid import OccupancyGrid
from shared.types import Side
from visibility.sampling.utils.sampling_space_builder import build_sampling_space


def _open_og(shape=(8, 8, 8), resolution=1.0) -> OccupancyGrid:
    return OccupancyGrid(
        grid=cp.zeros(shape, dtype=cp.bool_),
        origin=cp.zeros(3, dtype=cp.float64),
        resolution=resolution,
    )


def _full_og(shape=(8, 8, 8), resolution=1.0) -> OccupancyGrid:
    return OccupancyGrid(
        grid=cp.ones(shape, dtype=cp.bool_),
        origin=cp.zeros(3, dtype=cp.float64),
        resolution=resolution,
    )


def _make_layered_sdf(shape, value_per_x):
    """Build a fine SDF grid where value depends only on the x index.

    value_per_x[i] is broadcast to all (i, *, *).
    """
    sdf = cp.zeros(shape, dtype=cp.float32)
    for i, v in enumerate(value_per_x):
        sdf[i, :, :] = v
    return sdf


def _dummy_targets():
    """Curvature-weighting is off in these tests, so only shape matters."""
    return cp.zeros((1, 3), dtype=cp.float32), cp.zeros((1, 3), dtype=cp.float32)


class TestSamplingSpaceFiltering:
    def test_outside_distance_bounds(self):
        """All returned positions must map to fine SDF values in [min_dist, max_dist]."""
        og = _open_og(shape=(8, 8, 8), resolution=1.0)
        # SDF values depend on the fine x index (0..7).
        sdf_values = list(range(8))  # 0, 1, 2, 3, 4, 5, 6, 7
        sdf = _make_layered_sdf((8, 8, 8), sdf_values)
        targets, normals = _dummy_targets()

        # coarse_res=2.0 -> coarse shape (4, 4, 4). Fine voxel centers are at
        # (1, *, *), (3, *, *), (5, *, *), (7, *, *) -> SDF values 1, 3, 5, 7.
        # Filter [2, 6] should match SDF 3, 5 -> 2 layers x 16 voxels = 32 positions.
        positions, weights, _ = build_sampling_space(
            og,
            sdf,
            targets,
            normals,
            free_space_resolution=2.0,
            side=Side.OUTSIDE,
            min_dist=2.0,
            max_dist=6.0,
        )
        assert len(positions) == 32

        # Verify SDF lookup at every returned position is in range
        # World coord -> fine voxel ijk = floor((x - origin) / resolution)
        positions_np = cp.asnumpy(positions)
        sdf_np = cp.asnumpy(sdf)
        for p in positions_np:
            ijk = np.floor(p / og.resolution).astype(int)
            sdf_value = sdf_np[ijk[0], ijk[1], ijk[2]]
            assert 2.0 <= sdf_value <= 6.0

        # Weights are normalized to a probability distribution
        assert cp.asnumpy(weights).sum() == pytest.approx(1.0)

    def test_inside_distance_bounds(self):
        """INSIDE filters SDF values in [-max_dist, -min_dist]."""
        og = _open_og(shape=(8, 8, 8), resolution=1.0)
        # Negative SDF values for the inside case
        sdf_values = [-(i) for i in range(8)]  # 0, -1, -2, ...
        sdf = _make_layered_sdf((8, 8, 8), sdf_values)
        targets, normals = _dummy_targets()

        # Fine voxel centers along x: (1, 3, 5, 7) -> SDF values -1, -3, -5, -7.
        # Filter min=2, max=6 (INSIDE) -> SDF in [-6, -2] -> -3, -5 -> 2 layers.
        positions, _, _ = build_sampling_space(
            og,
            sdf,
            targets,
            normals,
            free_space_resolution=2.0,
            side=Side.INSIDE,
            min_dist=2.0,
            max_dist=6.0,
        )
        assert len(positions) == 32

    def test_empty_when_no_free_voxels(self):
        """Fully occupied OG -> coarse grid all occupied -> empty result."""
        og = _full_og(shape=(8, 8, 8), resolution=1.0)
        sdf = cp.ones((8, 8, 8), dtype=cp.float32)
        targets, normals = _dummy_targets()

        positions, weights, _ = build_sampling_space(
            og,
            sdf,
            targets,
            normals,
            free_space_resolution=2.0,
            side=Side.OUTSIDE,
            min_dist=0.0,
            max_dist=10.0,
        )
        assert len(positions) == 0
        assert len(weights) == 0

    def test_empty_when_distance_filter_excludes_all(self):
        """No SDF values in the requested range -> empty result."""
        og = _open_og()
        sdf = cp.full((8, 8, 8), 1.0, dtype=cp.float32)
        targets, normals = _dummy_targets()

        positions, _, _ = build_sampling_space(
            og,
            sdf,
            targets,
            normals,
            free_space_resolution=2.0,
            side=Side.OUTSIDE,
            min_dist=10.0,  # No SDF value reaches 10
            max_dist=20.0,
        )
        assert len(positions) == 0
