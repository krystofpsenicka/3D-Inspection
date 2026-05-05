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


def _surface_with_varying_curvature():
    """A target cloud where points near x>0 have varying normals (high
    local curvature) while points near x<0 share a single normal (~zero
    curvature)."""
    n_per_side = 50
    rng = np.random.RandomState(42)
    flat_pts = np.stack(
        [
            rng.uniform(-3.0, -0.5, n_per_side),
            rng.uniform(-2.0, 2.0, n_per_side),
            rng.uniform(-2.0, 2.0, n_per_side),
        ],
        axis=1,
    )
    flat_normals = np.tile([1.0, 0.0, 0.0], (n_per_side, 1))

    curved_pts = np.stack(
        [
            rng.uniform(0.5, 3.0, n_per_side),
            rng.uniform(-2.0, 2.0, n_per_side),
            rng.uniform(-2.0, 2.0, n_per_side),
        ],
        axis=1,
    )
    curved_normals = rng.randn(n_per_side, 3)
    curved_normals /= np.linalg.norm(curved_normals, axis=1, keepdims=True)

    targets = cp.asarray(np.vstack([flat_pts, curved_pts]), dtype=cp.float32)
    normals = cp.asarray(np.vstack([flat_normals, curved_normals]), dtype=cp.float32)
    return targets, normals


class TestSamplingSpaceCurvatureWeighting:
    """Branches that the default-args tests skip: curvature_weighting=True
    and explicit position_weight."""

    def _common_kwargs(self):
        og = _open_og(shape=(8, 8, 8), resolution=1.0)
        sdf = _make_layered_sdf((8, 8, 8), [1.0] * 8)  # uniform feasible SDF
        return og, sdf

    def test_position_weight_zero_matches_no_weighting(self):
        """position_weight=0 makes the curvature term vanish: weights must
        equal those from curvature_weighting=False."""
        og, sdf = self._common_kwargs()
        targets, normals = _surface_with_varying_curvature()

        _, w_off, _ = build_sampling_space(
            og, sdf, targets, normals,
            free_space_resolution=2.0, side=Side.OUTSIDE,
            min_dist=0.0, max_dist=10.0,
            curvature_weighting=False,
        )
        _, w_zero, _ = build_sampling_space(
            og, sdf, targets, normals,
            free_space_resolution=2.0, side=Side.OUTSIDE,
            min_dist=0.0, max_dist=10.0,
            curvature_weighting=True,
            position_weight=0.0,
        )
        np.testing.assert_allclose(cp.asnumpy(w_off), cp.asnumpy(w_zero), atol=1e-6)

    def test_curvature_weighting_changes_distribution(self):
        """Non-zero position_weight + non-uniform curvature must shift the
        weight distribution away from uniform (vs. the no-weighting baseline)."""
        og, sdf = self._common_kwargs()
        targets, normals = _surface_with_varying_curvature()

        _, w_off, _ = build_sampling_space(
            og, sdf, targets, normals,
            free_space_resolution=2.0, side=Side.OUTSIDE,
            min_dist=0.0, max_dist=10.0,
            curvature_weighting=False,
        )
        _, w_on, _ = build_sampling_space(
            og, sdf, targets, normals,
            free_space_resolution=2.0, side=Side.OUTSIDE,
            min_dist=0.0, max_dist=10.0,
            curvature_weighting=True,
            position_weight=2.0,
        )

        # Both must still be probability distributions
        assert float(cp.asnumpy(w_on).sum()) == pytest.approx(1.0, abs=1e-5)
        # And they must differ noticeably -- if they're equal, the curvature
        # branch is a silent no-op.
        diff = float(cp.max(cp.abs(w_on - w_off)))
        assert diff > 1e-4, (
            f"curvature weighting did not change weights (max diff {diff:.2e})"
        )
