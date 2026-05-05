"""Tests for VRP/mapf/path_smoother.py:simplify_path_ompl.

OMPL is a black box; only one strict property is worth asserting -- the
smoothed path remains collision-free against the OG it was validated
against.
"""

from __future__ import annotations

import cupy as cp
import numpy as np

from shared.grid_utils import inflate_grid
from shared.occupancy_grid import OccupancyGrid
from VRP.mapf.path_smoother import simplify_path_ompl


class TestSimplifyPathOmpl:
    def test_smoothed_path_collision_free(self, corridor_og):
        """A coarse zig-zag path through an inflated corridor, smoothed,
        must stay collision-free against the inflated grid.

        Production callers pre-inflate the occupancy grid by
        ``robot_radius`` so the smoother's point-in-voxel validity check
        is sound for a robot of that radius; this test mirrors that.
        The corridor's 1-voxel gap is widened to 3x3 first so it
        survives a 1-voxel inflation as a 1-voxel passage.
        """
        robot_radius = 0.1
        resolution = float(corridor_og.resolution)
        inflation_voxels = int(np.ceil(robot_radius / resolution))

        grid = corridor_og.grid.copy()
        grid[5, 4:7, 4:7] = False  # widen the gap so it survives inflation
        og = OccupancyGrid(
            grid=inflate_grid(grid, inflation_voxels),
            origin=corridor_og.origin,
            resolution=resolution,
        )

        coarse = cp.array(
            [
                [0.5, 5.5, 5.5],
                [4.5, 5.5, 5.5],
                [5.5, 5.5, 5.5],  # gap voxel
                [9.5, 5.5, 5.5],
            ],
            dtype=cp.float64,
        )
        smoothed = simplify_path_ompl(coarse, og, robot_radius=robot_radius, max_time=0.5)

        # Sample densely along the smoothed polyline and check every sample is free
        smoothed_np = cp.asnumpy(smoothed)
        assert len(smoothed_np) >= 2
        samples = []
        for i in range(len(smoothed_np) - 1):
            p0, p1 = smoothed_np[i], smoothed_np[i + 1]
            n = max(20, int(np.linalg.norm(p1 - p0) / 0.1))
            for t in np.linspace(0.0, 1.0, n):
                samples.append(p0 + t * (p1 - p0))
        samples_gpu = cp.asarray(np.stack(samples))
        free_mask = og.is_free_world_batch(samples_gpu)
        n_collisions = int(cp.sum(~free_mask))
        assert n_collisions == 0, (
            f"smoothed path has {n_collisions}/{len(samples)} samples inside the wall"
        )

    def test_smoothed_path_endpoints_unchanged(self, corridor_og):
        """Smoothing preserves the path's start and end."""
        coarse = cp.array(
            [
                [0.5, 5.5, 5.5],
                [4.5, 5.5, 5.5],
                [5.5, 5.5, 5.5],
                [9.5, 5.5, 5.5],
            ],
            dtype=cp.float64,
        )
        smoothed = cp.asnumpy(
            simplify_path_ompl(coarse, corridor_og, robot_radius=0.1, max_time=0.5)
        )
        np.testing.assert_allclose(smoothed[0], cp.asnumpy(coarse[0]), atol=1e-3)
        np.testing.assert_allclose(smoothed[-1], cp.asnumpy(coarse[-1]), atol=1e-3)

    def test_smoothed_path_not_longer_than_input(self, corridor_og):
        """OMPL shortcutPath + smoothBSpline must not lengthen the path."""
        coarse = cp.array(
            [
                [0.5, 5.5, 5.5],
                [4.5, 5.5, 5.5],
                [5.5, 5.5, 5.5],
                [9.5, 5.5, 5.5],
            ],
            dtype=cp.float64,
        )
        coarse_np = cp.asnumpy(coarse)
        coarse_len = float(np.sum(np.linalg.norm(np.diff(coarse_np, axis=0), axis=1)))

        smoothed = cp.asnumpy(
            simplify_path_ompl(coarse, corridor_og, robot_radius=0.1, max_time=0.5)
        )
        smoothed_len = float(np.sum(np.linalg.norm(np.diff(smoothed, axis=0), axis=1)))
        # Allow tiny slack for B-spline rounding around the gap
        assert smoothed_len <= coarse_len + 1e-3, (
            f"smoothed length {smoothed_len:.4f} exceeds coarse length {coarse_len:.4f}"
        )
