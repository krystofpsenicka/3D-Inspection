"""Tests for VRP/mapf/path_smoother.py:simplify_path_ompl.

OMPL is a black box; only one strict property is worth asserting -- the
smoothed path remains collision-free against the OG it was validated
against. If this fails, the entire MAPF pipeline is unsound.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
pytest.importorskip("ompl")

from VRP.mapf.path_smoother import simplify_path_ompl


class TestSimplifyPathOmpl:
    def test_smoothed_path_collision_free(self, corridor_og):
        """A coarse zig-zag path through the gap, smoothed, must stay collision-free.

        Coarse path: (0.5, 5.5, 5.5) -> (4.5, 5.5, 5.5) -> (5.5, 5.5, 5.5)
        -> (9.5, 5.5, 5.5). The straight line from start to end clips through
        the wall, so the smoother must keep the path through the gap.
        """
        coarse = cp.array(
            [
                [0.5, 5.5, 5.5],
                [4.5, 5.5, 5.5],
                [5.5, 5.5, 5.5],  # gap voxel
                [9.5, 5.5, 5.5],
            ],
            dtype=cp.float64,
        )
        smoothed = simplify_path_ompl(coarse, corridor_og, robot_radius=0.1, max_time=0.5)

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
        free_mask = corridor_og.is_free_world_batch(samples_gpu)
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
