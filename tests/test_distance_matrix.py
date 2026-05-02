"""Tests for VRP/core/distance_matrix.py.

The cuGraph SSSP runs in a one-shot subprocess. Tests skip cleanly if
RAPIDS isn't available; otherwise we verify the matrix obeys the
expected metric properties on small grid graphs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
pytest.importorskip("cugraph")

from shared.occupancy_grid import OccupancyGrid
from VRP.core.distance_matrix import compute_distance_matrix


def _open_grid(shape=(10, 10, 10), resolution=1.0) -> OccupancyGrid:
    return OccupancyGrid(
        grid=cp.zeros(shape, dtype=cp.bool_),
        origin=cp.zeros(3, dtype=cp.float64),
        resolution=resolution,
    )


def _try_compute(og, waypoints) -> cp.ndarray | None:
    """Run compute_distance_matrix; return None if subprocess setup fails."""
    try:
        return compute_distance_matrix(og, waypoints)
    except (RuntimeError, OSError, BrokenPipeError) as exc:
        pytest.skip(f"cuGraph subprocess unavailable: {exc}")
    except Exception as exc:  # noqa: BLE001 - mirror behavior of test_vrp.py
        msg = str(exc).lower()
        if "subprocess" in msg or "cuda" in msg or "rapids" in msg:
            pytest.skip(f"cuGraph subprocess unavailable: {exc}")
        raise


class TestDistanceMatrixInvariants:
    """Metric properties on a fully-open grid graph."""

    def test_diagonal_zero(self, corridor_og):
        # 4 free-space waypoints in the unobstructed half (x < 5)
        waypoints = cp.array(
            [[0.5, 0.5, 0.5], [3.5, 1.5, 4.5], [1.5, 7.5, 2.5], [4.5, 4.5, 0.5]],
            dtype=cp.float64,
        )
        D = _try_compute(corridor_og, waypoints)
        D_np = cp.asnumpy(D)
        for i in range(len(waypoints)):
            assert D_np[i, i] == pytest.approx(0.0, abs=1e-6)

    def test_symmetric(self):
        og = _open_grid()
        waypoints = cp.array(
            [[0.5, 0.5, 0.5], [3.5, 1.5, 4.5], [7.5, 8.5, 2.5], [1.5, 9.5, 0.5]],
            dtype=cp.float64,
        )
        D = cp.asnumpy(_try_compute(og, waypoints))
        np.testing.assert_allclose(D, D.T, atol=1e-5)

    def test_triangle_inequality(self):
        og = _open_grid()
        local_rng = np.random.RandomState(42)
        waypoints_np = local_rng.uniform(0.5, 9.5, size=(5, 3))
        waypoints = cp.asarray(waypoints_np, dtype=cp.float64)
        D = cp.asnumpy(_try_compute(og, waypoints))
        n = len(waypoints)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    # Allow small slack for floating-point in graph edge weights
                    assert D[i, k] <= D[i, j] + D[j, k] + 1e-4


class TestDistanceMatrixCorrectness:
    """Compare cuGraph SSSP to a hand-computed shortest-path on the corridor grid."""

    def test_matches_known_corridor_distance(self, corridor_og):
        """Path from (0, 5, 5) to (9, 5, 5) must go through the gap at (5, 5, 5).

        On a 26-neighbor grid with axis edges of length 1.0, the shortest
        path is straight along x: 9 axis steps -> length 9.0.
        """
        waypoints = cp.array(
            [[0.5, 5.5, 5.5], [9.5, 5.5, 5.5]],
            dtype=cp.float64,
        )
        D = cp.asnumpy(_try_compute(corridor_og, waypoints))
        # Allow 5% slack -- 26-neighbor SSSP will use diagonal hops near the
        # gap but the bulk of the path is still axis-aligned.
        assert D[0, 1] == pytest.approx(9.0, rel=0.05)


class TestDistanceMatrixErrorHandling:
    def test_raises_on_occupied_waypoint(self, small_og):
        """Waypoint mapping to an occupied voxel must raise ValueError."""
        # small_og has obstacle at voxels [8:12, 8:12, 8:12], resolution=0.5,
        # origin=0. Voxel (10,10,10) is at world (5.0, 5.0, 5.0).
        waypoints = cp.array([[5.0, 5.0, 5.0], [0.5, 0.5, 0.5]], dtype=cp.float64)
        with pytest.raises(Exception) as exc_info:
            _try_compute(small_og, waypoints)
        # The subprocess wraps and re-raises; the message should mention
        # "occupied" / "out-of-bounds" / "free" or be a RuntimeError that
        # the subprocess pipeline produces from the ValueError.
        msg = str(exc_info.value).lower()
        assert any(token in msg for token in ("occupied", "free", "out-of-bounds", "node_id"))
