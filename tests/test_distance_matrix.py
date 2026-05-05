"""Tests for VRP/core/distance_matrix.py.

We verify the matrix obeys the
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


_SUBPROCESS_TOKENS = ("subprocess", "cuda", "rapids", "cugraph", "cudf")


def _try_compute(og, waypoints) -> cp.ndarray | None:
    """Run compute_distance_matrix; skip cleanly only on subprocess/RAPIDS setup failures.

    Other RuntimeErrors (e.g. real bugs in the production code) propagate so
    the test fails. We only skip when the exception message clearly points
    to the cuGraph subprocess being unavailable.
    """
    try:
        return compute_distance_matrix(og, waypoints)
    except (BrokenPipeError, OSError) as exc:
        # Pipe / fork failures are environmental, not production-code bugs.
        pytest.skip(f"cuGraph subprocess unavailable: {exc}")
    except RuntimeError as exc:
        msg = str(exc).lower()
        if any(tok in msg for tok in _SUBPROCESS_TOKENS):
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
        # Allow 5% slack.
        assert D[0, 1] == pytest.approx(9.0, rel=0.05)

    def test_blocked_corridor_is_unreachable(self):
        """Pin the gap as the only path: with NO gap in the wall, distance
        between waypoints on opposite sides must be infinite (cuGraph SSSP
        encodes unreachable as inf).

        This is the complement of test_matches_known_corridor_distance --
        without it, the production code could route around the wall via
        some unintended path and the corridor test would still pass with
        ~5% slack."""
        grid = cp.zeros((10, 10, 10), dtype=cp.bool_)
        grid[5, :, :] = True  # solid wall, no gap
        sealed = OccupancyGrid(grid=grid, origin=cp.zeros(3), resolution=1.0)
        waypoints = cp.array(
            [[0.5, 5.5, 5.5], [9.5, 5.5, 5.5]],
            dtype=cp.float64,
        )
        D = cp.asnumpy(_try_compute(sealed, waypoints))
        # cuGraph encodes unreachable as float32 max (~3.4e38), not np.inf.
        # Either is fine; just assert the distance is impossibly large for
        # a 10x10x10 grid (whose maximum finite path length is < 50).
        assert D[0, 1] > 1e6, (
            f"Sealed corridor should be unreachable, got D[0,1]={D[0, 1]}"
        )


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
