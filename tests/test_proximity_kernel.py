"""Tests for visibility/sampling/utils/proximity.py.

The CUDA kernel computes per-query weight = sum(exp(-d * inv_sigma)) over
the K nearest targets. We verify the easy cases where the answer is
analytically known.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from visibility.sampling.utils.proximity import compute_knn_proximity_weights


class TestKnnProximityWeights:
    def test_zero_distance_max_weight(self):
        """Query points coincide with target points -> weight = K * exp(0) = K."""
        # 5 query points equal to 5 target points. With k=3, weight should be
        # 3 (the 3 nearest are the query itself + 2 closest neighbours, but
        # those two contribute exp(-d) < 1; total < 3 unless every point is
        # at the same location). Build a degenerate case where ALL targets
        # are at the same point as the query.
        queries = cp.array([[0.0, 0.0, 0.0]], dtype=cp.float32)
        targets = cp.zeros((10, 3), dtype=cp.float32)  # all at origin
        k = 5
        weights = compute_knn_proximity_weights(queries, targets, k=k, sigma=1.0)
        assert float(weights[0]) == pytest.approx(float(k), abs=1e-3)

    def test_far_targets_negligible_weight(self):
        """Targets very far from query -> weight ~ 0 for any reasonable sigma."""
        queries = cp.array([[1000.0, 0.0, 0.0]], dtype=cp.float32)
        targets = cp.random.uniform(-1.0, 1.0, (50, 3)).astype(cp.float32)
        weights = compute_knn_proximity_weights(queries, targets, k=10, sigma=1.0)
        # exp(-~999 * 1) is far below float32 precision -> exactly 0
        assert float(weights[0]) < 1e-30

    def test_monotonic_in_inv_sigma(self):
        """Smaller sigma (larger inv_sigma) -> sharper falloff -> smaller weights
        for non-zero distance neighbours."""
        # 1 query, 20 targets in a small cluster offset from the query
        queries = cp.array([[5.0, 0.0, 0.0]], dtype=cp.float32)
        local_rng = np.random.RandomState(0)
        targets_np = local_rng.uniform(-0.5, 0.5, (20, 3)).astype(np.float32)
        targets = cp.asarray(targets_np)

        loose = compute_knn_proximity_weights(queries, targets, k=5, sigma=10.0)  # gentle decay
        tight = compute_knn_proximity_weights(queries, targets, k=5, sigma=0.5)  # sharp decay

        assert float(loose[0]) > float(tight[0])

    def test_known_two_neighbour_weight(self):
        """Hand-computed weight for a tiny instance.

        Query at (0,0,0), targets at (1,0,0) and (10,0,0). With k=2 and
        sigma=1.0: weight = exp(-1) + exp(-10).
        """
        queries = cp.array([[0.0, 0.0, 0.0]], dtype=cp.float32)
        targets = cp.array([[1.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=cp.float32)
        weights = compute_knn_proximity_weights(queries, targets, k=2, sigma=1.0)
        expected = math.exp(-1.0) + math.exp(-10.0)
        assert float(weights[0]) == pytest.approx(expected, abs=1e-5)
