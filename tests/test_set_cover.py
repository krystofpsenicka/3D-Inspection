"""Tests for visibility/set_cover/.

Strongest property: standard greedy and Minoux 1978 lazy greedy must select
the same viewpoints in the same order on the same input. Both pick by
argmax / heap-pop with ties broken to lowest index, so the sequences must
be bit-identical -- this single test cross-validates two non-trivial
implementations.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from visibility.set_cover.greedy import GreedySetCover
from visibility.set_cover.greedy_cuda import GreedySetCoverCuda
from visibility.set_cover.lazy_greedy import LazyGreedySetCover


def _dummy_pos_rot(n_candidates: int):
    """Greedy/lazy-greedy ignore positions and rotmats algorithmically;
    they only forward them to OptimizationResult. Provide dummies."""
    positions = np.zeros((n_candidates, 3), dtype=np.float32)
    rotmats = np.tile(np.eye(3, dtype=np.float32), (n_candidates, 1, 1))
    return positions, rotmats


class TestGreedyEqualsLazyGreedy:
    """Minoux 1978: lazy greedy must produce the same sequence as standard greedy."""

    def test_identical_selection_on_random_instance(self, random_visibility_matrix):
        n_candidates, n_points = 30, 80
        V = random_visibility_matrix(n_candidates, n_points, density=0.15)
        positions, rotmats = _dummy_pos_rot(n_candidates)

        g = GreedySetCover(n_points, positions, rotmats, V).optimize(
            target_coverage=0.95, max_viewpoints=n_candidates
        )
        # lazy greedy mutates V in-place; pass a copy
        lg = LazyGreedySetCover(n_points, positions, rotmats, V.copy()).optimize(
            target_coverage=0.95, max_viewpoints=n_candidates
        )

        assert g.num_viewpoints == lg.num_viewpoints
        assert g.total_coverage == pytest.approx(lg.total_coverage)
        np.testing.assert_array_equal(g.selected_indices, lg.selected_indices)

    def test_identical_on_dense_instance(self, random_visibility_matrix):
        # Higher density -> fewer iterations to reach 95% coverage; tests a
        # different regime than the sparse case.
        V = random_visibility_matrix(20, 50, density=0.5)
        positions, rotmats = _dummy_pos_rot(20)

        g = GreedySetCover(50, positions, rotmats, V).optimize(target_coverage=0.95)
        lg = LazyGreedySetCover(50, positions, rotmats, V.copy()).optimize(target_coverage=0.95)

        np.testing.assert_array_equal(g.selected_indices, lg.selected_indices)


class TestGreedyCorrectness:
    """Single-step and termination properties on hand-built instances."""

    def test_first_pick_maximizes_gain(self):
        """Construct V where row 7 covers 50 unique points; others ≤ 10. First pick must be 7."""
        n_candidates, n_points = 20, 100
        V = np.zeros((n_candidates, n_points), dtype=np.bool_)
        V[7, :50] = True  # row 7 covers the most
        local_rng = np.random.RandomState(0)
        for i in range(n_candidates):
            if i == 7:
                continue
            indices = local_rng.choice(n_points, size=10, replace=False)
            V[i, indices] = True

        positions, rotmats = _dummy_pos_rot(n_candidates)
        cover = GreedySetCover(n_points, positions, rotmats, V)
        result = cover.select_next()
        assert result is not None
        cover.commit_selection(result[2])
        assert cover.last_selected_index == 7
        assert int(cover.uncovered.sum()) == n_points - 50

    def test_optimize_terminates_at_target_coverage(self):
        """3 carefully chosen rows cover 95% of points -> optimize stops after ≤ 3 picks."""
        n_points = 100
        # Rows 0,1,2 partition the first 95 points; rows 3..9 each cover only 5 points
        # within those first 95 -> rows 0,1,2 are clearly the top 3 in marginal gain.
        V = np.zeros((10, n_points), dtype=np.bool_)
        V[0, 0:32] = True
        V[1, 32:64] = True
        V[2, 64:95] = True
        for i in range(3, 10):
            V[i, (i - 3) * 5 : (i - 3) * 5 + 5] = True

        positions, rotmats = _dummy_pos_rot(10)
        result = LazyGreedySetCover(n_points, positions, rotmats, V).optimize(
            target_coverage=0.95, max_viewpoints=10
        )
        assert result.num_viewpoints <= 3
        assert result.total_coverage >= 0.95 - 1e-9

    def test_optimize_terminates_at_max_viewpoints(self):
        """When target=0.99 is unreachable, num_viewpoints == max_viewpoints."""
        # 50 points, 10 candidates each covering disjoint 4-point chunks:
        # union = 40 / 50 = 80% < 99%. With max_viewpoints=5, we stop early.
        n_points = 50
        V = np.zeros((10, n_points), dtype=np.bool_)
        for i in range(10):
            V[i, i * 4 : i * 4 + 4] = True

        positions, rotmats = _dummy_pos_rot(10)
        result = GreedySetCover(n_points, positions, rotmats, V).optimize(
            target_coverage=0.99, max_viewpoints=5
        )
        assert result.num_viewpoints == 5
        assert result.total_coverage < 0.99

    def test_unreachable_points_do_not_block_termination(self):
        """5 points covered by no row -> optimize halts at the maximum reachable coverage."""
        n_points = 50
        V = np.zeros((8, n_points), dtype=np.bool_)
        # Points 0..4 are covered by no row; rows fully partition the remaining 45
        for i in range(8):
            start = 5 + i * 6
            V[i, start : start + 6] = True

        positions, rotmats = _dummy_pos_rot(8)
        result = LazyGreedySetCover(n_points, positions, rotmats, V).optimize(
            target_coverage=0.99, max_viewpoints=20
        )
        assert result.total_coverage == pytest.approx((n_points - 5) / n_points)
        # Should not have selected the same row twice or kept asking after gain hit 0
        assert result.num_viewpoints <= 8


class TestSetCoverInvariants:
    """Properties that must hold over the full optimize() loop."""

    def test_uncovered_strictly_decreases_per_step(self):
        """While select_next returns a positive-gain pick, uncovered must shrink."""
        V = np.array(
            [
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                [1, 0, 0, 1, 0, 0, 1, 0, 0, 0],  # high overlap with above
            ],
            dtype=np.bool_,
        )
        positions, rotmats = _dummy_pos_rot(4)
        cover = LazyGreedySetCover(10, positions, rotmats, V)

        prev_uncovered = int(cover.uncovered.sum())
        for _ in range(10):
            sel = cover.select_next()
            if sel is None:
                break
            cover.commit_selection(sel[2])
            new_uncovered = int(cover.uncovered.sum())
            assert new_uncovered < prev_uncovered, (
                f"uncovered did not decrease: {prev_uncovered} -> {new_uncovered}"
            )
            prev_uncovered = new_uncovered

    def test_redundancy_at_least_one_when_covered(self, random_visibility_matrix):
        """Every covered point is covered ≥ 1 time, so mean coverage ≥ 1."""
        V = random_visibility_matrix(15, 60, density=0.2)
        positions, rotmats = _dummy_pos_rot(15)
        result = GreedySetCover(60, positions, rotmats, V).optimize(
            target_coverage=0.9, max_viewpoints=15
        )
        if result.num_viewpoints > 0:
            assert result.redundancy >= 1.0


class TestGreedyCpuEqualsCuda:
    """The CUDA greedy implementation must select the same sequence as the
    CPU version on the same input. Both use argmax with first-tie-wins, so
    the sequences must be bit-identical."""

    def test_identical_selection_random_instance(self, random_visibility_matrix):
        n_candidates, n_points = 25, 80
        V_np = random_visibility_matrix(n_candidates, n_points, density=0.2)
        positions, rotmats = _dummy_pos_rot(n_candidates)

        cpu = GreedySetCover(n_points, positions, rotmats, V_np).optimize(
            target_coverage=0.95, max_viewpoints=n_candidates
        )

        # GPU variant takes cupy arrays; visibility_map is uint8 on GPU.
        V_cp = cp.asarray(V_np.astype(np.uint8))
        positions_cp = cp.asarray(positions)
        rotmats_cp = cp.asarray(rotmats)
        gpu = GreedySetCoverCuda(n_points, positions_cp, rotmats_cp, V_cp).optimize(
            target_coverage=0.95, max_viewpoints=n_candidates
        )

        assert cpu.num_viewpoints == gpu.num_viewpoints
        assert cpu.total_coverage == pytest.approx(gpu.total_coverage)
        np.testing.assert_array_equal(cpu.selected_indices, gpu.selected_indices)
