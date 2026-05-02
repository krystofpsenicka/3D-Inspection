"""Tests for the factoradic permutation encoding in VRP/mapf/mapf_planner.py.

The two functions are pure -- the only correctness property is that they
form a bijection between [0, n!) and the permutations of range(n).
"""

from __future__ import annotations

import itertools
import math

import pytest

from VRP.mapf.mapf_planner import _index_to_perm, _perm_to_index


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_index_to_perm_to_index_bijection(n):
    """For every idx in [0, n!), perm(idx) round-trips back to idx."""
    for idx in range(math.factorial(n)):
        perm = _index_to_perm(idx, n)
        assert sorted(perm) == list(range(n)), f"not a permutation: {perm}"
        assert _perm_to_index(perm) == idx


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_perm_to_index_to_perm_bijection(n):
    """Every permutation round-trips back through index."""
    for perm in itertools.permutations(range(n)):
        idx = _perm_to_index(list(perm))
        assert _index_to_perm(idx, n) == list(perm)


def test_index_zero_is_identity():
    """Convention: index 0 corresponds to the identity permutation [0, 1, ..., n-1]."""
    for n in [3, 5, 8]:
        assert _index_to_perm(0, n) == list(range(n))


def test_indices_cover_full_range():
    """The mapping from permutations to indices is exactly onto [0, n!)."""
    n = 5
    indices = {_perm_to_index(list(p)) for p in itertools.permutations(range(n))}
    assert indices == set(range(math.factorial(n)))
