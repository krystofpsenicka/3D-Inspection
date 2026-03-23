"""Tests for shared/grid_utils.py — OFFSETS_26, WEIGHTS_26."""
import math

import numpy as np
import pytest
from shared.grid_utils import OFFSETS_26, WEIGHTS_26


class TestOffsets26:
    def test_count(self):
        assert len(OFFSETS_26) == 26

    def test_no_identity(self):
        assert (0, 0, 0) not in OFFSETS_26


class TestWeights26:
    def test_face_neighbours(self):
        """6 face neighbours (1 non-zero coord) should have weight 1.0."""
        face = [i for i, (di, dj, dk) in enumerate(OFFSETS_26)
                if [abs(di), abs(dj), abs(dk)].count(0) == 2]
        assert len(face) == 6
        for i in face:
            assert abs(WEIGHTS_26[i] - 1.0) < 1e-9

    def test_edge_neighbours(self):
        """12 edge neighbours (2 non-zero coords) should have weight sqrt(2)."""
        edge = [i for i, (di, dj, dk) in enumerate(OFFSETS_26)
                if [abs(di), abs(dj), abs(dk)].count(0) == 1]
        assert len(edge) == 12
        for i in edge:
            assert abs(WEIGHTS_26[i] - math.sqrt(2)) < 1e-9

    def test_corner_neighbours(self):
        """8 corner neighbours (3 non-zero coords) should have weight sqrt(3)."""
        corner = [i for i, (di, dj, dk) in enumerate(OFFSETS_26)
                  if [abs(di), abs(dj), abs(dk)].count(0) == 0]
        assert len(corner) == 8
        for i in corner:
            assert abs(WEIGHTS_26[i] - math.sqrt(3)) < 1e-9
