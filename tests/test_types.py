"""Tests for visibility/core/types.py."""
import numpy as np
from visibility.core.types import normalize_vector, FrustumParams, OptimizationResult


class TestNormalizeVector:
    def test_unit_unchanged(self):
        v = np.array([1.0, 0.0, 0.0])
        assert np.allclose(normalize_vector(v), v)

    def test_zero_returns_zeros(self):
        v = np.array([0.0, 0.0, 0.0])
        assert np.allclose(normalize_vector(v), np.zeros(3))

    def test_arbitrary_magnitude(self):
        v = np.array([3.0, 4.0, 0.0])
        n = normalize_vector(v)
        assert abs(np.linalg.norm(n) - 1.0) < 1e-9


class TestFrustumParams:
    def test_fields(self):
        fp = FrustumParams(fov_y=1.0, aspect=1.5, near=0.1, far=10.0)
        assert fp.fov_y == 1.0
        assert fp.aspect == 1.5
        assert fp.near == 0.1
        assert fp.far == 10.0


class TestOptimizationResult:
    def test_construction(self):
        r = OptimizationResult(
            method_name="test",
            viewpoints=[],
            total_coverage=0.95,
            num_viewpoints=10,
            total_time=1.0,
            coverage_per_viewpoint=[],
            redundancy=0.1,
            visibility_computation_time=0.5,
            optimization_time=0.5,
        )
        assert r.total_coverage == 0.95
        assert r.num_viewpoints == 10
