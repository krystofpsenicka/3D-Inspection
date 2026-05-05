"""Tests for visibility/sampling/  --  focused on non-obvious math and algorithmic correctness."""

import cupy as cp
import numpy as np
import pytest

from visibility.sampling.utils.curvature import compute_local_curvature
from visibility.sampling.utils.direction import apply_angular_noise, knn_centroid_direction


class TestComputeLocalCurvature:
    """KNN + arccos + mean angular deviation  --  non-trivial math."""

    def test_flat_surface_zero_curvature(self):
        """100 points on a plane with uniform normals -> curvature ~ 0."""
        pts = cp.random.uniform(-5, 5, (100, 3)).astype(cp.float32)
        pts[:, 2] = 0.0  # flatten to z=0
        normals = cp.zeros((100, 3), dtype=cp.float32)
        normals[:, 2] = 1.0  # all point up

        curv = compute_local_curvature(pts, pts, normals, k=10)
        assert float(curv.max()) < 0.05, (
            f"Flat surface curvature should be ~0, got max={float(curv.max())}"
        )

    def test_curved_surface_nonzero_curvature(self):
        """Points on a hemisphere with radial normals -> curvature >> 0."""
        n = 200
        # Random points on upper unit hemisphere
        phi = cp.random.uniform(0, 2 * cp.pi, n)
        theta = cp.random.uniform(0, cp.pi / 2, n)
        x = cp.cos(phi) * cp.sin(theta)
        y = cp.sin(phi) * cp.sin(theta)
        z = cp.cos(theta)
        pts = cp.stack([x, y, z], axis=1).astype(cp.float32)
        # Radial normals (pointing outward from origin)
        norms = cp.linalg.norm(pts, axis=1, keepdims=True)
        normals = (pts / cp.maximum(norms, 1e-8)).astype(cp.float32)

        curv = compute_local_curvature(pts, pts, normals, k=10)
        mean_curv = float(curv.mean())
        assert mean_curv > 0.1, f"Hemisphere curvature should be significant, got mean={mean_curv}"


class TestApplyAngularNoise:
    """Rodrigues rotation"""

    def test_angle_bounded(self):
        """All output angles <= max_angle from input directions."""
        n = 200
        max_angle = 0.3
        dirs = cp.random.randn(n, 3).astype(cp.float32)
        dirs /= cp.linalg.norm(dirs, axis=1, keepdims=True)

        rotated = apply_angular_noise(dirs, max_angle)

        # Angle between original and rotated
        cos_sim = cp.clip(cp.sum(dirs * rotated, axis=1), -1.0, 1.0)
        angles = cp.arccos(cos_sim)
        max_observed = float(angles.max())
        assert max_observed <= max_angle + 1e-5, (
            f"Max observed angle {max_observed:.4f} exceeds max_angle {max_angle}"
        )

    def test_zero_noise_preserves_input(self):
        """max_angle=0 -> output identical to input."""
        dirs = cp.random.randn(50, 3).astype(cp.float32)
        dirs /= cp.linalg.norm(dirs, axis=1, keepdims=True)

        rotated = apply_angular_noise(dirs, 0.0)
        # float32 unit vectors: 1e-7 is below machine eps after Rodrigues ops
        assert cp.allclose(dirs, rotated, atol=1e-6)


class TestKnnCentroidDirection:
    """Centroid-minus-query direction logic."""

    def test_direction_toward_targets(self):
        """Query at (10,0,0), targets around origin -> direction approx. (-1,0,0)."""
        query = cp.array([[10.0, 0.0, 0.0]], dtype=cp.float32)
        # Target points
        rng = np.random.RandomState(42)
        targets = np.vstack(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, -1.0, 0.0],
                ],
                rng.randn(10, 3).astype(np.float32) * 0.5,  # small cluster near origin
            ]
        )
        targets = cp.asarray(targets, dtype=cp.float32)

        dirs = knn_centroid_direction(query, targets, k=5)
        d = cp.asnumpy(dirs[0])
        # Should point in -x
        assert d[0] < -0.9, f"Expected direction toward -x, got {d}"
        assert abs(d[1]) < 0.1
        assert abs(d[2]) < 0.1
