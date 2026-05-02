"""Tests for visibility/visibility/ and visibility/core/.

Covers normalize_vector, compute_redundancy, frustum bounding-sphere geometry,
KDTree frustum culling, and the two visibility query implementations
(RaycastingVisibilityQuery, EpsilonVisibilityQuery).
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
o3d = pytest.importorskip("open3d")

from visibility.core.types import FrustumParams, normalize_vector
from visibility.core.utils import compute_redundancy
from visibility.visibility.base import VisibilityQuery, get_frustum_bounding_sphere
from visibility.visibility.epsilon import EpsilonVisibilityQuery
from visibility.visibility.raycast import RaycastingVisibilityQuery


# ── normalize_vector ────────────────────────────────────────────────────────


class TestNormalizeVector:
    def test_unit_length_for_random_vectors(self):
        local_rng = np.random.RandomState(42)
        for _ in range(50):
            v = local_rng.randn(3) * local_rng.uniform(0.01, 100.0)
            n = normalize_vector(v)
            assert np.linalg.norm(n) == pytest.approx(1.0, abs=1e-7)

    def test_zero_returns_zero(self):
        assert np.allclose(normalize_vector(np.zeros(3)), np.zeros(3))

    def test_below_eps_returns_zero(self):
        # NORM_EPS = 1e-12; a vector with norm 1e-15 should hit the guard
        v = np.array([1e-16, 0.0, 0.0])
        assert np.allclose(normalize_vector(v), np.zeros(3))


# ── compute_redundancy ──────────────────────────────────────────────────────


class TestComputeRedundancy:
    def test_identity_matrix_redundancy_one(self):
        """K=M, each viewpoint covers exactly one unique point -> redundancy = 1."""
        V = cp.eye(20, dtype=cp.uint8)
        assert compute_redundancy(V) == pytest.approx(1.0)

    def test_full_coverage_redundancy_k(self):
        """All-ones (K, M) -> every point is covered K times."""
        V = cp.ones((5, 30), dtype=cp.uint8)
        assert compute_redundancy(V) == pytest.approx(5.0)

    def test_uncovered_points_excluded_from_mean(self):
        """Half the points get 0 coverage, half get 2x. Mean over covered = 2."""
        V = cp.zeros((2, 20), dtype=cp.uint8)
        V[0, :10] = 1  # first 10 points
        V[1, :10] = 1  # same first 10 points (covered twice)
        # Last 10 points covered zero times -- must NOT be included in the mean
        assert compute_redundancy(V) == pytest.approx(2.0)

    def test_empty_returns_zero(self):
        V = cp.empty((0, 50), dtype=cp.uint8)
        assert compute_redundancy(V) == 0.0


# ── Frustum bounding sphere ─────────────────────────────────────────────────


class TestFrustumBoundingSphere:
    def test_sphere_contains_all_eight_corners(self):
        """The defining property of a bounding sphere: all 8 frustum corners
        must lie within the sphere (radius + small epsilon)."""
        params = FrustumParams(fov_y=np.deg2rad(60.0), aspect=1.6, near=0.5, far=10.0)
        local_rng = np.random.RandomState(7)

        for _ in range(10):
            viewpoint = local_rng.randn(3) * 5.0
            forward = normalize_vector(local_rng.randn(3))
            up_world = np.array([0.0, 0.0, 1.0])
            right = normalize_vector(np.cross(forward, up_world))
            if np.linalg.norm(right) < 1e-6:
                continue
            up = normalize_vector(np.cross(right, forward))

            center, radius = get_frustum_bounding_sphere(viewpoint, forward, params)

            tan_half_fov = np.tan(params.fov_y / 2.0)
            for depth in (params.near, params.far):
                half_h = depth * tan_half_fov
                half_w = half_h * params.aspect
                for dh in (-half_h, half_h):
                    for dw in (-half_w, half_w):
                        corner = viewpoint + forward * depth + right * dw + up * dh
                        assert np.linalg.norm(corner - center) <= radius + 1e-6


# ── KDTree-based frustum culling ────────────────────────────────────────────


def _down_camera(viewpoint=(0.0, 0.0, 2.0)):
    """Camera looking straight down (-z). Returns (viewpoint, rotmat)."""
    forward = np.array([0.0, 0.0, -1.0])
    right = np.array([1.0, 0.0, 0.0])
    up = np.array([0.0, 1.0, 0.0])
    rotmat = np.column_stack([forward, right, up])
    return np.asarray(viewpoint, dtype=np.float64), rotmat


class _CullOnlyQuery(VisibilityQuery):
    """Subclass exposing the protected frustum-culling helper for testing."""

    def compute_visibility(self, viewpoint, rotmat):  # pragma: no cover -- not used here
        return self.points_in_frustum_with_kdtree(viewpoint, rotmat), 0.0


class TestFrustumCullingKDTree:
    """Frustum culling at axis-aligned positions where the answer is obvious."""

    def _query(self, target_points, normals=None):
        if normals is None:
            normals = np.tile(np.array([0.0, 0.0, 1.0]), (len(target_points), 1))
        params = FrustumParams(fov_y=np.deg2rad(60.0), aspect=1.0, near=0.1, far=5.0)
        return _CullOnlyQuery(target_points, normals, params)

    def test_includes_axis_point(self):
        targets = np.array([[0.0, 0.0, 0.0]])
        viewpoint, rotmat = _down_camera()
        idx = self._query(targets).points_in_frustum_with_kdtree(viewpoint, rotmat)
        assert 0 in idx

    def test_excludes_behind_camera(self):
        targets = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]])  # second is behind camera
        viewpoint, rotmat = _down_camera()
        idx = self._query(targets).points_in_frustum_with_kdtree(viewpoint, rotmat)
        assert 1 not in idx

    def test_excludes_beyond_far(self):
        targets = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -100.0]])  # second beyond far
        viewpoint, rotmat = _down_camera()
        idx = self._query(targets).points_in_frustum_with_kdtree(viewpoint, rotmat)
        assert 1 not in idx

    def test_excludes_outside_fov(self):
        targets = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])  # second off to the side
        viewpoint, rotmat = _down_camera()
        idx = self._query(targets).points_in_frustum_with_kdtree(viewpoint, rotmat)
        assert 1 not in idx


# ── Raycasting visibility ───────────────────────────────────────────────────


class TestRaycastVisibility:
    def _query(self, mesh, target_points, normals=None):
        if normals is None:
            normals = np.tile(np.array([0.0, 0.0, 1.0]), (len(target_points), 1))
        params = FrustumParams(fov_y=np.deg2rad(60.0), aspect=1.0, near=0.1, far=5.0)
        return RaycastingVisibilityQuery(mesh, target_points, normals, params)

    def test_unobstructed_target_visible(self):
        """A target on a clear line of sight to the camera is visible."""
        # A tiny cube offset far away from the line of sight -- the scene
        # has triangles (RaycastingScene needs them) but they are not in
        # the way of the ray (camera at z=2, target at z=0, cube at x=10).
        offset_cube = o3d.geometry.TriangleMesh.create_box(0.1, 0.1, 0.1)
        offset_cube.translate((10.0, 10.0, 10.0))
        targets = np.array([[0.0, 0.0, 0.0]])
        viewpoint, rotmat = _down_camera()
        visible, _ = self._query(offset_cube, targets).compute_visibility(viewpoint, rotmat)
        assert 0 in visible

    def test_occluder_blocks_target(self, cube_mesh):
        """A cube between camera and target hides the target."""
        # Camera high above origin looks down at a target at z=-2 (below the cube).
        # Cube spans [-0.5, 0.5]^3 -- it sits between camera (z=2) and target (z=-2).
        targets = np.array([[0.0, 0.0, -2.0]])
        params = FrustumParams(fov_y=np.deg2rad(60.0), aspect=1.0, near=0.1, far=10.0)
        normals = np.array([[0.0, 0.0, 1.0]])
        q = RaycastingVisibilityQuery(cube_mesh, targets, normals, params)

        viewpoint, rotmat = _down_camera()
        visible, _ = q.compute_visibility(viewpoint, rotmat)
        assert 0 not in visible


# ── Epsilon visibility ──────────────────────────────────────────────────────


class TestEpsilonVisibility:
    def _query(self, target_points, normals):
        params = FrustumParams(fov_y=np.deg2rad(60.0), aspect=1.0, near=0.1, far=5.0)
        # Use a fixed, generous epsilon so the test does not depend on
        # delta-estimation noise on tiny clouds.
        return EpsilonVisibilityQuery(target_points, normals, params, epsilon_deg=2.0)

    def test_back_face_culled(self):
        """A point with normal pointing AWAY from the camera must not be visible."""
        # Camera at (0,0,2) looks down (-z). Normal points down (-z) too -> back face.
        targets = np.array([[0.0, 0.0, 0.0]])
        normals = np.array([[0.0, 0.0, -1.0]])
        viewpoint, rotmat = _down_camera()
        visible, _ = self._query(targets, normals).compute_visibility(viewpoint, rotmat)
        assert 0 not in visible

    def test_front_face_visible_no_occluders(self, flat_target_cloud):
        """A flat patch with upward normals viewed from above -> all points front-facing
        and unoccluded -> visible."""
        targets, normals = flat_target_cloud
        viewpoint, rotmat = _down_camera()
        visible, _ = self._query(targets, normals).compute_visibility(viewpoint, rotmat)
        # Every point in the small patch under the camera should be reported visible
        assert len(visible) == len(targets)
