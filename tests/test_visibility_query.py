"""Tests for visibility/visibility/ and visibility/core/."""

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
    """Subclass exposing the frustum-culling helper for testing."""

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
        offset_cube = o3d.geometry.TriangleMesh.create_box(0.1, 0.1, 0.1)
        offset_cube.translate((10.0, 10.0, 10.0))
        targets = np.array([[0.0, 0.0, 0.0]])
        viewpoint, rotmat = _down_camera()
        visible, _ = self._query(offset_cube, targets).compute_visibility(viewpoint, rotmat)
        assert 0 in visible

    def test_occluder_blocks_target(self, cube_mesh):
        """A cube between camera and target hides the target."""
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

    def test_estimated_delta_runs_on_density_varying_cloud(self):
        """When epsilon_deg=None, EpsilonVisibilityQuery must estimate
        delta from the input cloud. Verify the estimation path runs and
        produces sensible values on a cloud with sharply varying density.
        """
        # Dense cluster around origin + sparse tail far away
        rng = np.random.RandomState(0)
        dense = rng.uniform(-0.5, 0.5, size=(80, 3))
        dense[:, 2] = 0.0
        sparse = rng.uniform(2.0, 5.0, size=(20, 3))
        sparse[:, 2] = 0.0
        targets = np.vstack([dense, sparse])
        normals = np.tile([0.0, 0.0, 1.0], (len(targets), 1))

        params = FrustumParams(fov_y=np.deg2rad(70.0), aspect=1.0, near=0.1, far=10.0)
        # epsilon_deg=None -> delta must be estimated from the cloud
        q = EpsilonVisibilityQuery(targets, normals, params, epsilon_deg=None)
        # Estimated delta must be finite and positive
        assert q.delta is not None
        assert 0.0 < float(q.delta) < 10.0, f"delta={q.delta} outside plausible range"

        # And the query must still produce visibility on the dense cluster.
        viewpoint, rotmat = _down_camera(viewpoint=(0.0, 0.0, 3.0))
        visible, _ = q.compute_visibility(viewpoint, rotmat)
        # At least the dense, on-axis points must be visible.
        assert len(visible) > 0


# ── CPU/CUDA equivalence ─────────────────────────

class TestRaycastCpuVsCuda:
    """The CUDA raycaster (Triro/OptiX) must agree with the CPU raycaster
    (Open3D) on a small scene."""

    def test_visibility_matches_on_cube_scene(self, cube_mesh):
        triro = pytest.importorskip("triro")  # noqa: F841
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from visibility.visibility.raycast_cuda import RaycastingVisibilityQueryCuda

        # 4 targets: 2 unobstructed (offset cubes), 2 behind the cube_mesh
        targets = np.array(
            [
                [0.0, 0.0, -2.0],   # behind cube (occluded from above)
                [3.0, 0.0, -2.0],   # offset, no occlusion
                [0.0, 3.0, -2.0],   # offset, no occlusion
                [0.0, 0.0, -3.0],   # also behind cube, deeper
            ]
        )
        normals = np.tile([0.0, 0.0, 1.0], (len(targets), 1))
        params = FrustumParams(fov_y=np.deg2rad(80.0), aspect=1.0, near=0.1, far=10.0)
        viewpoint, rotmat = _down_camera(viewpoint=(0.0, 0.0, 3.0))

        cpu = RaycastingVisibilityQuery(cube_mesh, targets, normals, params)
        gpu = RaycastingVisibilityQueryCuda(
            cube_mesh, cp.asarray(targets, dtype=cp.float32),
            cp.asarray(normals, dtype=cp.float32), params,
        )

        cpu_vis, _ = cpu.compute_visibility(viewpoint, rotmat)
        gpu_vis, _ = gpu.compute_visibility(
            cp.asarray(viewpoint, dtype=cp.float32),
            cp.asarray(rotmat, dtype=cp.float32),
        )
        assert set(cpu_vis.tolist()) == set(cp.asnumpy(gpu_vis).tolist())
