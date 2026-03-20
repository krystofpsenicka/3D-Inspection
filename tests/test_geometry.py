"""Tests for shared/geometry.py — direction_to_quaternion."""
import numpy as np
import pytest
from shared.geometry import direction_to_quaternion
from scipy.spatial.transform import Rotation as R


def _rotate_vector(quat_wxyz, vec):
    """Rotate vec by quaternion [qw, qx, qy, qz]."""
    qw, qx, qy, qz = quat_wxyz
    rot = R.from_quat([qx, qy, qz, qw])  # scipy uses [x,y,z,w]
    return rot.apply(vec)


class TestDirectionToQuaternion:
    def test_plus_x(self):
        q = direction_to_quaternion(np.array([1.0, 0.0, 0.0]))
        assert np.allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-6)

    def test_minus_x(self):
        q = direction_to_quaternion(np.array([-1.0, 0.0, 0.0]))
        assert np.allclose(q, [0.0, 0.0, 1.0, 0.0], atol=1e-6)

    def test_plus_y(self):
        q = direction_to_quaternion(np.array([0.0, 1.0, 0.0]))
        rotated = _rotate_vector(q, [1.0, 0.0, 0.0])
        assert np.allclose(rotated, [0.0, 1.0, 0.0], atol=1e-6)

    def test_plus_z(self):
        q = direction_to_quaternion(np.array([0.0, 0.0, 1.0]))
        rotated = _rotate_vector(q, [1.0, 0.0, 0.0])
        assert np.allclose(rotated, [0.0, 0.0, 1.0], atol=1e-6)

    def test_arbitrary_round_trip(self):
        d = np.array([0.3, -0.7, 0.5])
        d_unit = d / np.linalg.norm(d)
        q = direction_to_quaternion(d_unit)
        rotated = _rotate_vector(q, [1.0, 0.0, 0.0])
        assert np.allclose(rotated, d_unit, atol=1e-6)

    def test_non_unit_normalised(self):
        d = np.array([3.0, 4.0, 0.0])
        q = direction_to_quaternion(d)
        rotated = _rotate_vector(q, [1.0, 0.0, 0.0])
        expected = d / np.linalg.norm(d)
        assert np.allclose(rotated, expected, atol=1e-6)
