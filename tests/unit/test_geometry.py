from math import isclose, sqrt

from slac.core.geometry import (
    SE3,
    normalize_quaternion_xyzw,
    quaternion_xyzw_from_rotation_matrix,
)


def test_identity_transform_point() -> None:
    transform = SE3.identity()
    assert transform.transform_point((1.0, 2.0, 3.0)) == (1.0, 2.0, 3.0)


def test_inverse_composition_returns_identity_translation() -> None:
    transform = SE3((1.0, 2.0, 3.0), normalize_quaternion_xyzw((0.0, 0.0, 1.0, 1.0)))
    identity = transform.compose(transform.inverse())
    assert all(isclose(value, 0.0, abs_tol=1e-9) for value in identity.translation_m)
    assert isclose(identity.rotation_quat_xyzw[3], 1.0, abs_tol=1e-9)


def test_rotation_preserves_vector_norm() -> None:
    transform = SE3((0.0, 0.0, 0.0), normalize_quaternion_xyzw((0.0, 0.0, 1.0, 1.0)))
    point = transform.transform_point((3.0, 4.0, 0.0))
    assert isclose(sqrt(sum(value * value for value in point)), 5.0, abs_tol=1e-9)


def test_quaternion_from_identity_rotation_matrix() -> None:
    quaternion = quaternion_xyzw_from_rotation_matrix([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    assert quaternion == (0.0, 0.0, 0.0, 1.0)
