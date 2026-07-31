import math

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_rotation_only_solver import apply_local_euler_delta
from calibrex.visualization.bullseye import relative_rotation_vector_deg


def test_relative_rotation_vector_recovers_local_axis_angle() -> None:
    reference = SE3(
        (1.0, 2.0, 3.0),
        (0.0, 0.0, math.sin(math.radians(15.0) / 2.0), math.cos(math.radians(15.0) / 2.0)),
    )
    candidate = apply_local_euler_delta(reference, (2.0, 0.0, 0.0))

    vector = relative_rotation_vector_deg(candidate, reference)

    assert math.isclose(vector[0], 2.0, abs_tol=1.0e-9)
    assert math.isclose(vector[1], 0.0, abs_tol=1.0e-9)
    assert math.isclose(vector[2], 0.0, abs_tol=1.0e-9)
