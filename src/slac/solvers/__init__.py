"""Solver adapter interfaces."""

from slac.solvers.base import SolverAdapter, SolverAdapterResult
from slac.solvers.fixed_trajectory_se3_solver import (
    FixedTrajectorySe3ExtrinsicSolver,
    FixedTrajectorySe3SolverOptions,
    FixedTrajectorySe3SolverResult,
)
from slac.solvers.koide_lidar_camera_solver import KoideLidarCameraSolver
from slac.solvers.native_lidar_point_to_plane_solver import (
    NATIVE_LIDAR_POINT_TO_PLANE_BACKEND,
    NativeLidarPointToPlaneSolver,
)
from slac.solvers.open3d_slac_solver import Open3DSLACSolver

__all__ = [
    "NATIVE_LIDAR_POINT_TO_PLANE_BACKEND",
    "FixedTrajectorySe3ExtrinsicSolver",
    "FixedTrajectorySe3SolverOptions",
    "FixedTrajectorySe3SolverResult",
    "KoideLidarCameraSolver",
    "NativeLidarPointToPlaneSolver",
    "Open3DSLACSolver",
    "SolverAdapter",
    "SolverAdapterResult",
]
