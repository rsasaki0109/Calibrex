"""Solver adapter interfaces."""

from calibrex.solvers.base import SolverAdapter, SolverAdapterResult
from calibrex.solvers.fixed_trajectory_se3_solver import (
    FixedTrajectorySe3ExtrinsicSolver,
    FixedTrajectorySe3SolverOptions,
    FixedTrajectorySe3SolverResult,
)
from calibrex.solvers.koide_lidar_camera_solver import KoideLidarCameraSolver
from calibrex.solvers.native_lidar_point_to_plane_solver import (
    NATIVE_LIDAR_POINT_TO_PLANE_BACKEND,
    NativeLidarPointToPlaneSolver,
)
from calibrex.solvers.open3d_slac_solver import Open3DSLACSolver

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
