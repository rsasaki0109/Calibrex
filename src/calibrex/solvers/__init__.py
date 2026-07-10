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
from calibrex.solvers.park_martin_hand_eye_solver import (
    HandEyeEvaluation,
    HandEyeMotionPair,
    ParkMartinHandEyeOptions,
    ParkMartinHandEyeResult,
    ParkMartinHandEyeSolver,
    evaluate_hand_eye_motions,
)
from calibrex.solvers.planar_board_lidar_camera_solver import (
    OrientedPlane,
    PlanarBoardEvaluation,
    PlanarBoardLidarCameraResult,
    PlanarBoardLidarCameraSolver,
    PlanarBoardObservation,
    PlanarBoardSolverOptions,
    evaluate_planar_board_observations,
)
from calibrex.solvers.planar_board_line_plane_solver import (
    BoardBoundaryCorrespondence,
    LinePlaneBoardObservation,
    LinePlaneEvaluation,
    LinePlaneSolverOptions,
    OrientedLine3D,
    PlanarBoardLinePlaneResult,
    PlanarBoardLinePlaneSolver,
    evaluate_line_plane_observations,
)
from calibrex.solvers.radar_ego_velocity_solver import (
    RadarDopplerObservation,
    RadarEgoVelocityResult,
    RadarEgoVelocitySolver,
    RadarEgoVelocitySolverOptions,
)
from calibrex.solvers.radar_trajectory_yaw_solver import (
    RadarTrajectoryVelocityPair,
    RadarTrajectoryYawResult,
    RadarTrajectoryYawSolver,
    RadarTrajectoryYawSolverOptions,
    RadarYawProbeResult,
    radar_trajectory_yaw_rmse,
)
from calibrex.solvers.robust_point_to_point_icp_solver import (
    IcpIteration,
    IcpPoint,
    RobustPointToPointIcpOptions,
    RobustPointToPointIcpResult,
    RobustPointToPointIcpSolver,
)

__all__ = [
    "NATIVE_LIDAR_POINT_TO_PLANE_BACKEND",
    "BoardBoundaryCorrespondence",
    "FixedTrajectorySe3ExtrinsicSolver",
    "FixedTrajectorySe3SolverOptions",
    "FixedTrajectorySe3SolverResult",
    "HandEyeEvaluation",
    "HandEyeMotionPair",
    "IcpIteration",
    "IcpPoint",
    "KoideLidarCameraSolver",
    "LinePlaneBoardObservation",
    "LinePlaneEvaluation",
    "LinePlaneSolverOptions",
    "NativeLidarPointToPlaneSolver",
    "Open3DSLACSolver",
    "OrientedLine3D",
    "OrientedPlane",
    "ParkMartinHandEyeOptions",
    "ParkMartinHandEyeResult",
    "ParkMartinHandEyeSolver",
    "PlanarBoardEvaluation",
    "PlanarBoardLidarCameraResult",
    "PlanarBoardLidarCameraSolver",
    "PlanarBoardLinePlaneResult",
    "PlanarBoardLinePlaneSolver",
    "PlanarBoardObservation",
    "PlanarBoardSolverOptions",
    "RadarDopplerObservation",
    "RadarEgoVelocityResult",
    "RadarEgoVelocitySolver",
    "RadarEgoVelocitySolverOptions",
    "RadarTrajectoryVelocityPair",
    "RadarTrajectoryYawResult",
    "RadarTrajectoryYawSolver",
    "RadarTrajectoryYawSolverOptions",
    "RadarYawProbeResult",
    "RobustPointToPointIcpOptions",
    "RobustPointToPointIcpResult",
    "RobustPointToPointIcpSolver",
    "SolverAdapter",
    "SolverAdapterResult",
    "evaluate_hand_eye_motions",
    "evaluate_line_plane_observations",
    "evaluate_planar_board_observations",
    "radar_trajectory_yaw_rmse",
]
