"""Solver adapter interfaces."""

from calibrex.solvers.base import SolverAdapter, SolverAdapterResult
from calibrex.solvers.daniilidis_hand_eye_solver import (
    DaniilidisHandEyeOptions,
    DaniilidisHandEyeResult,
    DaniilidisHandEyeSolver,
)
from calibrex.solvers.fixed_trajectory_se3_solver import (
    FixedTrajectorySe3ExtrinsicSolver,
    FixedTrajectorySe3SolverOptions,
    FixedTrajectorySe3SolverResult,
)
from calibrex.solvers.koide_lidar_camera_solver import KoideLidarCameraSolver
from calibrex.solvers.native_hand_eye_comparison_solver import (
    NATIVE_HAND_EYE_COMPARISON_BACKEND,
    NativeHandEyeComparisonSolver,
)
from calibrex.solvers.native_lidar_point_to_plane_solver import (
    NATIVE_LIDAR_POINT_TO_PLANE_BACKEND,
    NativeLidarPointToPlaneSolver,
)
from calibrex.solvers.native_planar_board_solver import (
    ACFR_VLP_FORMAT,
    ACFR_VLP_SOURCE_URL,
    NATIVE_PLANAR_BOARD_BACKEND,
    NativePlanarBoardSolver,
    read_acfr_vlp_plane_observations,
)
from calibrex.solvers.native_registration_comparison_solver import (
    NATIVE_REGISTRATION_COMPARISON_BACKEND,
    NativeRegistrationComparisonSolver,
    voxel_downsample_icp_points,
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
    PlanarBoardProbeResult,
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
from calibrex.solvers.registration_adapters import (
    NdtSubprocessAdapter,
    NdtSubprocessOptions,
    Open3DGeneralizedIcpAdapter,
    Open3DGeneralizedIcpOptions,
    RegistrationAdapterResult,
)
from calibrex.solvers.robust_point_to_point_icp_solver import (
    IcpCandidateEvaluation,
    IcpIteration,
    IcpMultiStartTrial,
    IcpPoint,
    IcpRematchingDiagnostics,
    RobustPointToPointIcpOptions,
    RobustPointToPointIcpResult,
    RobustPointToPointIcpSolver,
    evaluate_icp_candidate,
    split_icp_source_points,
)
from calibrex.solvers.tsai_lenz_hand_eye_solver import (
    HandEyeProbeResult,
    TsaiLenzHandEyeOptions,
    TsaiLenzHandEyeResult,
    TsaiLenzHandEyeSolver,
    evaluate_hand_eye_known_bad_probes,
)

__all__ = [
    "ACFR_VLP_FORMAT",
    "ACFR_VLP_SOURCE_URL",
    "NATIVE_HAND_EYE_COMPARISON_BACKEND",
    "NATIVE_LIDAR_POINT_TO_PLANE_BACKEND",
    "NATIVE_PLANAR_BOARD_BACKEND",
    "NATIVE_REGISTRATION_COMPARISON_BACKEND",
    "BoardBoundaryCorrespondence",
    "DaniilidisHandEyeOptions",
    "DaniilidisHandEyeResult",
    "DaniilidisHandEyeSolver",
    "FixedTrajectorySe3ExtrinsicSolver",
    "FixedTrajectorySe3SolverOptions",
    "FixedTrajectorySe3SolverResult",
    "HandEyeEvaluation",
    "HandEyeMotionPair",
    "HandEyeProbeResult",
    "IcpCandidateEvaluation",
    "IcpIteration",
    "IcpMultiStartTrial",
    "IcpPoint",
    "IcpRematchingDiagnostics",
    "KoideLidarCameraSolver",
    "LinePlaneBoardObservation",
    "LinePlaneEvaluation",
    "LinePlaneSolverOptions",
    "NativeHandEyeComparisonSolver",
    "NativeLidarPointToPlaneSolver",
    "NativePlanarBoardSolver",
    "NativeRegistrationComparisonSolver",
    "NdtSubprocessAdapter",
    "NdtSubprocessOptions",
    "Open3DGeneralizedIcpAdapter",
    "Open3DGeneralizedIcpOptions",
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
    "PlanarBoardProbeResult",
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
    "RegistrationAdapterResult",
    "RobustPointToPointIcpOptions",
    "RobustPointToPointIcpResult",
    "RobustPointToPointIcpSolver",
    "SolverAdapter",
    "SolverAdapterResult",
    "TsaiLenzHandEyeOptions",
    "TsaiLenzHandEyeResult",
    "TsaiLenzHandEyeSolver",
    "evaluate_hand_eye_known_bad_probes",
    "evaluate_hand_eye_motions",
    "evaluate_icp_candidate",
    "evaluate_line_plane_observations",
    "evaluate_planar_board_observations",
    "radar_trajectory_yaw_rmse",
    "read_acfr_vlp_plane_observations",
    "split_icp_source_points",
    "voxel_downsample_icp_points",
]
