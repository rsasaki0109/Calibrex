"""Problem graph extension points."""

from calibrex.graph.factors import (
    FactorPlugin,
    get_factor_plugin,
    list_factor_plugins,
    register_factor,
)
from calibrex.graph.joint_factors import (
    JointPointToPlaneMeasurement,
    JointRadarDopplerMeasurement,
    make_joint_point_to_plane_factor,
    make_joint_prior_factor,
    make_joint_radar_doppler_factor,
    se3_from_tangent,
)
from calibrex.graph.joint_optimization import (
    BackendNeutralJointOptimizer,
    JointKnownBadProbe,
    JointOptimizerIteration,
    JointOptimizerOptions,
    JointOptimizerResult,
    JointParameterBlock,
    JointResidualBlock,
    split_joint_factors,
)
from calibrex.graph.problem import CalibrationProblemSpec, build_problem

__all__ = [
    "BackendNeutralJointOptimizer",
    "CalibrationProblemSpec",
    "FactorPlugin",
    "JointKnownBadProbe",
    "JointOptimizerIteration",
    "JointOptimizerOptions",
    "JointOptimizerResult",
    "JointParameterBlock",
    "JointPointToPlaneMeasurement",
    "JointRadarDopplerMeasurement",
    "JointResidualBlock",
    "build_problem",
    "get_factor_plugin",
    "list_factor_plugins",
    "make_joint_point_to_plane_factor",
    "make_joint_prior_factor",
    "make_joint_radar_doppler_factor",
    "register_factor",
    "se3_from_tangent",
    "split_joint_factors",
]
