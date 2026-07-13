"""Problem graph extension points."""

from calibrex.graph.factors import (
    FactorPlugin,
    get_factor_plugin,
    list_factor_plugins,
    register_factor,
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
    "JointResidualBlock",
    "build_problem",
    "get_factor_plugin",
    "list_factor_plugins",
    "register_factor",
    "split_joint_factors",
]
