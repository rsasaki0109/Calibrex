"""Problem graph extension points."""

from calibrex.graph.factors import (
    FactorPlugin,
    get_factor_plugin,
    list_factor_plugins,
    register_factor,
)
from calibrex.graph.problem import CalibrationProblemSpec, build_problem

__all__ = [
    "CalibrationProblemSpec",
    "FactorPlugin",
    "build_problem",
    "get_factor_plugin",
    "list_factor_plugins",
    "register_factor",
]
