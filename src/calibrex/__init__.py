"""Calibrex public package API, imported on first use."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from calibrex.core.config import CalibrationConfig, load_config
    from calibrex.core.result import CalibrationResult, load_result
    from calibrex.core.solid_state import (
        SolidStateCaptureWindow,
        SolidStateEvaluationConfig,
        SolidStateLidarCalibrationContext,
        SolidStateLidarProfile,
    )

_EXPORT_MODULES = {
    "CalibrationConfig": "calibrex.core.config",
    "load_config": "calibrex.core.config",
    "CalibrationResult": "calibrex.core.result",
    "load_result": "calibrex.core.result",
    "SolidStateCaptureWindow": "calibrex.core.solid_state",
    "SolidStateEvaluationConfig": "calibrex.core.solid_state",
    "SolidStateLidarCalibrationContext": "calibrex.core.solid_state",
    "SolidStateLidarProfile": "calibrex.core.solid_state",
}


def __getattr__(name: str) -> Any:
    module = _EXPORT_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "CalibrationConfig",
    "CalibrationResult",
    "SolidStateCaptureWindow",
    "SolidStateEvaluationConfig",
    "SolidStateLidarCalibrationContext",
    "SolidStateLidarProfile",
    "load_config",
    "load_result",
]

__version__ = "0.5.1"
