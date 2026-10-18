"""Evaluation helpers, imported on first use."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from calibrex.evaluation.metrics import evaluate_quality
    from calibrex.evaluation.registry import (
        MetricDefinition,
        get_metric_definition,
        list_metric_definitions,
        register_metric,
    )
    from calibrex.evaluation.thresholds import MetricThreshold, thresholds_for_profile

_EXPORT_MODULES = {
    "evaluate_quality": "calibrex.evaluation.metrics",
    "MetricDefinition": "calibrex.evaluation.registry",
    "get_metric_definition": "calibrex.evaluation.registry",
    "list_metric_definitions": "calibrex.evaluation.registry",
    "register_metric": "calibrex.evaluation.registry",
    "MetricThreshold": "calibrex.evaluation.thresholds",
    "thresholds_for_profile": "calibrex.evaluation.thresholds",
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
    "MetricDefinition",
    "MetricThreshold",
    "evaluate_quality",
    "get_metric_definition",
    "list_metric_definitions",
    "register_metric",
    "thresholds_for_profile",
]
