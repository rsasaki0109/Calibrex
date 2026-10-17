"""One native rematching score snapshot, local to a synchronous probe call."""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import ParamSpec, TypeVar

from calibrex.graph.joint_optimization import (
    _BUILTIN_RESIDUALS_METHOD,
    JointResidualBlock,
    ParameterValues,
)

_P = ParamSpec("_P")
_R = TypeVar("_R")


@dataclass
class _ProbeScores:
    state: object = None
    values: dict[str, tuple[float, ...]] = field(default_factory=dict)
    residuals: dict[str, float] = field(default_factory=dict)
    valid: Callable[[], bool] = lambda: False
    captures: int = 0
    hits: int = 0
    population: frozenset[str] = frozenset()


_SCORES: ContextVar[_ProbeScores | None] = ContextVar("joint_probe_scores", default=None)
_REPORTS: ContextVar[tuple[dict[str, int], ...]] = ContextVar("joint_probe_reports", default=())
_MAX_SCORE_ROWS = 65536


def _with_probe_scores(function: Callable[_P, _R]) -> Callable[_P, _R]:
    @wraps(function)
    def evaluate(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        scope = _ProbeScores()
        token = _SCORES.set(scope)
        try:
            return function(*args, **kwargs)
        finally:
            _REPORTS.set((*_REPORTS.get(), {"captures": scope.captures, "hits": scope.hits})[-2:])
            _SCORES.reset(token)

    return evaluate


def _capture_scores(
    state: object, values: ParameterValues, residuals: dict[str, float],
    valid: Callable[[], bool],
) -> None:
    scope = _SCORES.get()
    if scope is None or type(values) is not dict or len(residuals) > _MAX_SCORE_ROWS:
        return
    scope.state = state
    scope.values = dict(values)
    scope.residuals = residuals
    scope.valid = valid
    scope.captures += 1


def _set_probe_population(population: tuple[str, ...]) -> None:
    scope = _SCORES.get()
    if scope is not None and all(type(name) is str for name in population):
        scope.population = frozenset(population)


def _captured_scores(
    state: object, values: ParameterValues, population: tuple[str, ...],
) -> dict[str, float] | None:
    scope = _SCORES.get()
    if (scope is None or state is not scope.state or type(values) is not dict
        or JointResidualBlock.residuals is not _BUILTIN_RESIDUALS_METHOD
        or len(values) != len(scope.values)
        or any(name not in scope.population for name in population)
        or any(values.get(name) is not vector for name, vector in scope.values.items())
        or not scope.valid()):
        return None
    scope.hits += 1
    return scope.residuals
