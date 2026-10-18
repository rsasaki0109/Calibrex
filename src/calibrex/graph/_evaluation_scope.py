"""Bounded, context-local scratch values for one joint graph evaluation."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from contextvars import ContextVar
from functools import wraps
from typing import ParamSpec, TypeVar

_EVALUATION_CACHE_LIMIT = 128
_EVALUATION_CACHE: ContextVar[OrderedDict[bytes, object] | None] = ContextVar(
    "joint_evaluation_cache", default=None
)
_P = ParamSpec("_P")
_R = TypeVar("_R")


def _with_evaluation_cache(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """Release scratch values on return/error and isolate nested evaluations."""

    @wraps(function)
    def evaluate(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        token = _EVALUATION_CACHE.set(OrderedDict())
        try:
            return function(*args, **kwargs)
        finally:
            _EVALUATION_CACHE.reset(token)

    return evaluate
