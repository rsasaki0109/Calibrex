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
_PREPARATION_CACHE: ContextVar[OrderedDict[int, object] | None] = ContextVar(
    "joint_preparation_cache", default=None
)
_RGBD_PREPARATION_CACHE: ContextVar[OrderedDict[int, object] | None] = ContextVar(
    "joint_rgbd_preparation_cache", default=None
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


def _with_preparation_cache(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """Keep bounded graph preparation local to one synchronous solver call."""

    @wraps(function)
    def solve(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        token = _PREPARATION_CACHE.set(OrderedDict())
        rgbd_token = _RGBD_PREPARATION_CACHE.set(OrderedDict())
        try:
            return function(*args, **kwargs)
        finally:
            _RGBD_PREPARATION_CACHE.reset(rgbd_token)
            _PREPARATION_CACHE.reset(token)

    return solve


def _discard_preparation_caches() -> None:
    """A scalar fallback can execute user callbacks that invalidate owned data."""
    for cache in (_PREPARATION_CACHE.get(), _RGBD_PREPARATION_CACHE.get()):
        if cache is not None:
            cache.clear()
