"""Bounded, caller-owned reuse of candidate-independent image and depth edges."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from typing import Any, Literal, TypeAlias, TypeVar, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from calibrex.core.result import StrictModel

_Value = TypeVar("_Value")
_Key: TypeAlias = tuple[str, str, tuple[int, ...], tuple[int, ...], bytes, str]


class EdgePreparationCacheDiagnostics(StrictModel):
    """Cumulative reuse counters; retained bytes include an entry metadata charge."""

    schema_version: Literal["calibrex.edge_preparation_cache_diagnostics/v0.1"] = (
        "calibrex.edge_preparation_cache_diagnostics/v0.1"
    )
    max_bytes: int = Field(ge=0)
    retained_bytes: int = Field(ge=0)
    entries: int = Field(ge=0)
    hits: int = Field(ge=0)
    misses: int = Field(ge=0)
    evictions: int = Field(ge=0)
    bypasses: int = Field(ge=0)


@dataclass(frozen=True)
class _Entry:
    value: Any
    charge: int


def _array_bytes(value: object) -> int:
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, tuple):
        return sum(_array_bytes(item) for item in value)
    return 0


class EdgePreparationCache:
    """Reuse pure preprocessing across checks of different candidate transforms.

    Keys bind every input array byte, shape, dtype, layout and preprocessing setting.
    Candidate-dependent visibility, repeated-view selection, search and evidence
    are recomputed. Returned arrays are independent writable copies, so changing
    a prepared frame cannot corrupt a later check. Do not mutate input arrays
    concurrently with preparation.

    The LRU retains at most ``max_bytes`` of arrays plus a conservative metadata
    charge. Caller arrays, returned copies and computations in flight are outside
    that budget. Zero disables reuse. There is no disk or process-global cache.
    """

    def __init__(self, max_bytes: int = 128 * 1024 * 1024) -> None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
            raise ValueError("edge preparation cache bytes must be a non-negative integer")
        self._max_bytes = max_bytes
        self._entries: OrderedDict[_Key, _Entry] = OrderedDict()
        self._retained_bytes = 0
        self._hits = self._misses = self._evictions = self._bypasses = 0
        self._epoch = 0
        self._lock = Lock()

    @property
    def diagnostics(self) -> EdgePreparationCacheDiagnostics:
        """A validated snapshot of the counters, without exposing retained arrays."""
        with self._lock:
            return EdgePreparationCacheDiagnostics(
                max_bytes=self._max_bytes,
                retained_bytes=self._retained_bytes,
                entries=len(self._entries),
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                bypasses=self._bypasses,
            )

    def clear(self) -> None:
        """Release entries; computations already in flight cannot repopulate them.

        Diagnostic counters remain cumulative over this instance's lifetime.
        """
        with self._lock:
            self._entries.clear()
            self._retained_bytes = 0
            self._epoch += 1

    def _get_or_compute(
        self,
        namespace: str,
        array: NDArray[Any],
        settings: tuple[object, ...],
        compute: Callable[[], _Value],
    ) -> _Value:
        with self._lock:
            epoch = self._epoch
        if self._max_bytes == 0 or array.dtype.hasobject:
            with self._lock:
                self._bypasses += 1
            return compute()
        try:
            encoded = json.dumps(settings, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError):
            # Unusual settings retain the original function's behavior/errors.
            with self._lock:
                self._bypasses += 1
            return compute()
        content = np.ascontiguousarray(array)
        digest = hashlib.sha256(content).digest()
        key = namespace, content.dtype.str, content.shape, array.strides, digest, encoded
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                self._hits += 1
            else:
                self._misses += 1
        if entry is not None:
            return cast(_Value, copy.deepcopy(entry.value))
        value = compute()  # Exceptions are never cached or converted to a success.
        charge = _array_bytes(value) + 1024 + len(encoded) + len(namespace)
        if charge > self._max_bytes:
            with self._lock:
                self._bypasses += 1
            return value
        try:
            owned = copy.deepcopy(value)
        except MemoryError:
            with self._lock:
                self._bypasses += 1
            return value
        with self._lock:
            if epoch != self._epoch:
                self._bypasses += 1
            elif key not in self._entries:
                while self._retained_bytes + charge > self._max_bytes:
                    _, removed = self._entries.popitem(last=False)
                    self._retained_bytes -= removed.charge
                    self._evictions += 1
                self._entries[key] = _Entry(owned, charge)
                self._retained_bytes += charge
        return value
