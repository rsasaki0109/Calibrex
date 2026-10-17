"""Bounded row sums with the supported CPython float accumulation order."""

from __future__ import annotations

import sys

import numpy as np
from numpy.typing import NDArray

_MIN_ROWS = 256
_MAX_ROWS = 65536
_MAX_COLUMNS = 64
_SUM_MODE: tuple[int, int] | None = (
    sys.version_info[:2] if sys.implementation.name == "cpython" else None
)


def _ordered_float_sums(
    products: NDArray[np.float64], *, version: tuple[int, int] | None = _SUM_MODE,
    minimum_rows: int = _MIN_ROWS,
) -> NDArray[np.float64] | None:
    """Avoid Python row/list allocation without using a reordered reduction."""
    if (
        version not in ((3, 10), (3, 11), (3, 12))
        or type(products) is not np.ndarray or products.dtype != np.float64
        or products.ndim not in (2, 3)
        or not 0 < products.shape[-1] <= _MAX_COLUMNS
        or not minimum_rows <= products.size // products.shape[-1] <= _MAX_ROWS
        or not np.isfinite(products).all() or np.any(np.abs(products) > 1e100)
    ):
        return None
    with np.errstate(all="ignore"):
        total = 0.0 + products[..., 0]
        if version != (3, 12):
            for column in range(1, products.shape[-1]):
                total += products[..., column]
            return total
        correction = np.zeros_like(total)
        for column in range(1, products.shape[-1]):
            term = products[..., column]
            updated = total + term
            correction += np.where(
                np.abs(total) >= np.abs(term), (total - updated) + term,
                (term - updated) + total,
            )
            total = updated
        return np.where(correction != 0.0, total + correction, total)
