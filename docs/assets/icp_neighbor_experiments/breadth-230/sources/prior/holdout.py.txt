"""Deterministic holdout split helpers."""

from __future__ import annotations

import random


def split_indices(
    count: int,
    holdout_ratio: float,
    seed: int | None = None,
) -> tuple[list[int], list[int]]:
    """Split indices into deterministic train and holdout sets."""

    if count < 0:
        msg = "count must be non-negative"
        raise ValueError(msg)
    if not 0.0 <= holdout_ratio <= 0.9:
        msg = "holdout_ratio must be between 0.0 and 0.9"
        raise ValueError(msg)
    indices = list(range(count))
    rng = random.Random(seed)
    rng.shuffle(indices)
    holdout_count = round(count * holdout_ratio)
    holdout = sorted(indices[:holdout_count])
    train = sorted(indices[holdout_count:])
    return train, holdout
