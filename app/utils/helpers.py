"""
helpers.py
----------
Shared utility functions used across services.
"""

from typing import Any
import math


def safe_round(value: Any, decimals: int = 4) -> float | None:
    """Round a numeric value safely; returns None if not numeric or NaN."""
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, decimals)
    except (TypeError, ValueError):
        return None


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))


def normalize(value: float, min_val: float, max_val: float) -> float:
    """
    Normalize a value to [-1, 1] given its possible range.
    Avoids division by zero.
    """
    span = max_val - min_val
    if span == 0:
        return 0.0
    return clamp((value - min_val) / span * 2 - 1, -1.0, 1.0)


def pct_change(old: float, new: float) -> float | None:
    """Percentage change from old to new. Returns None if old is 0."""
    if old == 0:
        return None
    return round((new - old) / old * 100, 2)
