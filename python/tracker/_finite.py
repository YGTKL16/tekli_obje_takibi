"""Numeric finiteness guards for tracker hot paths.

Set env var ``TRACKER_ASSERT_FINITE=1`` to escalate warnings to FloatingPointError.
Default behavior: warn-once-per-name + sanitize. Hot paths should call
``sanitize`` (cheap, fills non-finite with ``fill``) and reserve ``assert_finite``
for diagnostic / smoke runs.
"""
from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np

_RAISE = os.environ.get("TRACKER_ASSERT_FINITE", "0") not in ("0", "", "false", "False")
_warned: set[str] = set()


def _is_finite(x: Any) -> bool:
    arr = np.asarray(x)
    if arr.dtype.kind not in ("f", "c"):
        return True
    return bool(np.isfinite(arr).all())


def assert_finite(name: str, x: Any) -> None:
    """Warn (or raise if TRACKER_ASSERT_FINITE=1) when ``x`` has NaN/Inf."""
    if _is_finite(x):
        return
    msg = f"[finite-guard] non-finite values in '{name}'"
    if _RAISE:
        raise FloatingPointError(msg)
    if name not in _warned:
        _warned.add(name)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)


def sanitize(x: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Replace NaN/Inf with ``fill`` (in-place safe; returns same dtype)."""
    if _is_finite(x):
        return x
    return np.nan_to_num(x, nan=fill, posinf=fill, neginf=fill)
