from __future__ import annotations

import math
from typing import Optional

import numpy as np


def _is_finite_pos(x: float) -> bool:
    return math.isfinite(x) and x > 0.0

def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    if not math.isfinite(v):
        return float(default)
    return float(v)

def _project_to_capped_simplex(v: np.ndarray, z: float = 1.0, cap: Optional[float] = None) -> np.ndarray:
    """
    Euclidean projection onto {w >= 0, sum(w)=z, w_i <= cap if cap is not None}.

    Uses bisection on the dual variable. Stable and dependency-light.
    """
    x = np.asarray(v, dtype=float)
    n = int(x.size)
    if n == 0:
        return x.copy()

    if z <= 0.0:
        return np.zeros_like(x)

    if cap is not None:
        cap = float(cap)
        if cap <= 0.0:
            return np.zeros_like(x)
        if cap * n < z - 1e-12:
            # Infeasible cap; relax to uniform feasible allocation.
            return np.full(n, z / n, dtype=float)

    lo = float(np.min(x) - (cap if cap is not None else z) - 1.0)
    hi = float(np.max(x) + 1.0)

    for _ in range(80):
        tau = 0.5 * (lo + hi)
        w = x - tau
        if cap is None:
            w = np.maximum(w, 0.0)
        else:
            w = np.clip(w, 0.0, cap)
        s = float(w.sum())
        if s > z:
            lo = tau
        else:
            hi = tau

    w = x - hi
    if cap is None:
        w = np.maximum(w, 0.0)
    else:
        w = np.clip(w, 0.0, cap)

    s = float(w.sum())
    if s <= 0.0:
        if cap is None:
            return np.full(n, z / n, dtype=float)
        out = np.zeros(n, dtype=float)
        remaining = z
        for i in np.argsort(-x):
            take = min(cap, remaining)
            out[i] = take
            remaining -= take
            if remaining <= 1e-12:
                break
        return out
    return w * (z / s)

def _condition_number(mat: np.ndarray) -> float:
    try:
        eigvals = np.linalg.eigvalsh(mat)
    except np.linalg.LinAlgError:
        return float("inf")
    eigvals = np.asarray(eigvals, dtype=float)
    if eigvals.size == 0:
        return 0.0
    mn = float(np.min(np.abs(eigvals)))
    mx = float(np.max(np.abs(eigvals)))
    if mn <= 1e-18:
        return float("inf")
    return mx / mn
