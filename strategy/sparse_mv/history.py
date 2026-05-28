from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

from common.events import MarketSnapshot
from .math_utils import _is_finite_pos, _safe_float


class SparseHistoryMixin:
    """Daily aggregation and return-matrix construction for sparse allocation."""

    def _period_key(self, ts: datetime) -> object:
        if self.estimation_frequency == "1d":
            return ts.date()
        raise ValueError(f"Unsupported estimation_frequency={self.estimation_frequency}")

    def _update_aggregated_history(self, snap: MarketSnapshot) -> None:
        pkey = self._period_key(snap.ts)
        if self._current_period_key is None:
            self._current_period_key = pkey
        elif pkey != self._current_period_key:
            self._finalize_current_period()
            self._current_period_key = pkey
            self._current_period_close = {}

        for sym, px in snap.prices.items():
            v = _safe_float(px, default=float("nan"))
            if _is_finite_pos(v):
                self._current_period_close[str(sym)] = v

    def _finalize_current_period(self) -> None:
        if self._current_period_key is None or not self._current_period_close:
            return
        self._period_closes.append((self._current_period_key, dict(self._current_period_close)))
        while len(self._period_closes) > max(8, int(self.lookback_bars)):
            self._period_closes.popleft()

    def _snapshot_history_with_current(self) -> List[Tuple[object, Dict[str, float]]]:
        items = list(self._period_closes)
        if self._current_period_key is not None and self._current_period_close:
            items.append((self._current_period_key, dict(self._current_period_close)))
        if len(items) > max(8, int(self.lookback_bars)):
            items = items[-int(self.lookback_bars):]
        return items

    def _build_return_matrix(self, snapshots: Sequence[Tuple[object, Dict[str, float]]]) -> Tuple[List[str], np.ndarray]:
        symbol_set = set()
        for _, closes in snapshots:
            symbol_set.update(closes.keys())
        symbols = sorted(symbol_set)
        if not symbols or len(snapshots) < 2:
            return [], np.zeros((0, 0), dtype=float)

        rows: List[List[float]] = []
        for i in range(1, len(snapshots)):
            prev_close = snapshots[i - 1][1]
            cur_close = snapshots[i][1]
            row = []
            any_valid = False
            for sym in symbols:
                p0 = _safe_float(prev_close.get(sym, float("nan")), default=float("nan"))
                p1 = _safe_float(cur_close.get(sym, float("nan")), default=float("nan"))
                if _is_finite_pos(p0) and _is_finite_pos(p1):
                    r = (p1 / p0) - 1.0
                    if math.isfinite(r):
                        row.append(float(r))
                        any_valid = True
                        continue
                row.append(float("nan"))
            if any_valid:
                rows.append(row)

        if not rows:
            return symbols, np.zeros((0, len(symbols)), dtype=float)
        return symbols, np.asarray(rows, dtype=float)

    def _extract_prev_weights(self, symbols: Sequence[str]) -> np.ndarray:
        raw = np.array([_safe_float(self._last_target_weights.get(sym, 0.0)) for sym in symbols], dtype=float)
        raw[raw < 0.0] = 0.0
        s = float(raw.sum())
        if s <= 0.0:
            return np.zeros_like(raw)
        return raw / s
