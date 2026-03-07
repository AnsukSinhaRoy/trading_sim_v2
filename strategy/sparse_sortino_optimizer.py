from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List
from collections import deque
import math
import numpy as np
from datetime import timedelta

from common.events import MarketSnapshot, OrderRequest, PositionSnapshot


def _is_good_px(x: float) -> bool:
    return math.isfinite(x) and x > 0.0


def _safe_ret(p1: float, p0: float) -> float:
    if not (_is_good_px(p1) and _is_good_px(p0)):
        return 0.0
    return (p1 / p0) - 1.0


def _sortino_ratio(returns: np.ndarray) -> float:
    if len(returns) < 5:
        return 0.0
    mean = returns.mean()
    downside = returns[returns < 0]
    if len(downside) == 0:
        return mean
    downside_std = downside.std()
    return mean / (downside_std + 1e-8)


@dataclass
class SparseSortinoOptimizerStrategy:
    """
    Online sparse Sortino optimizer adapted for trading_sim_v2.

    - Periodic re-optimization
    - Rolling return window
    - Outputs target weights only
    """

    rebalance_every_minutes: int = 60
    lookback: int = 240                 # bars
    max_assets: int = 5
    max_gross_leverage: float = 1.0

    _last_rebalance_ts: object = None
    _px_history: Dict[str, deque] = field(default_factory=dict)
    _ret_history: Dict[str, deque] = field(default_factory=dict)
    _last_target_weights: Dict[str, float] = field(default_factory=dict)

    def on_snapshot(
        self,
        snap: MarketSnapshot,
        portfolio: PositionSnapshot
    ) -> List[OrderRequest]:

        ts = snap.ts

        # -------------------------------
        # 1. Update price & return history
        # -------------------------------
        for sym, px in snap.prices.items():
            if not _is_good_px(px):
                continue

            self._px_history.setdefault(sym, deque(maxlen=self.lookback))
            self._ret_history.setdefault(sym, deque(maxlen=self.lookback))

            px_hist = self._px_history[sym]
            if px_hist:
                ret = _safe_ret(px, px_hist[-1])
                self._ret_history[sym].append(ret)

            px_hist.append(px)

        # -------------------------------
        # 2. Rebalance timing
        # -------------------------------
        if self._last_rebalance_ts is not None:
            if ts - self._last_rebalance_ts < timedelta(minutes=self.rebalance_every_minutes):
                return []

        self._last_rebalance_ts = ts

        # -------------------------------
        # 3. Compute Sortino scores
        # -------------------------------
        scores = []
        for sym, rets in self._ret_history.items():
            if len(rets) < max(20, self.lookback // 4):
                continue
            s = _sortino_ratio(np.array(rets))
            scores.append((sym, s))

        if not scores:
            self._last_target_weights = {}
            return []

        scores.sort(key=lambda x: x[1], reverse=True)
        selected = scores[: self.max_assets]

        # -------------------------------
        # 4. Weight allocation
        # -------------------------------
        raw_scores = np.array([max(s, 0.0) for _, s in selected])
        if raw_scores.sum() <= 0:
            w = np.ones(len(selected)) / len(selected)
        else:
            w = raw_scores / raw_scores.sum()

        w *= self.max_gross_leverage

        self._last_target_weights = {
            sym: float(weight)
            for (sym, _), weight in zip(selected, w)
            if abs(weight) > 1e-6
        }

        # Engine converts weights -> orders
        return []
