from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from collections import deque
import math

from common.events import MarketSnapshot, OrderRequest, PositionSnapshot


def _is_good_px(x: float) -> bool:
    return math.isfinite(x) and x > 0.0


def _safe_logret(p1: float, p0: float) -> float:
    if not (_is_good_px(p1) and _is_good_px(p0)):
        return float("nan")
    return math.log(p1 / p0)


def _mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))


def _std(xs: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(max(0.0, v))


@dataclass
class XSMomVolTargetStrategy:
    # Rebalance cadence (in minutes / ticks since your feed is 1-minute)
    rebalance_every_minutes: int = 60

    # Universe selection
    target_count: int = 10

    # Signal/vol lookbacks (in minutes)
    signal_lookback_mins: int = 120   # momentum window
    vol_lookback_mins: int = 120      # volatility window

    # Risk constraints
    max_gross_leverage: float = 1.0   # 1.0 = fully invested long-only
    cash_buffer: float = 0.05         # keep 5% cash
    max_weight: float = 0.15          # cap per-name weight
    max_turnover: float = 0.60        # cap sum(|w_new - w_old|) each rebalance

    # Filters
    min_signal: float = 0.0           # require momentum > this
    reset_on_new_day: bool = True     # avoid overnight artifacts for intraday signals

    # Optional safety: if NAV drops > drawdown_stop from peak, go to cash
    drawdown_stop: float = 0.25       # 25% dd -> flat; set to 0 to disable

    # Internal state
    _history: Dict[str, deque] = field(default_factory=dict)  # sym -> deque(prices)
    _ticks: int = 0
    _last_target_weights: Dict[str, float] = field(default_factory=dict)
    _last_date: str | None = None
    _peak_nav: float = 0.0

    def on_snapshot(self, snap: MarketSnapshot, portfolio: PositionSnapshot) -> List[OrderRequest]:
        self._ticks += 1

        # --- day boundary handling (intraday signal hygiene) ---
        ts_str = getattr(snap, "ts", None)
        if ts_str is not None:
            # expecting ISO string like "2015-02-04T13:22:00"
            day = str(ts_str)[:10]
            if self.reset_on_new_day and self._last_date is not None and day != self._last_date:
                self._history.clear()
            self._last_date = day

        # --- drawdown safety (optional) ---
        nav = getattr(portfolio, "nav", None)
        if nav is not None and math.isfinite(float(nav)):
            navf = float(nav)
            if navf > self._peak_nav:
                self._peak_nav = navf
            if self.drawdown_stop > 0 and self._peak_nav > 0:
                dd = 1.0 - (navf / self._peak_nav)
                if dd >= self.drawdown_stop:
                    self._last_target_weights = {}
                    return []  # go flat

        # --- update history with clean prices only ---
        maxlen = max(self.signal_lookback_mins, self.vol_lookback_mins) + 1
        for sym, px in snap.prices.items():
            try:
                v = float(px)
            except (TypeError, ValueError):
                continue
            if not _is_good_px(v):
                continue

            dq = self._history.get(sym)
            if dq is None:
                dq = deque(maxlen=maxlen)
                self._history[sym] = dq
            dq.append(v)

        # --- rebalance schedule ---
        if self._ticks % max(1, self.rebalance_every_minutes) != 0:
            return []

        # --- compute cross-sectional scores ---
        scored: List[Tuple[str, float, float]] = []  # (sym, signal, vol)
        for sym, dq in self._history.items():
            prices = list(dq)
            if len(prices) < 3:
                continue

            # signal: log return over signal_lookback_mins
            if len(prices) < self.signal_lookback_mins + 1:
                continue
            p0 = prices[-1 - self.signal_lookback_mins]
            p1 = prices[-1]
            sig = _safe_logret(p1, p0)
            if not math.isfinite(sig):
                continue
            if sig <= self.min_signal:
                continue

            # vol: std dev of 1-min log returns over vol_lookback_mins
            rets: List[float] = []
            start_i = max(1, len(prices) - self.vol_lookback_mins)
            for i in range(start_i, len(prices)):
                r = _safe_logret(prices[i], prices[i - 1])
                if math.isfinite(r):
                    rets.append(r)
            vol = _std(rets)
            if not math.isfinite(vol) or vol <= 0.0:
                continue

            scored.append((sym, sig, vol))

        if not scored:
            self._last_target_weights = {}
            return []

        # regime filter: if median signal is <= 0, go to cash
        sigs = sorted([s for _, s, _ in scored])
        med_sig = sigs[len(sigs) // 2]
        if med_sig <= 0.0:
            self._last_target_weights = {}
            return []

        # pick top-K by signal (momentum)
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[: self.target_count]

        # raw weights proportional to (signal / vol) (risk-adjusted momentum)
        raw: Dict[str, float] = {}
        for sym, sig, vol in top:
            w = sig / vol
            if math.isfinite(w) and w > 0:
                raw[sym] = w

        if not raw:
            self._last_target_weights = {}
            return []

        # normalize to gross exposure = max_gross_leverage * (1 - cash_buffer)
        gross_target = max(0.0, self.max_gross_leverage * (1.0 - self.cash_buffer))
        s = sum(raw.values())
        w_new = {sym: (gross_target * v / s) for sym, v in raw.items()}

        # cap per-name weights
        for sym in list(w_new.keys()):
            w_new[sym] = min(w_new[sym], self.max_weight)

        # renormalize after caps (keep gross <= gross_target)
        s2 = sum(w_new.values())
        if s2 > 0 and s2 > gross_target:
            scale = gross_target / s2
            for sym in w_new:
                w_new[sym] *= scale

        # turnover cap: scale deltas if too large
        w_old = self._last_target_weights or {}
        turnover = 0.0
        all_syms = set(w_old) | set(w_new)
        for sym in all_syms:
            turnover += abs(w_new.get(sym, 0.0) - w_old.get(sym, 0.0))

        if self.max_turnover > 0 and turnover > self.max_turnover:
            # shrink the move towards new weights
            alpha = self.max_turnover / turnover
            w_blend: Dict[str, float] = {}
            for sym in all_syms:
                w_blend[sym] = w_old.get(sym, 0.0) + alpha * (w_new.get(sym, 0.0) - w_old.get(sym, 0.0))
            # drop tiny
            w_new = {k: v for k, v in w_blend.items() if abs(v) > 1e-6}

        self._last_target_weights = w_new
        return []  # engine can convert weights -> orders (your stack already does this)
