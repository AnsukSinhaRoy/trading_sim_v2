from __future__ import annotations

"""
strategy/ema_long.py

EMA-based long-only allocator for 1-minute data with:

- Multi-EMA trend filter (fast/slow/trend)
- Cross-sectional ranking using a *trend-following* score:
    score = mom_weight * momentum + macd_weight * log(EMA_fast / EMA_slow)
  and risk-adjusted by EWMA volatility.
- Volatility-aware trailing stoploss:
    stop_pct = max(stoploss_pct, stop_vol_mult * daily_vol)
  where daily_vol is estimated from EWMA variance of 1-minute log-returns.
- Cooldown after stop / exit to avoid immediate re-entry.
- Optional session filter (highly recommended if your feed forward-fills non-trading minutes).

Engine contract (this repo):
- runner/engine.py converts `self._last_target_weights` into rebalance orders.
- If you want to liquidate, publish {held_symbol: 0.0, ...} for held symbols.

Why this is stronger than the previous version
- The repo's FolderMinuteFeed builds a full 1-minute calendar between start/end and
  forward-fills. If you run multi-day windows, that includes nights/weekends, which
  dilutes indicators and lookbacks. This strategy can ignore non-session minutes and
  measure lookbacks in *trading minutes*.
- Adds volatility targeting + overbought filter to avoid concentration in very volatile names.
"""

from dataclasses import dataclass, field
from collections import deque
from typing import Dict, Deque, List, Tuple, Optional
from datetime import datetime, time
import math

from common.events import MarketSnapshot, PositionSnapshot


# -------------------- helpers --------------------

def _is_good_px(x: float) -> bool:
    return math.isfinite(x) and x > 0.0


def _safe_logret(p1: float, p0: float) -> float:
    if not (_is_good_px(p1) and _is_good_px(p0)):
        return float("nan")
    return math.log(p1 / p0)


def _alpha_from_halflife(halflife: int) -> float:
    # EWMA alpha from half-life in ticks: weight halves every `halflife` updates.
    hl = max(1, int(halflife))
    return 1.0 - math.exp(math.log(0.5) / hl)


def _parse_time(x) -> Optional[time]:
    if x is None:
        return None
    if isinstance(x, time):
        return x
    if isinstance(x, str):
        s = x.strip()
        # Accept "HH:MM" or "HH:MM:SS"
        parts = s.split(":")
        if len(parts) >= 2:
            hh = int(parts[0]); mm = int(parts[1]); ss = int(parts[2]) if len(parts) >= 3 else 0
            return time(hour=hh, minute=mm, second=ss)
    return None


# -------------------- state --------------------

@dataclass
class _SymState:
    prices: Deque[float]

    ema_fast: float = float("nan")
    ema_slow: float = float("nan")
    ema_trend: float = float("nan")

    prev_ema_slow: float = float("nan")
    prev_ema_trend: float = float("nan")

    prev_px: float = float("nan")
    ewma_var: float = 0.0

    # Position/stop bookkeeping (tracked in *trading* ticks)
    was_held: bool = False
    entry_tick: int = 0
    entry_px: float = 0.0
    peak_px_while_held: float = 0.0
    cooldown_until_tick: int = 0


# -------------------- strategy --------------------

@dataclass
class EmaLongStrategy:
    # --- cadence / warmup (measured in *trading* minutes if session filter is set) ---
    rebalance_every_minutes: int = 1950     # ~5 NSE trading days (5 * 390ish)
    warmup_minutes: int = 3900             # wait until indicators stabilize
    target_count: int = 10

    # --- EMAs (minutes) ---
    ema_fast_mins: int = 390               # ~1 trading day
    ema_slow_mins: int = 1950              # ~5 trading days
    ema_trend_mins: int = 7800             # ~20 trading days
    ema_band: float = 0.0005               # fast must exceed slow by this band (fraction)

    # --- momentum / risk estimates ---
    mom_lookback_mins: int = 1950
    vol_halflife_mins: int = 1950          # EWMA vol half-life (trading minutes)

    # --- scoring ---
    mom_weight: float = 1.0
    macd_weight: float = 0.7               # weight for log(EMA_fast/EMA_slow)
    min_momentum: float = 0.0              # require momentum >= this
    min_score: float = 0.0                 # require score/(vol) >= this

    # --- filters ---
    require_price_above_trend: bool = True
    require_trend_up: bool = True          # trend EMA increasing
    require_slow_up: bool = True           # slow EMA increasing
    max_overbought_pct: float = 0.10       # don't enter if price > EMA_slow*(1+this)

    # --- portfolio / constraints ---
    max_gross_leverage: float = 1.0
    cash_buffer: float = 0.05
    max_weight: float = 0.15
    max_turnover: float = 0.35
    min_rebalance_change: float = 1e-4

    # --- exits / stops ---
    stoploss_pct: float = 0.06             # minimum trailing stop from peak (fraction)
    stop_vol_mult: float = 2.0             # extra stop = vol_mult * daily_vol (set 0 to disable)
    stop_max_pct: float = 0.18             # cap dynamic stop
    stop_cooldown_minutes: int = 1950
    min_hold_minutes: int = 390            # don't rotate out before this (unless stop/exit triggers)

    exit_on_trend_break: bool = True
    trend_break_buffer: float = 0.0        # require price < EMA_slow*(1-buffer) to exit (0 = strict)

    # --- session filter (recommended if your feed forward-fills) ---
    # Set these in YAML for your market. For NSE:
    #   session_start: "09:15"
    #   session_end:   "15:30"
    session_start: Optional[str] = None
    session_end: Optional[str] = None
    skip_weekends: bool = True
    minutes_per_day: int = 375             # NSE regular session minutes. Use 390 for US.

    # --- state ---
    _ticks_all: int = 0                    # all snapshots received
    _ticks_trading: int = 0                # trading minutes only (when session filter passes)
    _states: Dict[str, _SymState] = field(default_factory=dict)

    # Persist last intended weights (for turnover and stop events)
    _cur_target_weights: Dict[str, float] = field(default_factory=dict)

    # Published to engine only on rebalance/exit events
    _last_target_weights: Dict[str, float] = field(default_factory=dict)

    def _ensure_state(self, sym: str) -> _SymState:
        st = self._states.get(sym)
        if st is None:
            st = _SymState(prices=deque(maxlen=int(self.mom_lookback_mins) + 1))
            self._states[sym] = st
        return st

    def _is_trading_minute(self, dt: datetime) -> bool:
        if self.skip_weekends and dt.weekday() >= 5:
            return False

        s0 = _parse_time(self.session_start)
        s1 = _parse_time(self.session_end)
        if s0 is None or s1 is None:
            return True
        t = dt.time()
        return (t >= s0) and (t <= s1)

    @staticmethod
    def _ema_update(prev: float, price: float, span: int) -> float:
        a = 2.0 / (max(1, int(span)) + 1.0)
        if not math.isfinite(prev):
            return float(price)
        return a * float(price) + (1.0 - a) * prev

    def _set_publish_weights(self, w: Dict[str, float]) -> None:
        # Engine only trades if dict is non-empty.
        self._last_target_weights = {k: float(v) for k, v in w.items() if math.isfinite(float(v))}

    def _go_cash_weights(self, portfolio: PositionSnapshot) -> Dict[str, float]:
        pos = dict(getattr(portfolio, "positions", {}) or {})
        return {sym: 0.0 for sym, qty in pos.items() if int(qty) != 0}

    def _l1_diff(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in (set(a) | set(b)))

    def _turnover_blend(self, w_old: Dict[str, float], w_new: Dict[str, float]) -> Dict[str, float]:
        all_syms = set(w_old) | set(w_new)
        turnover = sum(abs(w_new.get(s, 0.0) - w_old.get(s, 0.0)) for s in all_syms)
        if self.max_turnover > 0 and turnover > self.max_turnover:
            alpha = self.max_turnover / max(1e-12, turnover)
            out = {s: w_old.get(s, 0.0) + alpha * (w_new.get(s, 0.0) - w_old.get(s, 0.0)) for s in all_syms}
            return {k: v for k, v in out.items() if abs(v) > 1e-8}
        return {k: v for k, v in w_new.items() if abs(v) > 1e-8}

    # -------------------- main hook --------------------

    def on_snapshot(self, snap: MarketSnapshot, portfolio: PositionSnapshot):
        self._ticks_all += 1

        # Clear publish weights by default (prevents continuous rebalance).
        self._last_target_weights = {}

        dt = getattr(snap, "ts", None)
        if not isinstance(dt, datetime):
            return []

        # If we are filtering session minutes, only update indicators on "trading" minutes.
        if not self._is_trading_minute(dt):
            return []

        self._ticks_trading += 1
        t = self._ticks_trading

        # --- 1) update per-symbol indicators ---
        a_fast = 2.0 / (max(1, int(self.ema_fast_mins)) + 1.0)
        a_slow = 2.0 / (max(1, int(self.ema_slow_mins)) + 1.0)
        a_trend = 2.0 / (max(1, int(self.ema_trend_mins)) + 1.0)
        a_vol = _alpha_from_halflife(self.vol_halflife_mins)

        for sym, px in (snap.prices or {}).items():
            try:
                p = float(px)
            except (TypeError, ValueError):
                continue
            if not _is_good_px(p):
                continue

            st = self._ensure_state(sym)
            st.prices.append(p)

            # EMA updates (with slope memory for slow/trend)
            st.prev_ema_slow = st.ema_slow
            st.prev_ema_trend = st.ema_trend

            st.ema_fast = p if not math.isfinite(st.ema_fast) else (a_fast * p + (1.0 - a_fast) * st.ema_fast)
            st.ema_slow = p if not math.isfinite(st.ema_slow) else (a_slow * p + (1.0 - a_slow) * st.ema_slow)
            st.ema_trend = p if not math.isfinite(st.ema_trend) else (a_trend * p + (1.0 - a_trend) * st.ema_trend)

            # EWMA variance of 1-min log returns
            if math.isfinite(st.prev_px):
                r = _safe_logret(p, st.prev_px)
                if math.isfinite(r):
                    st.ewma_var = (1.0 - a_vol) * st.ewma_var + a_vol * (r * r)
            st.prev_px = p

        # --- 2) exits: trailing stop + trend break (can trigger any trading minute) ---
        positions = dict(getattr(portfolio, "positions", {}) or {})
        held_syms = {sym for sym, qty in positions.items() if int(qty) > 0}

        exit_syms: List[str] = []
        stopped_syms: List[str] = []

        for sym in held_syms:
            st = self._states.get(sym)
            if st is None or not st.prices:
                continue
            px = st.prices[-1]

            # init entry bookkeeping
            if not st.was_held:
                st.was_held = True
                st.entry_tick = t
                st.entry_px = px
                st.peak_px_while_held = px
            else:
                st.peak_px_while_held = max(st.peak_px_while_held, px)

            age = t - int(st.entry_tick)

            # cooldown enforcement (if already cooling down, ensure we target 0)
            if t < int(st.cooldown_until_tick):
                if self._cur_target_weights.get(sym, 0.0) != 0.0:
                    self._cur_target_weights[sym] = 0.0
                    exit_syms.append(sym)
                continue

            # dynamic stop percent
            stop_pct = max(0.0, float(self.stoploss_pct))
            if self.stop_vol_mult and self.stop_vol_mult > 0:
                vol = math.sqrt(max(0.0, st.ewma_var))
                if math.isfinite(vol) and vol > 0:
                    daily_vol = vol * math.sqrt(max(1.0, float(self.minutes_per_day)))
                    stop_pct = max(stop_pct, float(self.stop_vol_mult) * daily_vol)
            if self.stop_max_pct and self.stop_max_pct > 0:
                stop_pct = min(stop_pct, float(self.stop_max_pct))

            # trailing stop
            if stop_pct > 0:
                stop_level = st.peak_px_while_held * (1.0 - stop_pct)
                if _is_good_px(stop_level) and px <= stop_level:
                    st.cooldown_until_tick = t + max(1, int(self.stop_cooldown_minutes))
                    self._cur_target_weights[sym] = 0.0
                    stopped_syms.append(sym)
                    continue

            # trend-break exit (so we don't hold deep mean-reversion)
            if self.exit_on_trend_break and age >= int(self.min_hold_minutes):
                if math.isfinite(st.ema_fast) and math.isfinite(st.ema_slow):
                    if st.ema_fast <= st.ema_slow:
                        st.cooldown_until_tick = t + max(1, int(self.stop_cooldown_minutes // 2))
                        self._cur_target_weights[sym] = 0.0
                        exit_syms.append(sym)
                        continue
                if math.isfinite(st.ema_slow):
                    thresh = st.ema_slow * (1.0 - float(self.trend_break_buffer))
                    if px < thresh:
                        st.cooldown_until_tick = t + max(1, int(self.stop_cooldown_minutes // 2))
                        self._cur_target_weights[sym] = 0.0
                        exit_syms.append(sym)
                        continue

        # Reset held-state for symbols no longer held
        for sym, st in self._states.items():
            if st.was_held and sym not in held_syms:
                st.was_held = False
                st.peak_px_while_held = 0.0

        # If any exits/stops triggered: publish *full* weight map (to avoid accidental liquidation)
        if stopped_syms or exit_syms:
            # Ensure all held symbols are represented; keep existing weights for the rest.
            for sym in held_syms:
                self._cur_target_weights.setdefault(sym, self._cur_target_weights.get(sym, 0.0))

            # Remove tiny weights
            publish = {k: float(v) for k, v in self._cur_target_weights.items() if abs(float(v)) > 1e-9}
            # Still include held symbols with 0 so engine will liquidate them
            for sym in held_syms:
                publish.setdefault(sym, 0.0)

            if publish:
                self._set_publish_weights(publish)
            else:
                w_cash = self._go_cash_weights(portfolio)
                if w_cash:
                    self._cur_target_weights = dict(w_cash)
                    self._set_publish_weights(w_cash)
            return []

        # --- 3) rebalance schedule ---
        if self.rebalance_every_minutes <= 0 or (t % int(self.rebalance_every_minutes) != 0):
            return []
        if t < int(self.warmup_minutes):
            return []

        # --- 4) score candidates ---
        scored: List[Tuple[str, float, float, float]] = []  # (sym, risk_adj_score, mom, vol)

        for sym, st in self._states.items():
            if t < int(st.cooldown_until_tick):
                continue
            if len(st.prices) < int(self.mom_lookback_mins) + 1:
                continue
            px = st.prices[-1]
            if not _is_good_px(px):
                continue

            # EMA filters
            if not (math.isfinite(st.ema_fast) and math.isfinite(st.ema_slow) and math.isfinite(st.ema_trend)):
                continue
            if st.ema_fast <= st.ema_slow * (1.0 + float(self.ema_band)):
                continue
            if self.require_price_above_trend and px <= st.ema_trend:
                continue
            if self.require_slow_up and (not math.isfinite(st.prev_ema_slow) or st.ema_slow <= st.prev_ema_slow):
                continue
            if self.require_trend_up and (not math.isfinite(st.prev_ema_trend) or st.ema_trend <= st.prev_ema_trend):
                continue

            # Overbought filter
            if self.max_overbought_pct and self.max_overbought_pct > 0:
                if px > st.ema_slow * (1.0 + float(self.max_overbought_pct)):
                    continue

            # Momentum (finite window)
            p0 = st.prices[0]
            mom = _safe_logret(px, p0)
            if not math.isfinite(mom) or mom < float(self.min_momentum):
                continue

            # MACD-like term
            macd = math.log(max(1e-12, st.ema_fast) / max(1e-12, st.ema_slow))

            # Volatility (EWMA)
            vol = math.sqrt(max(0.0, st.ewma_var))
            if not math.isfinite(vol) or vol <= 0:
                continue

            base_score = float(self.mom_weight) * mom + float(self.macd_weight) * macd
            risk_adj = base_score / (vol + 1e-12)

            if not math.isfinite(risk_adj) or risk_adj < float(self.min_score):
                continue

            scored.append((sym, float(risk_adj), float(mom), float(vol)))

        if not scored:
            # No opportunities: go cash / liquidate
            w_cash = self._go_cash_weights(portfolio)
            self._cur_target_weights = dict(w_cash)
            if w_cash:
                self._set_publish_weights(w_cash)
            return []

        # Regime filter: if median momentum <= 0 -> go cash (avoid buying in broad drawdowns)
        moms = sorted([m for _, _, m, _ in scored])
        if moms[len(moms) // 2] <= 0.0:
            w_cash = self._go_cash_weights(portfolio)
            self._cur_target_weights = dict(w_cash)
            if w_cash:
                self._set_publish_weights(w_cash)
            return []

        # pick top-K by risk-adjusted score
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[: int(self.target_count)]

        # weights proportional to risk-adjusted score
        raw: Dict[str, float] = {sym: max(0.0, s) for sym, s, _, _ in top if math.isfinite(s) and s > 0.0}
        if not raw:
            w_cash = self._go_cash_weights(portfolio)
            self._cur_target_weights = dict(w_cash)
            if w_cash:
                self._set_publish_weights(w_cash)
            return []

        gross_target = max(0.0, float(self.max_gross_leverage) * (1.0 - float(self.cash_buffer)))
        sraw = sum(raw.values())
        w_new = {sym: (gross_target * v / sraw) for sym, v in raw.items()}

        # cap per-name
        for sym in list(w_new.keys()):
            w_new[sym] = min(float(self.max_weight), float(w_new[sym]))

        # renormalize after caps (keep gross <= gross_target)
        s2 = sum(max(0.0, v) for v in w_new.values())
        if s2 > 0 and s2 > gross_target:
            scale = gross_target / s2
            for sym in w_new:
                w_new[sym] *= scale

        # enforce minimum hold: keep weight for too-young positions
        # (only if we already had a target weight for them)
        for sym in held_syms:
            st = self._states.get(sym)
            if st is None or not st.was_held:
                continue
            age = t - int(st.entry_tick)
            if age < int(self.min_hold_minutes) and sym in self._cur_target_weights:
                w_new[sym] = float(self._cur_target_weights.get(sym, 0.0))

        # blend with old weights to cap turnover
        w_old = dict(self._cur_target_weights)
        w_new = self._turnover_blend(w_old, w_new)

        if self._l1_diff(w_old, w_new) < float(self.min_rebalance_change):
            return []

        # Persist intended weights
        self._cur_target_weights = dict(w_new)

        # Ensure held symbols not selected get explicitly liquidated
        for sym in held_syms:
            w_new.setdefault(sym, 0.0)

        if w_new:
            self._set_publish_weights(w_new)
        return []


# Convenience hook for engine fallback resolution
STRATEGY_CLASS = EmaLongStrategy
