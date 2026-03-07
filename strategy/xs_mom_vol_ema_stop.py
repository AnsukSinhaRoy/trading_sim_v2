from __future__ import annotations

"""Cross-sectional momentum + vol targeting with EMA-trend filter and stoploss.

This is designed to be a "serious" long-only allocator for longer holding periods.
It builds on the core ideas of xs_mom_vol_target:
  - rank by momentum
  - scale by volatility
  - cap turnover and per-name weights

Improvements:
  - EMA fast/slow trend gate (helps avoid buying downtrends)
  - optional EWMA correlation-to-market penalty (diversification bias)
  - per-symbol trailing stoploss with cooldown
  - one-shot publishing of target weights to avoid unintended continuous rebalancing

Engine contract reminder:
  - runner/engine.py will place orders when strategy sets `_last_target_weights` to a
    non-empty dict. To avoid rebalancing every tick, this strategy sets that dict only
    on a rebalance/stop event, and clears it on other ticks.
"""

from dataclasses import dataclass, field
from collections import deque
from typing import Dict, Deque, List, Tuple
import math

from common.events import MarketSnapshot, PositionSnapshot


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


def _alpha_from_halflife(halflife: int) -> float:
    # EWMA alpha from half-life in ticks: weight halves every `halflife` updates
    hl = max(1, int(halflife))
    return 1.0 - math.exp(math.log(0.5) / hl)


@dataclass
class _SymState:
    # Prices for finite-window momentum
    prices: Deque[float]

    # EMA trend
    ema_fast: float = float("nan")
    ema_slow: float = float("nan")

    # EWMA volatility
    ewma_var: float = 0.0

    # EWMA correlation to "market" proxy
    ewma_cov_mkt: float = 0.0
    ewma_var_mkt: float = 0.0

    prev_px: float = float("nan")

    # Stoploss state
    peak_px_while_held: float = 0.0
    cooldown_until_tick: int = 0
    was_held: bool = False


@dataclass
class XSMomVolEmaStopStrategy:
    # ---- cadence / universe ----
    rebalance_every_minutes: int = 10080  # ~ 1 trading week (7 * 24 * 60) in 1-min ticks
    target_count: int = 10

    # ---- signals ----
    mom_lookback_mins: int = 1950  # ~ 5 trading days (5 * 390)
    ema_fast_mins: int = 1950      # align with short horizon
    ema_slow_mins: int = 7800      # ~ 20 trading days
    ema_band: float = 0.0          # require ema_fast > ema_slow * (1+band)

    # ---- risk / constraints ----
    max_gross_leverage: float = 1.0
    cash_buffer: float = 0.05
    max_weight: float = 0.15
    max_turnover: float = 0.50
    min_signal: float = 0.0

    # ---- risk overlays ----
    stoploss_pct: float = 0.10         # trailing stop from peak while held
    stop_cooldown_minutes: int = 1950  # block re-entry for ~5 trading days
    drawdown_stop: float = 0.0         # optional portfolio-level dd -> go cash (0 disables)

    # ---- volatility + correlation estimators ----
    vol_halflife_mins: int = 1950
    corr_halflife_mins: int = 1950
    corr_penalty: float = 0.0          # 0 disables; otherwise subtract penalty * |corr|

    # ---- state ----
    _ticks: int = 0
    _states: Dict[str, _SymState] = field(default_factory=dict)

    # Store last *intended* weights (persist for turnover calculations)
    _cur_target_weights: Dict[str, float] = field(default_factory=dict)

    # Publish weights to engine only when we want to rebalance right now.
    _last_target_weights: Dict[str, float] = field(default_factory=dict)

    _peak_nav: float = 0.0

    def _ensure_state(self, sym: str) -> _SymState:
        st = self._states.get(sym)
        if st is None:
            st = _SymState(prices=deque(maxlen=self.mom_lookback_mins + 1))
            self._states[sym] = st
        return st

    def _set_publish_weights(self, w: Dict[str, float]) -> None:
        # Engine only trades if dict is non-empty. To go "all cash" and liquidate,
        # include held symbols with 0 weights.
        self._last_target_weights = {k: float(v) for k, v in w.items() if math.isfinite(float(v))}

    def _go_cash_weights(self, portfolio: PositionSnapshot) -> Dict[str, float]:
        pos = dict(getattr(portfolio, "positions", {}) or {})
        return {sym: 0.0 for sym, qty in pos.items() if int(qty) != 0}

    def on_snapshot(self, snap: MarketSnapshot, portfolio: PositionSnapshot):
        self._ticks += 1

        # Clear publish weights by default to avoid continuous rebalancing.
        self._last_target_weights = {}

        # Portfolio-level drawdown stop (optional)
        nav = getattr(portfolio, "nav", None)
        if nav is not None and math.isfinite(float(nav)):
            navf = float(nav)
            self._peak_nav = max(self._peak_nav, navf)
            if self.drawdown_stop and self._peak_nav > 0:
                dd = 1.0 - (navf / self._peak_nav)
                if dd >= self.drawdown_stop:
                    w_cash = self._go_cash_weights(portfolio)
                    self._cur_target_weights = dict(w_cash)
                    if w_cash:
                        self._set_publish_weights(w_cash)
                    return []

        # --- Update per-symbol estimators (EMA, EWMA vol, and optional corr) ---
        # We compute a "market" proxy return as the mean log-return over symbols we can update.
        market_rets: List[float] = []
        updated_syms: List[Tuple[str, float, float]] = []  # (sym, px, ret)

        for sym, px in snap.prices.items():
            try:
                p = float(px)
            except (TypeError, ValueError):
                continue
            if not _is_good_px(p):
                continue

            st = self._ensure_state(sym)
            st.prices.append(p)

            # EMA update
            a_fast = 2.0 / (max(1, int(self.ema_fast_mins)) + 1.0)
            a_slow = 2.0 / (max(1, int(self.ema_slow_mins)) + 1.0)
            if not math.isfinite(st.ema_fast):
                st.ema_fast = p
            else:
                st.ema_fast = a_fast * p + (1.0 - a_fast) * st.ema_fast

            if not math.isfinite(st.ema_slow):
                st.ema_slow = p
            else:
                st.ema_slow = a_slow * p + (1.0 - a_slow) * st.ema_slow

            # Return update
            r = float("nan")
            if math.isfinite(st.prev_px):
                r = _safe_logret(p, st.prev_px)
            st.prev_px = p

            if math.isfinite(r):
                updated_syms.append((sym, p, r))
                market_rets.append(r)

        mkt_r = _mean(market_rets) if market_rets else float("nan")

        # EWMA updates for vol + corr
        a_vol = _alpha_from_halflife(self.vol_halflife_mins)
        a_corr = _alpha_from_halflife(self.corr_halflife_mins)
        if math.isfinite(mkt_r):
            # Update global market variance estimate inside each state (so we can compute corr)
            for sym, _, r in updated_syms:
                st = self._states[sym]

                # EWMA variance of symbol returns
                st.ewma_var = (1.0 - a_vol) * st.ewma_var + a_vol * (r * r)

                # EWMA covariance with market + market variance
                st.ewma_cov_mkt = (1.0 - a_corr) * st.ewma_cov_mkt + a_corr * (r * mkt_r)
                st.ewma_var_mkt = (1.0 - a_corr) * st.ewma_var_mkt + a_corr * (mkt_r * mkt_r)
        else:
            # If market proxy not available, at least update symbol vol
            for sym, _, r in updated_syms:
                st = self._states[sym]
                st.ewma_var = (1.0 - a_vol) * st.ewma_var + a_vol * (r * r)

        # --- Stoploss overlay (trailing stop) ---
        # If stop triggers for any held symbol, we immediately publish updated weights.
        stop_triggered = False
        positions = dict(getattr(portfolio, "positions", {}) or {})
        if self.stoploss_pct and self.stoploss_pct > 0:
            for sym, qty in positions.items():
                if int(qty) <= 0:
                    continue
                st = self._states.get(sym)
                if st is None or not st.prices:
                    continue
                px = st.prices[-1]

                # Track peak since held
                if not st.was_held:
                    st.peak_px_while_held = px
                    st.was_held = True
                else:
                    st.peak_px_while_held = max(st.peak_px_while_held, px)

                if self._ticks < st.cooldown_until_tick:
                    # Already in cooldown; enforce flat
                    if self._cur_target_weights.get(sym, 0.0) != 0.0:
                        self._cur_target_weights[sym] = 0.0
                        stop_triggered = True
                    continue

                peak = st.peak_px_while_held
                if peak > 0 and px <= peak * (1.0 - self.stoploss_pct):
                    # Trigger stop: set to zero and start cooldown
                    st.cooldown_until_tick = self._ticks + max(1, int(self.stop_cooldown_minutes))
                    self._cur_target_weights[sym] = 0.0
                    stop_triggered = True

        # Reset was_held for symbols not held anymore
        held_syms = {sym for sym, qty in positions.items() if int(qty) > 0}
        for sym, st in self._states.items():
            if st.was_held and sym not in held_syms:
                st.was_held = False
                st.peak_px_while_held = 0.0

        if stop_triggered:
            # Ensure we also liquidate any symbol not in current targets
            # by including all held symbols explicitly.
            for sym in held_syms:
                self._cur_target_weights.setdefault(sym, 0.0)

            publish = {sym: w for sym, w in self._cur_target_weights.items() if abs(w) > 1e-9}
            # Even if publish becomes empty, we still want to sell existing holdings.
            # So publish held symbols with 0 weights.
            for sym in held_syms:
                publish.setdefault(sym, 0.0)

            if publish:
                self._set_publish_weights(publish)
            return []

        # --- Rebalance schedule ---
        if self._ticks % max(1, int(self.rebalance_every_minutes)) != 0:
            return []

        # --- Compute cross-sectional scores ---
        scored: List[Tuple[str, float, float]] = []  # (sym, score, vol)

        for sym, st in self._states.items():
            prices = st.prices
            if len(prices) < self.mom_lookback_mins + 1:
                continue

            # Momentum over finite window
            p0 = prices[0]
            p1 = prices[-1]
            mom = _safe_logret(p1, p0)
            if not math.isfinite(mom) or mom <= self.min_signal:
                continue

            # EMA trend filter
            if not (math.isfinite(st.ema_fast) and math.isfinite(st.ema_slow)):
                continue
            if st.ema_fast <= st.ema_slow * (1.0 + self.ema_band):
                continue

            # Vol estimate (EWMA)
            vol = math.sqrt(max(0.0, st.ewma_var))
            if not math.isfinite(vol) or vol <= 0.0:
                continue

            # Optional correlation penalty (diversification bias)
            penalty = 0.0
            if self.corr_penalty and self.corr_penalty > 0.0:
                den = math.sqrt(max(1e-12, st.ewma_var * st.ewma_var_mkt))
                if den > 0 and math.isfinite(den):
                    corr = st.ewma_cov_mkt / den
                    if math.isfinite(corr):
                        penalty = self.corr_penalty * abs(corr)

            score = (mom / (vol + 1e-12)) - penalty
            if math.isfinite(score) and score > 0.0:
                # Respect stoploss cooldown: don't select if in cooldown
                if self._ticks < st.cooldown_until_tick:
                    continue
                scored.append((sym, score, vol))

        if not scored:
            # Go cash / liquidate existing positions
            w_cash = self._go_cash_weights(portfolio)
            self._cur_target_weights = dict(w_cash)
            if w_cash:
                self._set_publish_weights(w_cash)
            return []

        # Pick top-K by score
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[: self.target_count]

        # Raw weights proportional to score / vol (extra vol scaling)
        raw: Dict[str, float] = {}
        for sym, score, vol in top:
            w = score / (vol + 1e-12)
            if math.isfinite(w) and w > 0.0:
                raw[sym] = w

        if not raw:
            w_cash = self._go_cash_weights(portfolio)
            self._cur_target_weights = dict(w_cash)
            if w_cash:
                self._set_publish_weights(w_cash)
            return []

        gross_target = max(0.0, self.max_gross_leverage * (1.0 - self.cash_buffer))
        s = sum(raw.values())
        w_new = {sym: (gross_target * v / s) for sym, v in raw.items()}

        # Cap per-name
        for sym in list(w_new.keys()):
            w_new[sym] = min(w_new[sym], self.max_weight)

        # Renormalize after caps
        s2 = sum(w_new.values())
        if s2 > 0 and s2 > gross_target:
            scale = gross_target / s2
            for sym in w_new:
                w_new[sym] *= scale

        # Turnover cap (relative to last intended weights)
        w_old = self._cur_target_weights or {}
        all_syms = set(w_old) | set(w_new)
        turnover = 0.0
        for sym in all_syms:
            turnover += abs(w_new.get(sym, 0.0) - w_old.get(sym, 0.0))

        if self.max_turnover > 0 and turnover > self.max_turnover:
            alpha = self.max_turnover / max(1e-12, turnover)
            w_blend: Dict[str, float] = {}
            for sym in all_syms:
                w_blend[sym] = w_old.get(sym, 0.0) + alpha * (w_new.get(sym, 0.0) - w_old.get(sym, 0.0))
            w_new = {k: v for k, v in w_blend.items() if abs(v) > 1e-6}

        # Persist intended weights for next turnover calc
        self._cur_target_weights = dict(w_new)

        # Publish weights for this tick only
        # Also include current holdings with 0 weights so we actually liquidate dropped names.
        for sym, qty in positions.items():
            if int(qty) != 0:
                w_new.setdefault(sym, 0.0)

        if w_new:
            self._set_publish_weights(w_new)

        return []
